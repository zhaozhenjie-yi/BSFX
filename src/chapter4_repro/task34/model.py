from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from chapter4_repro.embedding.aligned_hpe import FormulaAlignedHierarchicalPositionalEncoder
from chapter4_repro.embedding.pretrained import initialize_word_embedding_from_artifact
from chapter4_repro.models.multilevel_classifier import Task32MultilevelClassifier
from chapter4_repro.models.paper_dwaf import (
    PaperCNNChannelBranch,
    PaperGRUTokenBranch,
    PaperLikeDWAF,
)
from chapter4_repro.models.structure_sensitive_transformer import (
    SPECIAL_TOKEN_PAD,
    SPECIAL_TOKEN_TYPE_COUNT,
    RESIDUAL_STANDARD,
    StructuralTransformerInput,
    StructureSensitiveTransformerConfig,
    StructureSensitiveTransformerEncoder,
)


PAPER_REPRO_V1 = "PAPER_REPRO_V1"
ENGINEERING_ALIGNED_V1 = "ENGINEERING_ALIGNED_V1"
BCE_SINGLE_SIGMOID_V1 = "BCE_SINGLE_SIGMOID_V1"
CE_TWO_LOGIT_V1 = "CE_TWO_LOGIT_V1"
TASK34_ARCHITECTURE_ID = "task34-dual-profile-full-chain-v1"


def _masked_softmax(scores: Tensor, valid_mask: Tensor, *, dim: int = -1) -> Tensor:
    if scores.shape != valid_mask.shape:
        valid_mask = valid_mask.expand_as(scores)
    scores_fp32 = scores.float().masked_fill(~valid_mask, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(scores_fp32, dim=dim)
    probabilities = probabilities * valid_mask.to(probabilities.dtype)
    denominator = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(probabilities),
    ).to(scores.dtype)


def _masked_max(values: Tensor, valid_mask: Tensor, *, dim: int) -> tuple[Tensor, Tensor]:
    masked = values.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
    pooled = masked.max(dim=dim).values
    group_valid = valid_mask.any(dim=dim)
    return pooled.masked_fill(~group_valid.unsqueeze(-1), 0.0), group_valid


class AdditiveGRUTokenBranch(nn.Module):
    """PAPER scorer: v^T tanh(W h + b), retaining the token-aligned sequence."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        sequence: Tensor,
        *,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if sequence.ndim != 3 or padding_mask.shape != sequence.shape[:2]:
            raise ValueError("GRU sequence/padding shapes must be [N,L,H] and [N,L]")
        lengths = (~padding_mask).sum(dim=1)
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        if not torch.equal(padding_mask, positions >= lengths.unsqueeze(1)):
            raise ValueError("Task34 GRU requires right padding")
        packed = pack_padded_sequence(
            sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0),
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        output = self.dropout(output).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        scores = self.attention_vector(torch.tanh(self.attention_projection(output))).squeeze(-1)
        attention = _masked_softmax(scores, ~padding_mask)
        weighted = output * attention.unsqueeze(-1)
        return weighted, attention, weighted.sum(dim=1)


class PaperAbsoluteRelationBias(nn.Module):
    """Frozen PAPER interpretation of formula 4-37 with absolute scalar terms."""

    def __init__(self, *, num_heads: int, depth_initial: float = 1.0, order_initial: float = 0.25) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.raw_depth = nn.Parameter(torch.tensor(math.log(math.expm1(depth_initial))))
        self.raw_order = nn.Parameter(torch.tensor(math.log(math.expm1(order_initial))))

    @staticmethod
    def _expand(value: Tensor, heads: int) -> Tensor:
        return value.unsqueeze(1).expand(-1, heads, -1, -1)

    def forward(self, structural_input: StructuralTransformerInput) -> dict[str, Tensor]:
        valid = structural_input.valid_pair_mask()
        depth_delta = (
            structural_input.depth_ids.unsqueeze(-1) - structural_input.depth_ids.unsqueeze(-2)
        ).abs().float()
        order_delta = (
            structural_input.sibling_order_ids.unsqueeze(-1)
            - structural_input.sibling_order_ids.unsqueeze(-2)
        ).abs().float()
        same_segment = structural_input.segment_ids.unsqueeze(-1).eq(
            structural_input.segment_ids.unsqueeze(-2)
        ).float()
        ref_related = structural_input.ref_link_matrix.gt(0).float()
        pieces = {
            "depth_bias": -F.softplus(self.raw_depth) * depth_delta,
            "sibling_bias": -F.softplus(self.raw_order) * order_delta,
            "same_segment_bias": 2.0 * same_segment,
            "ref_bias": ref_related,
        }
        valid_heads = valid.unsqueeze(1)
        return {
            name: self._expand(value, self.num_heads).masked_fill(~valid_heads, 0.0)
            for name, value in pieces.items()
        }

    def parameter_count(self) -> dict[str, int]:
        return {
            "depth_bias": 1,
            "sibling_bias": 1,
            "same_segment_bias": 0,
            "ref_bias": 0,
        }


class PaperSingleSigmoidClassifier(nn.Module):
    """Single-logit segment predictions and probability attention for PAPER."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.segment_classifier = nn.Linear(hidden_dim, 1)
        self.probability_attention_weight = nn.Parameter(torch.ones(()))
        self.probability_attention_bias = nn.Parameter(torch.zeros(()))
        self.epsilon = float(epsilon)

    def forward(
        self,
        hierarchy: Tensor,
        token_valid_mask: Tensor,
        *,
        path_valid_mask: Tensor | None = None,
        segment_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if hierarchy.ndim != 5 or token_valid_mask.shape != hierarchy.shape[:-1]:
            raise ValueError("PAPER classifier requires [B,S,P,T,D] and [B,S,P,T]")
        path_vectors, actual_paths = _masked_max(hierarchy, token_valid_mask, dim=3)
        if path_valid_mask is not None:
            actual_paths = actual_paths & path_valid_mask
        path_vectors = path_vectors.masked_fill(~actual_paths.unsqueeze(-1), 0.0)
        segment_vectors, actual_segments = _masked_max(path_vectors, actual_paths, dim=2)
        if segment_valid_mask is not None:
            actual_segments = actual_segments & segment_valid_mask
        if not bool(actual_segments.any(dim=1).all()):
            raise ValueError("every sample requires a valid segment")
        segment_vectors = segment_vectors.masked_fill(~actual_segments.unsqueeze(-1), 0.0)
        segment_logits = self.segment_classifier(self.dropout(segment_vectors)).squeeze(-1)
        segment_logits = segment_logits.masked_fill(~actual_segments, 0.0)
        segment_probability = torch.sigmoid(segment_logits.float()).masked_fill(
            ~actual_segments, 0.0
        )
        scores = (
            self.probability_attention_weight.float() * torch.tanh(segment_probability)
            + self.probability_attention_bias.float()
        )
        weights = _masked_softmax(scores, actual_segments)
        final_probability = (weights * segment_probability).sum(dim=-1)
        safe_probability = final_probability.clamp(self.epsilon, 1.0 - self.epsilon)
        final_logit = torch.log(safe_probability) - torch.log1p(-safe_probability)
        return {
            "final_logits": final_logit,
            "final_probability": final_probability,
            "segment_logits": segment_logits,
            "segment_probabilities": segment_probability,
            "segment_attention_scores": scores.masked_fill(~actual_segments, 0.0),
            "segment_attention_weights": weights,
            "segment_valid_mask": actual_segments,
            "path_valid_mask": actual_paths,
        }


@dataclass
class Task34IntegratedOutput:
    family_id: str
    loss_protocol: str
    final_logits: Tensor
    final_probability: Tensor
    segment_logits: Tensor
    segment_probabilities: Tensor
    segment_attention_weights: Tensor
    hierarchy: Tensor
    token_valid_mask: Tensor
    path_valid_mask: Tensor
    segment_valid_mask: Tensor
    audit: dict[str, Any]


class Task34IntegratedModel(nn.Module):
    """X1 -> H1 -> GRU/CNN -> D1 -> family ST -> family classifier."""

    architecture_id = TASK34_ARCHITECTURE_ID

    def __init__(
        self,
        *,
        family_id: str,
        vocab_size: int,
        padding_idx: int = 0,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        d_model: int = 512,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        classifier_dropout: float | None = None,
        residual_branch_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if family_id not in {PAPER_REPRO_V1, ENGINEERING_ALIGNED_V1}:
            raise ValueError(f"unsupported Task34 family: {family_id!r}")
        if d_model != 512 or transformer_layers != 6 or transformer_heads != 8:
            raise ValueError("Task33 freezes Transformer at 6 layers, 8 heads, d_model=512")
        if dim_feedforward != 3072:
            raise ValueError("Task33 freezes Transformer FFN width at 3072")
        effective_classifier_dropout = (
            float(dropout)
            if classifier_dropout is None
            else float(classifier_dropout)
        )
        if not 0.0 <= effective_classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0,1)")
        self.family_id = family_id
        self.loss_protocol = (
            BCE_SINGLE_SIGMOID_V1 if family_id == PAPER_REPRO_V1 else CE_TWO_LOGIT_V1
        )
        self.padding_idx = int(padding_idx)
        self.hpe = FormulaAlignedHierarchicalPositionalEncoder(
            vocab_size=vocab_size,
            word_dim=128,
            struct_dim=64,
            output_dim=64,
            padding_idx=padding_idx,
        )
        self.h1_to_d1 = nn.Linear(64, 128)
        self.gru_branch: nn.Module = (
            AdditiveGRUTokenBranch(128, dropout=0.0)
            if family_id == PAPER_REPRO_V1
            else PaperGRUTokenBranch(128, dropout=0.0)
        )
        self.cnn_branch = PaperCNNChannelBranch(128, kernel_size=3, dropout=0.0)
        self.dwaf = PaperLikeDWAF(hidden_dim=128, num_heads=8, attention_dropout=0.0)
        self.dwaf.set_observability(True)
        self.d1_to_transformer = nn.Linear(128, d_model)
        self.special_token_embedding = nn.Embedding(
            SPECIAL_TOKEN_TYPE_COUNT,
            d_model,
            padding_idx=SPECIAL_TOKEN_PAD,
        )
        transformer_config = StructureSensitiveTransformerConfig(
            profile_id="ST-PAPER" if family_id == PAPER_REPRO_V1 else "ST-ENG",
            structural_bias_enabled=True,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_dropout=dropout,
            max_depth_delta=16,
            max_sibling_delta=16,
            residual_protocol=RESIDUAL_STANDARD,
            residual_branch_scale=residual_branch_scale,
        )
        self.transformer = StructureSensitiveTransformerEncoder(transformer_config)
        if family_id == PAPER_REPRO_V1:
            for layer in self.transformer.layers:
                layer.self_attention.relation_bias = PaperAbsoluteRelationBias(
                    num_heads=transformer_heads
                )
            self.classifier: nn.Module = PaperSingleSigmoidClassifier(
                d_model,
                dropout=effective_classifier_dropout,
            )
        else:
            self.classifier = Task32MultilevelClassifier(
                d_model,
                dropout=effective_classifier_dropout,
            )
        self.embedding_metadata: dict[str, Any] = {"type": "trainable_random"}

    def initialize_paper_cbow(
        self,
        *,
        artifact_dir: str,
        ordered_vocab_tokens: Sequence[str],
        special_tokens: Sequence[str],
        freeze: bool = False,
    ) -> dict[str, Any]:
        if self.family_id != PAPER_REPRO_V1:
            raise ValueError("CBOW initialization is only allowed for PAPER_REPRO_V1")
        self.embedding_metadata = initialize_word_embedding_from_artifact(
            self.hpe.word_embedding,
            artifact_dir=artifact_dir,
            ordered_vocab_tokens=ordered_vocab_tokens,
            special_tokens=special_tokens,
            freeze=freeze,
        )
        return dict(self.embedding_metadata)

    @staticmethod
    def _require_batch(batch: Mapping[str, Any]) -> None:
        required = {"labels"}
        alternatives = {
            "token_ids": "packed_token_ids",
            "depth_ids": "packed_depth_ids",
            "sibling_ids": "packed_sibling_order_ids",
            "global_pos_ids": "packed_global_position_ids",
            "local_pos_ids": "packed_local_position_ids",
            "delta_seg": "packed_delta_seg",
            "ref_identity_ids": "packed_ref_identity_ids",
            "ref_definition_token_ids": "packed_ref_definition_token_ids",
            "padding_mask": "packed_padding_mask",
            "sibling_parent_ids": "packed_sibling_parent_ids",
            "ast_node_identity_ids": "packed_ast_node_identity_ids",
            "segment_ids": "packed_segment_ids",
            "special_token_ids": "packed_special_token_ids",
            "ref_link_matrix": "packed_ref_link_matrix",
            "path_to_sample": "packed_path_to_sample",
            "path_slot": "packed_path_slot",
        }
        missing = sorted(required - set(batch))
        missing.extend(
            name for name, packed in alternatives.items() if name not in batch and packed not in batch
        )
        if missing:
            raise KeyError(f"Task34 batch is missing keys: {missing}")

    @staticmethod
    def _field(batch: Mapping[str, Any], name: str, packed_name: str) -> Any:
        # Real Task34 collate intentionally exposes both canonical [B,S,P,T]
        # and packed-valid-path [N,L] views.  H1/DWAF/ST must consume only the
        # packed view so invalid path/segment padding never reaches an encoder.
        return batch[packed_name] if packed_name in batch else batch[name]

    @staticmethod
    def _scatter_hierarchy(
        encoded: Tensor,
        *,
        padding_mask: Tensor,
        segment_ids: Tensor,
        path_to_sample: Tensor,
        path_slot: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        valid = ~padding_mask
        if not bool(valid.any(dim=1).all()):
            raise ValueError("packed Task34 paths must all be nonempty")
        max_segments = int(segment_ids.masked_select(valid).max().item()) + 1
        max_paths = int(path_slot.max().item()) + 1
        max_tokens = 1
        for row in range(encoded.shape[0]):
            for segment in torch.unique(segment_ids[row][valid[row]]).tolist():
                max_tokens = max(
                    max_tokens,
                    int((valid[row] & segment_ids[row].eq(int(segment))).sum().item()),
                )
        hierarchy = encoded.new_zeros(
            (batch_size, max_segments, max_paths, max_tokens, encoded.shape[-1])
        )
        token_valid = torch.zeros(
            (batch_size, max_segments, max_paths, max_tokens),
            dtype=torch.bool,
            device=encoded.device,
        )
        for row in range(encoded.shape[0]):
            sample = int(path_to_sample[row].item())
            path = int(path_slot[row].item())
            for segment_value in torch.unique(segment_ids[row][valid[row]]).tolist():
                segment = int(segment_value)
                indices = valid[row] & segment_ids[row].eq(segment)
                values = encoded[row, indices]
                length = int(values.shape[0])
                hierarchy[sample, segment, path, :length] = values
                token_valid[sample, segment, path, :length] = True
        path_valid = token_valid.any(dim=3)
        segment_valid = path_valid.any(dim=2)
        return hierarchy, token_valid, path_valid, segment_valid

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_audit: bool = False,
    ) -> Task34IntegratedOutput:
        self._require_batch(batch)
        token_ids = self._field(batch, "token_ids", "packed_token_ids")
        depth_ids = self._field(batch, "depth_ids", "packed_depth_ids")
        sibling_ids = self._field(batch, "sibling_ids", "packed_sibling_order_ids")
        global_pos_ids = self._field(
            batch, "global_pos_ids", "packed_global_position_ids"
        )
        local_pos_ids = self._field(batch, "local_pos_ids", "packed_local_position_ids")
        delta_seg = self._field(batch, "delta_seg", "packed_delta_seg")
        ref_identity_ids = self._field(
            batch, "ref_identity_ids", "packed_ref_identity_ids"
        )
        ref_definition_token_ids = self._field(
            batch,
            "ref_definition_token_ids",
            "packed_ref_definition_token_ids",
        )
        padding_mask = self._field(batch, "padding_mask", "packed_padding_mask").bool()
        sibling_parent_ids = self._field(
            batch, "sibling_parent_ids", "packed_sibling_parent_ids"
        )
        ast_node_identity_ids = self._field(
            batch, "ast_node_identity_ids", "packed_ast_node_identity_ids"
        )
        segment_ids = self._field(batch, "segment_ids", "packed_segment_ids")
        special_token_ids = self._field(
            batch, "special_token_ids", "packed_special_token_ids"
        )
        ref_link_matrix = self._field(
            batch, "ref_link_matrix", "packed_ref_link_matrix"
        )
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = batch.get("packed_attention_mask")
        path_to_sample = self._field(
            batch, "path_to_sample", "packed_path_to_sample"
        )
        path_slot = self._field(batch, "path_slot", "packed_path_slot")
        h1 = self.hpe(
            token_ids,
            depth_ids,
            sibling_ids,
            global_pos_ids,
            local_pos_ids,
            delta_seg=delta_seg,
            ref_identity_ids=ref_identity_ids,
            ref_definition_token_ids=ref_definition_token_ids,
            padding_mask=padding_mask,
        )
        d1_input = self.h1_to_d1(h1).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        gru_tokens, gru_attention, gru_pooled = self.gru_branch(
            d1_input,
            padding_mask=padding_mask,
        )
        cnn_tokens, cnn_gate = self.cnn_branch(d1_input, padding_mask=padding_mask)
        fused, dwaf_attention = self.dwaf(
            d1_input,
            gru_tokens,
            cnn_tokens,
            padding_mask=padding_mask,
        )
        token_embeddings = self.d1_to_transformer(fused)
        token_embeddings = token_embeddings + self.special_token_embedding(
            special_token_ids
        )
        structural_input = StructuralTransformerInput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            depth_ids=depth_ids.long(),
            sibling_order_ids=sibling_ids.long(),
            sibling_parent_ids=sibling_parent_ids.long(),
            ast_node_identity_ids=ast_node_identity_ids.long(),
            segment_ids=segment_ids.long(),
            ref_link_matrix=ref_link_matrix.long(),
            special_token_ids=special_token_ids.long(),
        )
        transformed = self.transformer(
            structural_input,
            return_attention_audit=return_audit,
            audit_layer_indices=[0, 5] if return_audit else None,
            audit_head_indices=[0] if return_audit else None,
            audit_max_samples=1,
        )
        hierarchy, token_valid, path_valid, segment_valid = self._scatter_hierarchy(
            transformed.last_hidden_state,
            padding_mask=padding_mask,
            segment_ids=segment_ids,
            path_to_sample=path_to_sample,
            path_slot=path_slot,
            batch_size=int(batch["labels"].shape[0]),
        )
        if self.family_id == PAPER_REPRO_V1:
            classified = self.classifier(
                hierarchy,
                token_valid,
                path_valid_mask=path_valid,
                segment_valid_mask=segment_valid,
            )
            classifier_audit: dict[str, Any] = {}
        else:
            c1 = self.classifier(
                hierarchy,
                token_valid,
                path_valid_mask=path_valid,
                segment_valid_mask=segment_valid,
                return_multilevel_audit=return_audit,
                audit_max_samples=1,
            )
            classified = {
                "final_logits": c1.final_logits,
                "final_probability": c1.final_probability,
                "segment_logits": c1.segment_logits,
                "segment_probabilities": c1.segment_probabilities,
                "segment_attention_weights": c1.segment_attention_weights,
                "path_valid_mask": c1.path_valid_mask,
                "segment_valid_mask": c1.segment_valid_mask,
            }
            classifier_audit = c1.audit or {}
        audit: dict[str, Any] = {}
        if return_audit:
            hpe_observed = self.hpe.observability()
            audit = {
                "path_count": int(encoded_paths := transformed.last_hidden_state.shape[0]),
                "segment_count": int(segment_valid.sum().item()),
                "ref_count": int(ref_link_matrix.gt(0).sum().item()),
                "packed_path_count": int(encoded_paths),
                "hierarchy_shape": list(hierarchy.shape),
                "hpe_vector_gate": hpe_observed["vector_gate"][:1].detach().cpu(),
                "gru_attention": gru_attention[:1].detach().cpu(),
                "gru_pooled": gru_pooled[:1].detach().cpu(),
                "cnn_channel_gate": cnn_gate[:1].detach().cpu(),
                "dwaf_attention": dwaf_attention[:1].detach().cpu(),
                "structural_bias": transformed.attention_audit,
                "segment_probabilities": classified["segment_probabilities"][:1].detach().cpu(),
                "segment_attention_weights": classified["segment_attention_weights"][:1].detach().cpu(),
                "classifier": classifier_audit,
            }
        return Task34IntegratedOutput(
            family_id=self.family_id,
            loss_protocol=self.loss_protocol,
            final_logits=classified["final_logits"],
            final_probability=classified["final_probability"],
            segment_logits=classified["segment_logits"],
            segment_probabilities=classified["segment_probabilities"],
            segment_attention_weights=classified["segment_attention_weights"],
            hierarchy=hierarchy,
            token_valid_mask=token_valid,
            path_valid_mask=classified["path_valid_mask"],
            segment_valid_mask=classified["segment_valid_mask"],
            audit=audit,
        )


def _profile_mapping(profile: Any) -> Mapping[str, Any]:
    if isinstance(profile, Mapping):
        return profile
    if hasattr(profile, "resolved") and isinstance(profile.resolved, Mapping):
        return profile.resolved
    if hasattr(profile, "payload") and isinstance(profile.payload, Mapping):
        return profile.payload
    if hasattr(profile, "raw") and isinstance(profile.raw, Mapping):
        return profile.raw
    raise TypeError("profile must be a mapping or resolved profile object")


def build_task34_model(
    profile: Any,
    *,
    vocab_size: int,
    padding_idx: int = 0,
    ordered_vocab_tokens: Sequence[str] | None = None,
    special_tokens: Sequence[str] | None = None,
    cbow_artifact_dir: str | None = None,
    dropout: float = 0.1,
    classifier_dropout: float | None = None,
    residual_branch_scale: float = 1.0,
) -> Task34IntegratedModel:
    payload = _profile_mapping(profile)
    family = str(payload.get("family_id") or payload.get("profile_id") or "").upper()
    modules = payload.get("modules", {})
    model = Task34IntegratedModel(
        family_id=family,
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        transformer_layers=int(modules.get("transformer_layers", 6)),
        transformer_heads=int(modules.get("transformer_heads", 8)),
        d_model=512,
        dim_feedforward=int(modules.get("transformer_ffn_dim", 3072)),
        dropout=dropout,
        classifier_dropout=classifier_dropout,
        residual_branch_scale=residual_branch_scale,
    )
    model.architecture_id = str(payload.get("architecture_id") or TASK34_ARCHITECTURE_ID)
    if family == PAPER_REPRO_V1:
        if cbow_artifact_dir is None or ordered_vocab_tokens is None or special_tokens is None:
            raise ValueError("PAPER_REPRO_V1 requires a dataset-specific train-only CBOW artifact")
        model.initialize_paper_cbow(
            artifact_dir=cbow_artifact_dir,
            ordered_vocab_tokens=ordered_vocab_tokens,
            special_tokens=special_tokens,
            freeze=not bool(modules.get("cbow_fine_tune", True)),
        )
    elif cbow_artifact_dir is not None:
        raise ValueError("ENGINEERING_ALIGNED_V1 must not load a CBOW artifact")
    return model
