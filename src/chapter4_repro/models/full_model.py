from __future__ import annotations

from torch import Tensor, nn

from chapter4_repro.embedding.aligned_hpe import (
    H1_REF_IDENTITY_BUCKETS,
    HPE_MODE_CURRENT_V1,
    HPE_MODE_H1_V1,
    HPE_MODES,
    FormulaAlignedHierarchicalPositionalEncoder,
)
from chapter4_repro.embedding.hpe import HierarchicalPositionalEncoder

from .classifier import MaxPoolClassifier
from .struct_transformer import StructuralTransformerEncoder, build_structural_bias


class Chapter4BaseModel(nn.Module):
    """
    Minimal runnable baseline for:
    semantic embedding + HPE + structural transformer + classifier.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        word_dim: int = 128,
        struct_dim: int = 64,
        output_dim: int = 128,
        global_pos_dim: int = 16,
        local_pos_dim: int = 16,
        depth_dim: int = 32,
        sibling_dim: int = 16,
        alpha: float = 0.3,
        beta: float = 0.5,
        max_depth_bits: int = 8,
        max_global_pos: int = 4096,
        max_local_pos: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        lambda_d: float = 1.0,
        lambda_o: float = 0.25,
        ref_bias_scale: float = 1.0,
        padding_idx: int = 0,
        ref_cue_scale: float = 1.0,
        ref_cue_gate_reg_weight: float = 0.0,
        zero_delta_identity: bool = False,
        hpe_mode: str = HPE_MODE_CURRENT_V1,
        h1_ref_identity_buckets: int = H1_REF_IDENTITY_BUCKETS,
    ) -> None:
        super().__init__()
        if hpe_mode not in HPE_MODES:
            raise ValueError(f"Unsupported hpe_mode={hpe_mode!r}")
        self.hpe_mode = hpe_mode
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.ref_bias_scale = ref_bias_scale
        self.ref_cue_scale = float(ref_cue_scale)
        self.ref_cue_gate_reg_weight = float(ref_cue_gate_reg_weight)
        encoder_args = {
            "vocab_size": vocab_size,
            "word_dim": word_dim,
            "struct_dim": struct_dim,
            "output_dim": output_dim,
            "global_pos_dim": global_pos_dim,
            "local_pos_dim": local_pos_dim,
            "depth_dim": depth_dim,
            "sibling_dim": sibling_dim,
            "alpha": alpha,
            "beta": beta,
            "max_depth_bits": max_depth_bits,
            "max_global_pos": max_global_pos,
            "max_local_pos": max_local_pos,
            "padding_idx": padding_idx,
        }
        if hpe_mode == HPE_MODE_H1_V1:
            self.encoder = FormulaAlignedHierarchicalPositionalEncoder(
                **encoder_args,
                ref_identity_buckets=h1_ref_identity_buckets,
            )
        else:
            self.encoder = HierarchicalPositionalEncoder(
                **encoder_args,
                ref_cue_scale=ref_cue_scale,
                ref_cue_gate_reg_weight=ref_cue_gate_reg_weight,
                zero_delta_identity=zero_delta_identity,
            )
        self.input_projection = (
            nn.Identity() if output_dim == hidden_dim else nn.Linear(output_dim, hidden_dim)
        )
        self.transformer = StructuralTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.classifier = MaxPoolClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def auxiliary_losses(self) -> dict[str, Tensor]:
        return self.encoder.auxiliary_losses()

    def encode_hpe_tokens(
        self,
        *,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        padding_mask: Tensor | None = None,
        delta_seg: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
        ref_identity_ids: Tensor | None = None,
        ref_definition_token_ids: Tensor | None = None,
        explicit_padding_zero: bool = False,
    ) -> Tensor:
        """Return HPE tokens before the structural Transformer.

        ``explicit_padding_zero`` is opt-in so the historical H0/D0 path keeps
        its byte-for-byte topology.  D1 enables it because its GRU/CNN/DWAF
        stages run before the Transformer and therefore require an explicit
        token-mask boundary.
        """
        encoder_args = {
            "token_ids": token_ids,
            "depth_ids": depth_ids,
            "sibling_ids": sibling_ids,
            "global_pos_ids": global_pos_ids,
            "local_pos_ids": local_pos_ids,
            "delta_seg": delta_seg,
            "ref_cue_ids": ref_cue_ids,
        }
        if self.hpe_mode == HPE_MODE_H1_V1:
            encoder_args.update(
                {
                    "ref_identity_ids": ref_identity_ids,
                    "ref_definition_token_ids": ref_definition_token_ids,
                    "padding_mask": padding_mask,
                }
            )
        x = self.input_projection(self.encoder(**encoder_args))
        if explicit_padding_zero and padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask must match HPE token batch/time dimensions")
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def transform_tokens(
        self,
        x: Tensor,
        *,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        segment_ids: Tensor,
        padding_mask: Tensor | None = None,
        ref_matrix: Tensor | None = None,
    ) -> Tensor:
        """Apply the structural Transformer to an already encoded token sequence."""
        bias = build_structural_bias(
            depth_ids=depth_ids,
            sibling_ids=sibling_ids,
            segment_ids=segment_ids,
            lambda_d=self.lambda_d,
            lambda_o=self.lambda_o,
            ref_bias_scale=self.ref_bias_scale,
            ref_matrix=ref_matrix,
        )
        return self.transformer(x, padding_mask=padding_mask, structural_bias=bias)

    def encode_segments(
        self,
        *,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        segment_ids: Tensor,
        padding_mask: Tensor | None = None,
        delta_seg: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
        ref_identity_ids: Tensor | None = None,
        ref_definition_token_ids: Tensor | None = None,
        ref_matrix: Tensor | None = None,
    ) -> Tensor:
        x = self.encode_hpe_tokens(
            token_ids=token_ids,
            depth_ids=depth_ids,
            sibling_ids=sibling_ids,
            global_pos_ids=global_pos_ids,
            local_pos_ids=local_pos_ids,
            padding_mask=padding_mask,
            delta_seg=delta_seg,
            ref_cue_ids=ref_cue_ids,
            ref_identity_ids=ref_identity_ids,
            ref_definition_token_ids=ref_definition_token_ids,
            explicit_padding_zero=False,
        )
        return self.transform_tokens(
            x,
            depth_ids=depth_ids,
            sibling_ids=sibling_ids,
            segment_ids=segment_ids,
            padding_mask=padding_mask,
            ref_matrix=ref_matrix,
        )

    def pool_segments(self, encoded_segments: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if padding_mask is not None:
            masked = encoded_segments.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            return pooled.masked_fill(~pooled.isfinite(), 0.0)
        return encoded_segments.max(dim=1).values

    def forward(
        self,
        *,
        token_ids: Tensor,
        depth_ids: Tensor,
        sibling_ids: Tensor,
        global_pos_ids: Tensor,
        local_pos_ids: Tensor,
        segment_ids: Tensor,
        padding_mask: Tensor | None = None,
        delta_seg: Tensor | None = None,
        ref_cue_ids: Tensor | None = None,
        ref_identity_ids: Tensor | None = None,
        ref_definition_token_ids: Tensor | None = None,
        ref_matrix: Tensor | None = None,
    ) -> Tensor:
        encoded = self.encode_segments(
            token_ids=token_ids,
            depth_ids=depth_ids,
            sibling_ids=sibling_ids,
            global_pos_ids=global_pos_ids,
            local_pos_ids=local_pos_ids,
            segment_ids=segment_ids,
            padding_mask=padding_mask,
            delta_seg=delta_seg,
            ref_cue_ids=ref_cue_ids,
            ref_identity_ids=ref_identity_ids,
            ref_definition_token_ids=ref_definition_token_ids,
            ref_matrix=ref_matrix,
        )
        return self.classifier(encoded, padding_mask=padding_mask)
