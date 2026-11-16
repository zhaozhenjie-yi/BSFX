"""Versioned inference-only handling of explicitly unavailable context branches.

State-dict keys and learned tensors are unchanged. Missing branches are bypassed
and reported; no synthetic empty context is passed to a learned encoder.
"""
import torch
from semantic_encoder import ContextV2Model
from semantic_encoder_v3 import ContextV3Model

class MissingAwareV2(ContextV2Model):
    def forward(self,batch,context):
        output=self.base(batch)
        zero=torch.zeros_like(output.final_logits)
        legacy=self.context(context['legacy']) if context['availability']['legacy'] else zero
        semantic=self.semantic(context['semantics']) if context['availability']['v2'] else zero
        return dict(logits=output.final_logits+self.gate*legacy+self.semantic_gate*semantic,
            base_logits=output.final_logits,base_unclipped_probability=output.final_probability,
            context_delta=legacy,semantic_delta=semantic)

class MissingAwareV3(ContextV3Model):
    def forward(self,batch,context):
        old=self.legacy(batch,context['v2'])
        delta=self.v3(context['v3']) if self.gate and context['v2']['availability']['v3'] else torch.zeros_like(old['logits'])
        return dict(old,legacy_logits=old['logits'],logits=old['logits']+self.gate*delta,v3_delta=delta)
