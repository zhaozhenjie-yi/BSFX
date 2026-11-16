from __future__ import annotations

import torch
from torch import Tensor, nn


class MaxPoolClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if padding_mask is not None:
            masked = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        else:
            pooled = x.max(dim=1).values
        return self.output(self.dropout(pooled))
