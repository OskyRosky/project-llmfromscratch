# src/training/losses.py

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def language_modeling_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """
    Cross-entropy promedio por token.

    logits:  (B, T, V)
    targets: (B, T)  (token ids) o (B, T) con ignore_index para padding/masking

    Devuelve: scalar tensor (mean loss)
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,T,V). Got shape={tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must be (B,T). Got shape={tuple(targets.shape)}")

    # Asegurar dtype correcto
    if targets.dtype != torch.long:
        targets = targets.long()

    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)

    # IMPORTANT: reduction="mean" => per-token mean
    loss = F.cross_entropy(
        logits_2d,
        targets_1d,
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss