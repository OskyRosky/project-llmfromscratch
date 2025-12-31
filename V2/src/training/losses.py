# src/training/losses.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def language_modeling_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    loss_mask: Optional[Tensor] = None,
    pad_id: Optional[int] = None,
) -> Tensor:
    """
    Cross-entropy promedio por token.

    Compatible con versión anterior:
      language_modeling_loss(logits, targets, ignore_index=-100)

    Nuevo (para instruction tuning token-level):
      language_modeling_loss(logits, targets, loss_mask=mask, pad_id=pad_id)

    Args:
      logits:   (B, T, V)
      targets:  (B, T) token ids
      ignore_index: usado solo cuando loss_mask is None (modo "legacy")
      loss_mask: (B, T) bool. True => computar loss en esa posición.
                 (Ej: True solo para tokens de respuesta después de <resp>)
      pad_id: si se provee, excluye explícitamente targets==pad_id (extra safety)

    Returns:
      Scalar tensor (mean loss)
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B,T,V). Got shape={tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must be (B,T). Got shape={tuple(targets.shape)}")

    # dtype correcto para CE
    if targets.dtype != torch.long:
        targets = targets.long()

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must match logits (B,T)=({B},{T})"
        )

    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)

    # ---------------------------------------------------------
    # Legacy mode: compute over all tokens (optionally ignoring ignore_index)
    # ---------------------------------------------------------
    if loss_mask is None:
        # Si pad_id está presente y quieren ignorar padding, se puede usar como ignore_index
        # sin romper el comportamiento previo (por default ignore_index=-100).
        _ignore = int(pad_id) if pad_id is not None else int(ignore_index)

        return F.cross_entropy(
            logits_2d,
            targets_1d,
            ignore_index=_ignore,
            reduction="mean",
        )

    # ---------------------------------------------------------
    # Masked mode: compute CE ONLY where loss_mask=True (and not pad_id if provided)
    # ---------------------------------------------------------
    if loss_mask.shape != (B, T):
        raise ValueError(f"loss_mask must be (B,T). Got shape={tuple(loss_mask.shape)}")

    mask_1d = loss_mask.reshape(B * T).bool()

    # Extra safety: remove pad positions from loss, even if mask accidentally includes them
    if pad_id is not None:
        mask_1d = mask_1d & (targets_1d != int(pad_id))

    active = int(mask_1d.sum().item())
    if active == 0:
        raise ValueError(
            "Masked LM loss has 0 active tokens. "
            "Check that your dataset mask marks tokens after <resp> and that pad_id is correct."
        )

    per_token = F.cross_entropy(logits_2d, targets_1d, reduction="none")
    return per_token[mask_1d].mean()