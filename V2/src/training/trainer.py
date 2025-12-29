# src/training/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Any, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.training.losses import language_modeling_loss
from src.config.training_config import TrainingConfig


class Trainer:
    """
    Entrenador genérico para language modeling (GPT pequeño).

    Asume que:
      - los dataloaders devuelven (input_ids, target_ids) de enteros.
      - el modelo puede devolver:
          a) logits directamente (Tensor 3D: B,T,V)
          b) una tupla/lista que contiene logits en algún lugar
             (por ejemplo (logits, attn) o (loss, logits, extra)).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_cfg: TrainingConfig,
        ckpt_dir: str = "models/checkpoints",
        device: Optional[torch.device] = None,
        grad_clip: Optional[float] = None,
        # Bonus knobs:
        debug_first_batch: bool = False,
        expected_vocab_size: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = train_cfg

        # Dispositivo
        if device is not None:
            self.device = device
        else:
            self.device = train_cfg.resolved_device()

        # Grad clip (usa max_grad_norm si existe en el config)
        if grad_clip is not None:
            self.grad_clip = grad_clip
        else:
            self.grad_clip = getattr(train_cfg, "max_grad_norm", None)

        # Debug
        self.debug_first_batch = bool(debug_first_batch)
        self._debug_printed = False
        self.expected_vocab_size = expected_vocab_size  # si lo pasás, valida V == vocab_size

        # Directorio de checkpoints
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Enviar modelo al dispositivo
        self.model.to(self.device)

    # ---------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------
    def _move_batch_to_device(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return input_ids.to(self.device), target_ids.to(self.device)

    def _extract_logits(self, out: Any) -> Tensor:
        """
        Extrae logits (Tensor 3D: B, T, V) de la salida del modelo.

        IMPORTANTE:
        Antes buscabas "el primer tensor 3D" dentro de una tupla/lista.
        Eso puede agarrar un tensor 3D que NO son logits (por ejemplo hidden states,
        attention maps, etc.), lo cual rompe la loss (y explica losses gigantes).

        Estrategia:
        - Si es Tensor 3D -> ok
        - Si es tuple/list:
            1) Preferir explícitamente un elemento llamado 'logits' si out es dict
            2) Si no, elegir el tensor 3D cuyo último dim (V) sea el más grande
               (normalmente V=vocab_size es el dim más grande versus d_model).
        """
        # Caso: dict-like (algunos modelos devuelven {"logits": ..., ...})
        if isinstance(out, dict):
            if "logits" in out and isinstance(out["logits"], torch.Tensor):
                logits = out["logits"]
                if logits.ndim != 3:
                    raise ValueError(f"out['logits'] is not 3D. shape={tuple(logits.shape)}")
                return logits
            raise ValueError(f"Model output dict has no 'logits' key. keys={list(out.keys())}")

        # Caso 1: el modelo devuelve directamente logits
        if isinstance(out, torch.Tensor):
            if out.ndim != 3:
                raise ValueError(
                    "Salida del modelo es Tensor pero no es 3D (B,T,V). "
                    f"shape={tuple(out.shape)}"
                )
            return out

        # Caso 2: tuple/list -> elegir el mejor candidato
        if isinstance(out, (tuple, list)):
            candidates: List[Tensor] = [x for x in out if isinstance(x, torch.Tensor) and x.ndim == 3]
            if not candidates:
                shapes: List[Union[tuple, str]] = []
                for x in out:
                    if isinstance(x, torch.Tensor):
                        shapes.append(tuple(x.shape))
                    else:
                        shapes.append(type(x).__name__)
                raise ValueError(
                    "No encontré logits 3D (B,T,V) en la salida del modelo. "
                    f"Elementos={shapes}"
                )

            # Preferir el tensor con mayor V (última dimensión).
            # Logits típicamente: V=vocab_size (p.ej 4096)
            # Hidden states típicamente: V=d_model (p.ej 256)
            best = max(candidates, key=lambda t: int(t.shape[-1]))

            # Si se definió expected_vocab_size, validar
            if self.expected_vocab_size is not None:
                V = int(best.shape[-1])
                if V != int(self.expected_vocab_size):
                    # Mostramos info para depurar
                    cand_shapes = [tuple(t.shape) for t in candidates]
                    raise ValueError(
                        f"Extracted 3D tensor has V={V}, but expected_vocab_size={self.expected_vocab_size}. "
                        f"3D candidates={cand_shapes}"
                    )

            return best

        raise TypeError(f"Salida del modelo no soportada: {type(out)}")

    def _maybe_debug_batch(self, input_ids: Tensor, target_ids: Tensor, logits: Tensor) -> None:
        """Imprime una sola vez stats útiles."""
        if not self.debug_first_batch or self._debug_printed:
            return
        self._debug_printed = True

        with torch.no_grad():
            b, t = target_ids.shape
            v = logits.shape[-1]

            msg = []
            msg.append("\n[DEBUG Trainer]")
            msg.append(f"input_ids:  shape={tuple(input_ids.shape)} dtype={input_ids.dtype} min={int(input_ids.min())} max={int(input_ids.max())}")
            msg.append(f"target_ids: shape={tuple(target_ids.shape)} dtype={target_ids.dtype} min={int(target_ids.min())} max={int(target_ids.max())}")
            msg.append(f"logits:     shape={tuple(logits.shape)} dtype={logits.dtype}")
            msg.append(f"logits stats: min={float(logits.min()):.4f} max={float(logits.max()):.4f} mean={float(logits.mean()):.4f} std={float(logits.std()):.4f}")
            msg.append(f"B={b} T={t} V={v}")

            # Mini CE sanity on a slice (debería ~8-12 al inicio si todo está bien)
            logits_2d = logits.reshape(b * t, v)
            targets_1d = target_ids.reshape(b * t).long()
            n = min(1024, targets_1d.numel())
            tmp = F.cross_entropy(logits_2d[:n], targets_1d[:n], reduction="mean")
            msg.append(f"debug CE on first {n} tokens: {float(tmp.item()):.4f}")
            msg.append("[/DEBUG]\n")
            print("\n".join(msg))

    def _step(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Un paso de forward + loss (sin backward ni optimizer).
        Devuelve la loss (scalar tensor).
        """
        out = self.model(input_ids)
        logits = self._extract_logits(out)  # (B, T, V)

        self._maybe_debug_batch(input_ids, target_ids, logits)

        loss = language_modeling_loss(logits, target_ids)
        return loss

    # ---------------------------------------------------------
    #  Paso individual (lo que usará pretrain_gpt / pretrain_gpt_tokens)
    # ---------------------------------------------------------
    def train_step(self, input_ids: Tensor, target_ids: Tensor) -> float:
        """
        Un paso de entrenamiento (un batch):
          - mueve a device
          - forward
          - backward
          - clip grad (si aplica)
          - optimizer.step()

        Devuelve la loss como float.
        """
        self.model.train()
        input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

        self.optimizer.zero_grad(set_to_none=True)
        loss = self._step(input_ids, target_ids)
        loss.backward()

        if self.grad_clip is not None and self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return float(loss.detach().item())

    # ---------------------------------------------------------
    #  Época completa (compatibilidad)
    # ---------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Entrena el modelo sobre un dataloader (una época completa).
        Devuelve la loss media.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in dataloader:
            batch_loss = self.train_step(input_ids, target_ids)
            total_loss += batch_loss
            num_batches += 1

        mean_loss = total_loss / max(1, num_batches)
        return mean_loss

    # ---------------------------------------------------------
    #  Evaluación (BONUS: promedio por token opcional)
    # ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        logger=None,
        log_every: Optional[int] = None,
        max_batches: Optional[int] = None,
        average_by: str = "token",  # "batch" o "token"
    ) -> float:
        """
        Evalúa el modelo sobre un dataloader (sin gradientes).

        average_by:
          - "batch": promedio simple de losses por batch (tu versión original)
          - "token": promedio ponderado por cantidad de tokens (más correcto)
        """
        self.model.eval()

        if len(dataloader) == 0:
            return 0.0

        total_batches = len(dataloader)
        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        total_loss_sum = 0.0  # para average_by="batch"
        total_batches_count = 0

        total_token_loss_sum = 0.0  # suma de (loss_mean * num_tokens)
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            if batch_idx > total_batches:
                break

            input_ids, target_ids = batch
            input_ids, target_ids = self._move_batch_to_device(input_ids, target_ids)

            loss = self._step(input_ids, target_ids)
            loss_val = float(loss.detach().item())

            if average_by == "token":
                # loss_val ya es mean por token del batch, entonces ponderamos por tokens
                num_tok = int(target_ids.numel())
                total_token_loss_sum += loss_val * num_tok
                total_tokens += num_tok
            else:
                total_loss_sum += loss_val
                total_batches_count += 1

            if logger is not None and log_every and batch_idx % log_every == 0:
                progress = 100.0 * batch_idx / total_batches
                logger.info(
                    f"[val {batch_idx}/{total_batches} ({progress:.2f}%)] loss={loss_val:.4f}"
                )

        if average_by == "token":
            return total_token_loss_sum / max(1, total_tokens)
        return total_loss_sum / max(1, total_batches_count)

    # ---------------------------------------------------------
    #  Checkpoints
    # ---------------------------------------------------------
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        global_step: int,
        val_loss: Optional[float] = None,
    ) -> Path:
        """
        Guarda un checkpoint sencillo en self.ckpt_dir.
        """
        ckpt_path = self.ckpt_dir / filename

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "training_config": self.config.__dict__,
        }

        torch.save(payload, ckpt_path)
        return ckpt_path