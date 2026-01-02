# src/infer/answer.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List

import torch

from src.model.gpt import GPTModel, GPTConfig
from src.infer.tokenizer_wrapper import load_tokenizer_and_specials

# Reusar helpers validados
from src.cli.generate_tokens import (
    cfg_from_ckpt_or_fallback,
    build_banned_ids,
    generate,
)


@dataclass(frozen=True)
class Assets:
    model: GPTModel
    tokenizer: object  # HF Tokenizer
    cfg: GPTConfig
    eos_id: Optional[int]
    banned_ids: List[int]
    device: str  # keep original string for clarity


# -------------------------
# Text cleaning (robusto pero simple)
# -------------------------
def _try_fix_mojibake(s: str) -> str:
    """
    Arreglo best-effort para casos tipo 'JosÃ©' -> 'José'.
    No siempre aplica; solo lo intenta si ve patrones típicos.
    """
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def _clean_text(s: str) -> str:
    s = _try_fix_mojibake(s)

    # Normaliza espacios/saltos
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)      # colapsa espacios
    s = re.sub(r"\n{3,}", "\n\n", s)   # colapsa saltos múltiples
    s = s.strip()

    # Arreglos típicos de puntuación
    s = s.replace("..", ".")  # “París..”
    s = re.sub(r"\s+\.", ".", s)  # "hola ." -> "hola."

    return s


# -------------------------
# Config defaults via env
# -------------------------
def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# -------------------------
# Asset loading (cacheado correctamente por parámetros)
# -------------------------
@lru_cache(maxsize=4)
def _load_assets_once(
    meta_path: str,
    ckpt_path: str,
    tokenizer_path: str,
    device: str,
    n_heads_legacy: int,
    forbid_special: int,
    ban_replacement: int,
) -> Assets:
    dev = torch.device(device)

    tokinfo = load_tokenizer_and_specials(meta_path, tokenizer_path)
    tokenizer = tokinfo.tokenizer
    eos_id = tokinfo.eos_id

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=n_heads_legacy)
    cfg = GPTConfig(**cfg_dict)

    model = GPTModel(cfg).to(dev)
    model.load_state_dict(sd, strict=False)
    model.eval()

    banned_ids = build_banned_ids(
        tokenizer=tokenizer,
        special_ids={k: int(v) for k, v in tokinfo.special_ids.items()},
        forbid_special=bool(int(forbid_special)),
        ban_replacement=bool(int(ban_replacement)),
        vocab_size=int(cfg.vocab_size),
    )

    return Assets(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        eos_id=int(eos_id) if eos_id is not None else None,
        banned_ids=banned_ids,
        device=device,
    )


def clear_cache() -> None:
    """Forzar recarga de modelo/tokenizer (útil si cambias checkpoints a mano)."""
    _load_assets_once.cache_clear()


# -------------------------
# Public API
# -------------------------
def answer(
    user_prompt: str,
    *,
    meta_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    min_new_tokens: int = 2,
    stop_at_period: int = 1,
    period_id: int = 19,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    seed: Optional[int] = 42,
    n_heads_legacy: int = 4,
    forbid_special: int = 1,
    ban_replacement: int = 1,
) -> str:
    """
    API estable para Streamlit:
      - Carga model+tokenizer una sola vez (cache)
      - Genera y devuelve solo el texto generado (sin prompt)
      - Limpieza robusta del output
    """

    # Defaults por env (para Streamlit)
    meta_path = meta_path or _env("LLM_META", "models/tokenized/oscar_bpe_v4/meta.json")
    tokenizer_path = tokenizer_path or _env("LLM_TOKENIZER", "models/tokenizers/oscar_bpe_v4/tokenizer.json")
    ckpt_path = ckpt_path or _env("LLM_CKPT", "models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt")
    device = device or _env("LLM_DEVICE", "cpu")

    if seed is not None:
        torch.manual_seed(int(seed))

    assets = _load_assets_once(
        meta_path=meta_path,
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        device=device,
        n_heads_legacy=int(n_heads_legacy),
        forbid_special=int(forbid_special),
        ban_replacement=int(ban_replacement),
    )

    prompt = f"<instr> {user_prompt}<resp>"
    enc = assets.tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=torch.device(assets.device))

    # Greedy => temperatura no aplica (consistente con CLI)
    greedy = (int(top_k) == 0)
    temp = 1.0 if greedy else float(temperature)

    out_ids = generate(
        model=assets.model,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        block_size=int(assets.cfg.max_seq_len),
        temperature=float(temp),
        top_k=int(top_k),
        eos_id=assets.eos_id,
        repetition_penalty=float(repetition_penalty),
        no_repeat_ngram=int(no_repeat_ngram),
        min_new_tokens=int(min_new_tokens),
        banned_ids=assets.banned_ids,
        debug_next=0,
        tokenizer=assets.tokenizer,
        stop_at_period=bool(int(stop_at_period)),
        period_id=int(period_id),
    )

    full = out_ids[0].tolist()
    gen_only = full[len(enc.ids):]
    text = assets.tokenizer.decode(gen_only)

    return _clean_text(text)