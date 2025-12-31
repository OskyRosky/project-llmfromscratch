# src/cli/generate_tokens.py

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch

from src.model.gpt import GPTModel, GPTConfig

try:
    from tokenizers import Tokenizer
except ImportError as e:
    raise ImportError("Missing dependency: tokenizers. Install with: pip install tokenizers") from e


# -------------------------
# IO helpers
# -------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_tokenizer_path(meta_path: str, tokenizer_path: str) -> str:
    if os.path.exists(tokenizer_path):
        return tokenizer_path

    meta_dir = os.path.dirname(os.path.abspath(meta_path))

    candidate = os.path.join(meta_dir, tokenizer_path)
    if os.path.exists(candidate):
        return candidate

    base = os.path.basename(tokenizer_path)
    project_root = os.path.abspath(os.path.join(meta_dir, "..", "..", ".."))
    candidate2 = os.path.join(project_root, "models", "tokenizers", "oscar_bpe_v4", base)
    if os.path.exists(candidate2):
        return candidate2

    candidate3 = os.path.join(meta_dir, base)
    if os.path.exists(candidate3):
        return candidate3

    raise FileNotFoundError(
        "Tokenizer file not found. Tried:\n"
        f"- {tokenizer_path}\n"
        f"- {candidate}\n"
        f"- {candidate2}\n"
        f"- {candidate3}\n"
    )


def resolve_pack(pack_dir: str) -> Tuple[str, str, str, Optional[str]]:
    pdir = Path(pack_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"--pack not found: {pdir}")

    meta_path = pdir / "meta.json"
    tok_path = pdir / "tokenizer.json"
    cfg_path = pdir / "config.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {pdir}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {pdir}")

    ckpt_path = pdir.parent / "ckpt_final_step_5000.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Tip: ensure ckpt_final_step_5000.pt is in {pdir.parent}"
        )

    config_path = str(cfg_path) if cfg_path.exists() else None
    return str(meta_path), str(ckpt_path), str(tok_path), config_path


# -------------------------
# Model helpers
# -------------------------

def infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    vocab_size, d_model = sd["tok_embedding.embedding.weight"].shape
    max_seq_len = sd["pos_embedding.pos_embedding.weight"].shape[0]

    layer_ids = set()
    pat = re.compile(r"^blocks\.(\d+)\.")
    for k in sd.keys():
        m = pat.match(k)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = (max(layer_ids) + 1) if layer_ids else 0

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "max_seq_len": int(max_seq_len),
        "n_layers": int(n_layers),
    }


def cfg_from_ckpt_or_fallback(
    ckpt: Dict[str, Any],
    sd: Dict[str, torch.Tensor],
    legacy_n_heads: int,
) -> Dict[str, Any]:
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and len(model_cfg) > 0:
        allowed = {"vocab_size", "d_model", "n_layers", "n_heads", "max_seq_len", "dropout"}
        cfg_dict = {k: model_cfg[k] for k in allowed if k in model_cfg}
        cfg_dict.setdefault("dropout", 0.0)
        cfg_dict["n_heads"] = int(cfg_dict.get("n_heads", legacy_n_heads))
        return cfg_dict

    arch = infer_arch_from_state_dict(sd)
    return {
        "vocab_size": arch["vocab_size"],
        "d_model": arch["d_model"],
        "n_layers": arch["n_layers"],
        "n_heads": int(legacy_n_heads),
        "max_seq_len": arch["max_seq_len"],
        "dropout": 0.0,
    }


# -------------------------
# Decoding helpers
# -------------------------

def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """
    logits: (B, V)
    prev_ids: (B, Tprev)
    penalty > 1.0 reduces probability of already-seen tokens (deterministic).
    """
    if penalty is None or penalty <= 1.0:
        return logits

    # unique tokens per batch
    for b in range(logits.size(0)):
        seen = torch.unique(prev_ids[b])
        # If logit > 0 -> divide; if logit < 0 -> multiply (standard rep-penalty trick)
        lb = logits[b]
        lb_seen = lb[seen]
        lb[seen] = torch.where(lb_seen > 0, lb_seen / penalty, lb_seen * penalty)
        logits[b] = lb
    return logits


def calc_banned_tokens_no_repeat_ngram(
    input_ids: List[int],
    no_repeat_ngram: int,
) -> List[int]:
    """
    Returns a list of token ids that would create a repeated ngram if generated next.
    Deterministic.
    """
    n = int(no_repeat_ngram)
    if n <= 0 or len(input_ids) < n:
        return []

    # build mapping: prefix (n-1 tokens) -> set(next_tokens)
    prefix_len = n - 1
    ngrams = {}
    for i in range(len(input_ids) - n + 1):
        prefix = tuple(input_ids[i : i + prefix_len])
        nxt = input_ids[i + prefix_len]
        ngrams.setdefault(prefix, set()).add(nxt)

    current_prefix = tuple(input_ids[-prefix_len:])
    banned = list(ngrams.get(current_prefix, set()))
    return banned


@torch.no_grad()
def generate(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    top_k: int = 0,
    temperature: float = 1.0,
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    forbid_ids: Optional[List[int]] = None,
    min_new_tokens: int = 0,
) -> torch.Tensor:
    """
    Greedy by default (top_k=0). Deterministic safety knobs:
      - repetition_penalty
      - no_repeat_ngram
      - forbid_ids
      - min_new_tokens (avoid immediate EOS)
    """
    model.eval()
    greedy = (int(top_k) == 0)

    if forbid_ids is None:
        forbid_ids = []

    device = input_ids.device
    forbid_ids_t = torch.tensor(forbid_ids, dtype=torch.long, device=device) if len(forbid_ids) else None

    for t in range(int(max_new_tokens)):
        idx = input_ids[:, -int(block_size):]
        logits = model(idx)[:, -1, :]  # (B, V)

        # Deterministic: repetition penalty
        logits = apply_repetition_penalty(logits, input_ids, float(repetition_penalty))

        # Deterministic: forbid certain ids
        if forbid_ids_t is not None and forbid_ids_t.numel() > 0:
            logits[:, forbid_ids_t] = -1e10

        # Deterministic: no-repeat-ngram (greedy only; still deterministic even if sampling, but simplest here)
        if int(no_repeat_ngram) > 0:
            for b in range(logits.size(0)):
                hist = input_ids[b].tolist()
                banned = calc_banned_tokens_no_repeat_ngram(hist, int(no_repeat_ngram))
                if banned:
                    logits[b, torch.tensor(banned, device=device, dtype=torch.long)] = -1e10

        # Deterministic: avoid EOS too early
        if eos_id is not None and int(min_new_tokens) > 0 and t < int(min_new_tokens):
            logits[:, int(eos_id)] = -1e10

        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            temp = max(float(temperature), 1e-6)
            logits = logits / temp
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k=k, dim=-1)
            cutoff = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < cutoff, torch.full_like(logits, -1e10), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if eos_id is not None and int(next_id.item()) == int(eos_id) and t >= int(min_new_tokens):
            break

    return input_ids


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text from a token-level GPT checkpoint (BPE).")

    ap.add_argument("--pack", type=str, default=None, help="Path to inference_pack directory.")
    ap.add_argument("--meta", type=str, help="Path to meta.json.")
    ap.add_argument("--ckpt", type=str, help="Path to checkpoint .pt file.")
    ap.add_argument("--tokenizer_path", type=str, default=None, help="Override tokenizer.json path.")

    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--top_k", type=int, default=0, help="0=greedy. >0=top-k sampling.")
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_heads", type=int, default=4, help="Legacy only if ckpt lacks model_config.")

    # Deterministic anti-loop knobs
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 disables. Try 1.1–1.2.")
    ap.add_argument("--no_repeat_ngram", type=int, default=0, help="0 disables. Try 3.")
    ap.add_argument("--forbid_special", type=int, default=0, help="1 forbids special tokens during generation.")
    ap.add_argument("--min_new_tokens", type=int, default=0, help="If >0, EOS cannot appear before this.")

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # Resolve pack
    if args.pack:
        meta_path, ckpt_path, tok_path, _ = resolve_pack(args.pack)
        args.meta = meta_path
        args.ckpt = ckpt_path
        args.tokenizer_path = tok_path

    if not args.meta or not args.ckpt:
        raise ValueError("Either use --pack OR provide --meta and --ckpt.")

    meta = load_json(args.meta)
    tok_path_raw = args.tokenizer_path or meta["tokenizer_path"]
    tok_path = resolve_tokenizer_path(args.meta, tok_path_raw)
    tokenizer = Tokenizer.from_file(tok_path)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model_state_dict"]

    cfg_dict = cfg_from_ckpt_or_fallback(ckpt, sd, legacy_n_heads=int(args.n_heads))
    cfg = GPTConfig(**cfg_dict)

    device = torch.device(args.device)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    eos_id = meta.get("special_ids", {}).get("eos", None)

    # forbid special ids if requested (pad/unk/bos/eos/instr/resp)
    forbid_ids = []
    if int(args.forbid_special) == 1:
        sid = meta.get("special_ids", {})
        forbid_ids = [int(sid[k]) for k in ["pad", "unk", "bos", "instr", "resp"] if k in sid]
        # NOTE: we do NOT forbid eos globally; min_new_tokens handles early stop.

    enc = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)

    out_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=int(args.max_new_tokens),
        block_size=int(cfg.max_seq_len),
        top_k=int(args.top_k),
        temperature=float(args.temperature),
        eos_id=int(eos_id) if eos_id is not None else None,
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram=int(args.no_repeat_ngram),
        forbid_ids=forbid_ids,
        min_new_tokens=int(args.min_new_tokens),
    )

    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()