# src/cli/build_token_ids.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils.bpe_tokenizer import BPETokenizer


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json (BPE)")
    ap.add_argument("--corpus", required=True, help="Path to corpus .txt (one line per chunk)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "int32"], help="Storage dtype")
    ap.add_argument("--max_lines", type=int, default=0, help="0 = no limit (debug only)")
    args = ap.parse_args()

    tok_path = Path(args.tokenizer).expanduser().resolve()
    corpus_path = Path(args.corpus).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    tok = BPETokenizer(tok_path)

    # dtype seguro para vocab 4096
    if args.dtype == "uint16":
        dtype = np.uint16
    else:
        dtype = np.int32

    bin_path = out_dir / "tokens.bin"
    meta_path = out_dir / "meta.json"

    total_tokens = 0

    with bin_path.open("wb") as bf:
        for i, line in enumerate(tqdm(iter_lines(corpus_path), desc="Tokenizing lines")):
            if args.max_lines and i >= args.max_lines:
                break

            # Importante: agregar EOS para cortar “documentos”
            ids = tok.encode(line, add_eos=True)

            arr = np.asarray(ids, dtype=dtype)
            bf.write(arr.tobytes())
            total_tokens += int(arr.size)

    meta = {
        "tokenizer_path": str(tok_path),
        "corpus_path": str(corpus_path),
        "vocab_size": tok.vocab_size,
        "special_ids": {
            "pad": tok.pad_id,
            "unk": tok.unk_id,
            "bos": tok.bos_id,
            "eos": tok.eos_id,
            "instr": tok.instr_id,
            "resp": tok.resp_id,
        },
        "dtype": args.dtype,
        "n_tokens": total_tokens,
        "bin_file": str(bin_path),
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] tokens.bin: {bin_path}")
    print(f"[OK] meta.json:  {meta_path}")
    print(f"[OK] n_tokens:   {total_tokens}")


if __name__ == "__main__":
    main()