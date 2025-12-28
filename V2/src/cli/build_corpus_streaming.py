# V2/src/cli/build_corpus_streaming.py
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def clean_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_wikipedia_es(out_path: Path, target_mb: int = 600, min_chars: int = 200) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = target_mb * 1024 * 1024
    written = 0

    # Dataset oficial en HF: wikimedia/wikipedia
    # Config común: 20231101.es (si tu entorno tiene otra disponible, lo ajustamos luego)
    ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Streaming Wikipedia ES"):
            # Según el dataset, el campo típico es "text"
            text = row.get("text", "")
            text = clean_text(text)
            if len(text) < min_chars:
                continue

            # una línea por documento
            line = text + "\n"
            f.write(line)
            written += len(line.encode("utf-8"))

            if written >= target_bytes:
                break

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Corpus creado: {out_path}")
    print(f"[OK] Tamaño final: {size_mb:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Ruta de salida .txt (fuera del repo recomendado)")
    ap.add_argument("--target_mb", type=int, default=600, help="Tamaño objetivo en MB")
    ap.add_argument("--min_chars", type=int, default=200, help="Filtrar líneas muy cortas")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    build_wikipedia_es(out_path, target_mb=args.target_mb, min_chars=args.min_chars)


if __name__ == "__main__":
    main()