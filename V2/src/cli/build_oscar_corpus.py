# src/cli/build_oscar_corpus.py
import argparse
import os
import re
from datasets import load_dataset
from tqdm import tqdm


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_text_samples(ds, text_field: str):
    for ex in ds:
        if text_field in ex and ex[text_field]:
            yield ex[text_field]


def main():
    p = argparse.ArgumentParser(description="Build a large Spanish corpus (txt) by streaming a HF dataset.")
    p.add_argument("--dataset", type=str, default="allenai/c4", help="HF dataset path (default: allenai/c4)")
    p.add_argument("--config", type=str, default="es", help="Dataset config/subset (default: es)")
    p.add_argument("--split", type=str, default="train", help="Split name (default: train)")
    p.add_argument("--text_field", type=str, default="text", help="Field containing text (default: text)")

    p.add_argument("--out", type=str, required=True, help="Output .txt path")
    p.add_argument("--target_mb", type=int, default=550, help="Target size in MB (default: 550)")
    p.add_argument("--max_samples", type=int, default=0, help="Optional hard cap on samples (0 = no cap)")

    p.add_argument("--min_chars", type=int, default=200, help="Skip samples shorter than this (default: 200)")
    p.add_argument("--newline_between_samples", action="store_true", help="Write an empty line between samples")

    args = p.parse_args()

    target_bytes = args.target_mb * 1024 * 1024
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Stream dataset (no local full download)
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    written = 0
    kept = 0
    seen = 0

    with open(args.out, "w", encoding="utf-8") as f:
        bar = tqdm(total=target_bytes, unit="B", unit_scale=True, desc="Writing corpus")
        for text in iter_text_samples(ds, args.text_field):
            seen += 1
            if args.max_samples and seen > args.max_samples:
                break

            t = normalize_text(text)
            if len(t) < args.min_chars:
                continue

            line = t + "\n"
            if args.newline_between_samples:
                line = t + "\n\n"

            b = line.encode("utf-8")
            if written + len(b) > target_bytes:
                # write only what fits to hit target size closely
                remaining = target_bytes - written
                if remaining > 0:
                    f.write(b[:remaining].decode("utf-8", errors="ignore"))
                    written += remaining
                    bar.update(remaining)
                break

            f.write(line)
            written += len(b)
            kept += 1
            bar.update(len(b))

        bar.close()

    print("\nDone.")
    print(f"Dataset: {args.dataset} / {args.config} / {args.split}")
    print(f"Samples seen: {seen:,}")
    print(f"Samples kept: {kept:,}")
    print(f"Output: {args.out}")
    print(f"Size: {written/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()