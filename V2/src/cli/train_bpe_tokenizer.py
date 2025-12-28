# src/cli/train_bpe_tokenizer.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<instr>", "<resp>"]


def train_bpe_tokenizer(
    corpus_path: str,
    out_dir: str,
    vocab_size: int = 4096,
    min_frequency: int = 2,
) -> str:
    corpus_path = str(Path(corpus_path).expanduser())
    out_dir = str(Path(out_dir).expanduser())

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    os.makedirs(out_dir, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.decoder = ByteLevelDecoder()

    # Define BOS/EOS usable en generación si luego decides agregarlos explícitamente
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    return tokenizer_path


def quick_sanity_check(tokenizer_path: str) -> None:
    tok = Tokenizer.from_file(tokenizer_path)
    vocab = tok.get_vocab()

    specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<instr>", "<resp>"]
    missing = [t for t in specials if t not in vocab]
    if missing:
        raise RuntimeError(f"Tokenizer missing special tokens in vocab: {missing}")

    print("\n[CHECK] Special tokens in vocab ✅")
    print({t: vocab.get(t) for t in specials})
    print("vocab_size:", tok.get_vocab_size())

    tests = [
        "Los perros son caninos?",
        "<instr> Los perros son caninos?\n<resp>",
        "La capital de Costa Rica es San José.",
    ]

    for t in tests:
        enc = tok.encode(t)
        dec = tok.decode(enc.ids)
        print("\n---")
        print("TEXT: ", repr(t))
        print("IDS:  ", enc.ids[:40], "..." if len(enc.ids) > 40 else "")
        print("DECODE:", repr(dec))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer (subword) over a text corpus.")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus txt (e.g., data/raw/oscar_corpus.txt)")
    parser.add_argument("--out_dir", type=str, default="models/tokenizers/oscar_bpe_v4", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size (e.g., 4096, 8192, 16384)")
    parser.add_argument("--min_frequency", type=int, default=2, help="Min token frequency")
    args = parser.parse_args()

    tokenizer_path = train_bpe_tokenizer(
        corpus_path=args.corpus,
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    print(f"\n[INFO] Tokenizer saved to: {tokenizer_path}")
    quick_sanity_check(tokenizer_path)


if __name__ == "__main__":
    main()