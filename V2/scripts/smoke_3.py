import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer.answer import answer

META="models/tokenized/oscar_bpe_v4/meta.json"
TOK="models/tokenizers/oscar_bpe_v4/tokenizer.json"
CKPT="models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt"

tests = [
  "¿Cuál es la capital de Costa Rica?",
  "¿Cuál es la capital de Francia?",
  "Los perros pertenecen a qué familia",
]

for t in tests:
    print("-"*60)
    print("Q:", t)
    print("A:", answer(t, meta_path=META, ckpt_path=CKPT, tokenizer_path=TOK, device="mps",
                      max_new_tokens=40, min_new_tokens=2, stop_at_period=1, period_id=19))
