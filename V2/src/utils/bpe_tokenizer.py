from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from tokenizers import Tokenizer


@dataclass
class SpecialTokens:
    pad: str = "<PAD>"
    unk: str = "<UNK>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"
    instr: str = "<instr>"
    resp: str = "<resp>"


class BPETokenizer:
    """
    Wrapper estable alrededor de `tokenizers.Tokenizer` para usarlo en:
    datasets, entrenamiento e inferencia.
    """

    def __init__(self, tokenizer_path: str | Path, specials: Optional[SpecialTokens] = None):
        self.tokenizer_path = Path(tokenizer_path)
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

        self._tok = Tokenizer.from_file(str(self.tokenizer_path))
        self.specials = specials or SpecialTokens()

        # vocab: token -> id
        self.vocab: Dict[str, int] = self._tok.get_vocab()
        self.vocab_size: int = self._tok.get_vocab_size()

        # ids especiales (si no existen, levantamos error)
        self.pad_id = self._require_id(self.specials.pad)
        self.unk_id = self._require_id(self.specials.unk)
        self.bos_id = self._require_id(self.specials.bos)
        self.eos_id = self._require_id(self.specials.eos)
        self.instr_id = self._require_id(self.specials.instr)
        self.resp_id = self._require_id(self.specials.resp)

    def _require_id(self, token: str) -> int:
        tid = self.vocab.get(token, None)
        if tid is None:
            raise ValueError(
                f"Missing special token in tokenizer vocab: {token}\n"
                f"Tokenizer: {self.tokenizer_path}"
            )
        return tid

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        enc = self._tok.encode(text)
        ids = enc.ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token, None)