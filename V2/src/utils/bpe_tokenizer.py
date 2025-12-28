from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from tokenizers import Tokenizer


@dataclass(frozen=True)
class SpecialTokens:
    # Deben coincidir EXACTO con train_bpe_tokenizer.py (v4)
    pad: str = "<pad>"
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    instr: str = "<instr>"
    resp: str = "<resp>"


class BPETokenizer:
    """
    Wrapper estable para usar SIEMPRE el mismo tokenizer en:
    - datasets
    - training
    - finetuning
    - inference
    """

    def __init__(self, tokenizer_path: Union[str, Path], specials: Optional[SpecialTokens] = None):
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

        self._tok = Tokenizer.from_file(str(self.tokenizer_path))
        self.specials = specials or SpecialTokens()

        self.vocab: Dict[str, int] = self._tok.get_vocab()
        self.vocab_size: int = self._tok.get_vocab_size()

        # IDs especiales: ahora DEBEN existir en v4. Si falta algo, levantamos error.
        self.pad_id = self._require_id(self.specials.pad)
        self.unk_id = self._require_id(self.specials.unk)
        self.bos_id = self._require_id(self.specials.bos)
        self.eos_id = self._require_id(self.specials.eos)
        self.instr_id = self._require_id(self.specials.instr)
        self.resp_id = self._require_id(self.specials.resp)

    def _require_id(self, token: str) -> int:
        tid = self.vocab.get(token)
        if tid is None:
            raise ValueError(
                f"Missing special token in tokenizer vocab: {token}\n"
                f"Tokenizer: {self.tokenizer_path}"
            )
        return tid

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token)