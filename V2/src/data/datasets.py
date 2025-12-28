# src/data/datasets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------
# 1. CharacterDataset – pretraining (LM de caracteres)
# --------------------------------------------------------------------


class CharacterDataset(Dataset):
    """
    Dataset de nivel carácter para pretraining (next-token prediction).

    Dado un vector de IDs [t0, t1, ..., tN], construye ejemplos:

        x = [ti,     ti+1,   ..., ti+seq_len-1]
        y = [ti+1,   ti+2,   ..., ti+seq_len]

    para todos los offsets válidos i.
    """

    def __init__(self, token_ids: Sequence[int], seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        if len(token_ids) <= seq_len:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}. "
                "You need a longer corpus or a smaller seq_len."
            )

        self.token_ids: List[int] = list(token_ids)
        self.seq_len: int = seq_len

    def __len__(self) -> int:
        """
        Para una secuencia de longitud N y ventana L, hay N - L ejemplos.
        """
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (x, y) de shape (seq_len,).

        x: token_ids[idx : idx + seq_len]
        y: token_ids[idx + 1 : idx + 1 + seq_len]
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self) - 1}")

        x_ids = self.token_ids[idx : idx + self.seq_len]
        y_ids = self.token_ids[idx + 1 : idx + 1 + self.seq_len]

        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)

        return x, y


# --------------------------------------------------------------------
# 1B. TokenBinDataset – pretraining token-level (BPE ids desde tokens.bin)
# --------------------------------------------------------------------


class TokenBinDataset(Dataset):
    """
    Dataset token-level para pretraining (next-token prediction) leyendo
    un archivo binario tokens.bin (uint16/uint32) vía memmap.

    Construye ejemplos:
        x = tokens[i : i+seq_len]
        y = tokens[i+1 : i+1+seq_len]
    """

    def __init__(self, tokens_bin_path: str, seq_len: int, dtype: str = "uint16") -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.tokens_bin_path = str(tokens_bin_path)
        self.seq_len = int(seq_len)

        if dtype not in ("uint16", "uint32"):
            raise ValueError("dtype must be 'uint16' or 'uint32'")
        self.dtype = dtype

        # Import local para no forzar numpy si no se usa este dataset
        import numpy as np

        np_dtype = np.uint16 if dtype == "uint16" else np.uint32

        # memmap (no carga todo en RAM)
        self._mm = np.memmap(self.tokens_bin_path, mode="r", dtype=np_dtype)

        if self._mm.size <= self.seq_len:
            raise ValueError(
                f"Not enough tokens ({self._mm.size}) for seq_len={self.seq_len}. "
                "Need a longer corpus or a smaller seq_len."
            )

        self.n_tokens = int(self._mm.size)

    def __len__(self) -> int:
        return self.n_tokens - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self) - 1}")

        # Import local para no forzar numpy si no se usa este dataset
        import numpy as np

        x_np = self._mm[idx : idx + self.seq_len].astype(np.int64, copy=False)
        y_np = self._mm[idx + 1 : idx + 1 + self.seq_len].astype(np.int64, copy=False)

        x = torch.from_numpy(x_np).long()
        y = torch.from_numpy(y_np).long()
        return x, y


# --------------------------------------------------------------------
# 2. ClassificationDataset – finetuning de clasificación
# --------------------------------------------------------------------


@dataclass
class ClassificationExample:
    """
    Ejemplo de clasificación:
    - text: texto de entrada.
    - label: entero que representa la clase (0, 1, 2, ...).
    """
    text: str
    label: int


class ClassificationDataset(Dataset):
    """
    Dataset para clasificación de texto usando un tokenizer.

    Requisitos del tokenizer:
      - método encode(text: str) -> List[int]
      - opcionalmente un vocabulario .stoi para localizar pad_id.
    """

    def __init__(self, examples, tokenizer, seq_len: int):
        self.examples: List[ClassificationExample] = list(examples)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Intentar encontrar un pad_id razonable:
        # Primero "<PAD>", luego "<pad>", y si no existe, 0.
        stoi: Dict[str, int] = getattr(self.tokenizer, "stoi", {})
        self.pad_id: int = stoi.get("<PAD>", stoi.get("<pad>", 0))

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_text(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)

        if len(ids) > self.seq_len:
            ids = ids[: self.seq_len]
        elif len(ids) < self.seq_len:
            ids = ids + [self.pad_id] * (self.seq_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        input_ids = self._encode_text(ex.text)
        label = torch.tensor(ex.label, dtype=torch.long)

        return {
            "input_ids": input_ids,  # (seq_len,)
            "label": label,
        }


# --------------------------------------------------------------------
# 3. InstructionDataset – instruction tuning (Fase 6 / 8)
# --------------------------------------------------------------------


@dataclass
class InstructionExample:
    """
    Ejemplo de instrucción para fine-tuning:
    - prompt: instrucción / pregunta.
    - response: respuesta objetivo.
    """
    prompt: str
    response: str


class InstructionDataset(Dataset):
    """
    Dataset para instruction tuning.

    Construye texto:
        "<instr> {prompt}\n<resp> {response}"

    Luego tokeniza y aplica truncado/padding a seq_len.

    Retorna:
      - input_ids
      - labels (clon de input_ids; el shift se hace en entrenamiento/loss)
    """

    def __init__(
        self,
        examples: List[InstructionExample],
        tokenizer,
        seq_len: int,
        instr_prefix: str = "<instr>",
        resp_prefix: str = "<resp>",
    ):
        self.examples: List[InstructionExample] = list(examples)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.instr_prefix = instr_prefix
        self.resp_prefix = resp_prefix

        # Detectar pad_id coherente con el vocabulario del tokenizer
        stoi: Dict[str, int] = getattr(self.tokenizer, "stoi", {})
        self.pad_id: int = stoi.get("<PAD>", stoi.get("<pad>", 0))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        full_text = f"{self.instr_prefix} {ex.prompt}\n{self.resp_prefix} {ex.response}"
        ids = self.tokenizer.encode(full_text)

        # truncado
        ids = ids[: self.seq_len]

        # padding
        if len(ids) < self.seq_len:
            ids = ids + [self.pad_id] * (self.seq_len - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)

        return {
            "input_ids": input_ids,      # (T,)
            "labels": input_ids.clone(), # (T,)
        }