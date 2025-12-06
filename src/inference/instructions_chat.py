# src/inference/instructions_chat.py

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
from torch import nn

from src.model.gpt import GPTConfig, GPTModel


# ---------------------------------------------------------------------
# Utilidad de dispositivo
# ---------------------------------------------------------------------
def get_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------
# Tokenizer simple a partir de stoi guardado
# ---------------------------------------------------------------------
class CharTokenizerFromState:
    """
    Wrapper mínimo alrededor de stoi para:
      - encode(text) -> List[int]
      - decode(List[int]) -> str

    Asume que stoi es un diccionario {token(str) -> id(int)}.
    En tu caso, son caracteres + algunos tokens especiales ("<PAD>", etc.).
    """

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        # itos: inverso de stoi
        self.itos = {idx: ch for ch, idx in stoi.items()}

        # ID por defecto para caracteres desconocidos
        # (si existe "<unk>", lo usamos; si no, tomamos el primer id)
        self.default_id = self.stoi.get("<unk>", next(iter(self.stoi.values())))

    def encode(self, text: str) -> List[int]:
        """
        Convierte texto en lista de IDs.
        Si un carácter no está en stoi, usa default_id.
        """
        return [self.stoi.get(ch, self.default_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Convierte lista de IDs en texto.
        Si un id no está en itos, lo reemplaza por "?".
        """
        chars = [self.itos.get(i, "?") for i in ids]
        return "".join(chars)


# ---------------------------------------------------------------------
# Carga del modelo de instrucciones
# ---------------------------------------------------------------------
@dataclass
class InstructionsModelBundle:
    """
    Contenedor para todo lo que necesitamos en inferencia.
    """
    model: GPTModel
    tokenizer: CharTokenizerFromState
    config: GPTConfig
    device: torch.device


def load_instructions_model(
    ckpt_dir: str = "models/checkpoints_oscar_long",
    device_str: str = "auto",
) -> InstructionsModelBundle:
    """
    Carga el modelo de instrucciones (gpt_char_instructions.pt)
    + tokenizer (desde el propio checkpoint) y devuelve un bundle
    listo para usar en inferencia.
    """
    device = get_device(device_str)

    ckpt_path = os.path.join(ckpt_dir, "gpt_char_instructions.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró el checkpoint de instrucciones en: {ckpt_path}")

    print(f"[INFO] Cargando modelo de instrucciones desde: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")

    # Extraer config, pesos y vocabulario
    config_dict = obj["config"]
    state_dict = obj["model_state_dict"]
    stoi = obj["stoi"]

    config = GPTConfig(**config_dict)
    tokenizer = CharTokenizerFromState(stoi)

    # Instanciar modelo y cargar pesos
    model = GPTModel(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return InstructionsModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )


# ---------------------------------------------------------------------
# Construcción de prompt y generación
# ---------------------------------------------------------------------
def build_prompt(question: str) -> str:
    """
    Construye el prompt para instruction tuning.
    """
    return f"<instr> {question}\n<resp> "


def encode_prompt(
    text: str,
    tokenizer: CharTokenizerFromState,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Codifica el prompt a ids, trunca/paddea a seq_len y lo pone en device.
    """
    ids = tokenizer.encode(text)
    ids = ids[:seq_len]

    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    return input_ids  # (1, T)


def postprocess_generated_text(
    generated: str,
    resp_token: str = "<resp>",
    pad_token: str = "<PAD>",
) -> str:
    """
    Toma el texto completo generado (incluye prompt + respuesta),
    corta lo que viene después de <resp> y limpia tokens <PAD>.
    """
    if resp_token in generated:
        after_resp = generated.split(resp_token, 1)[1]
    else:
        after_resp = generated

    # Limpiar <PAD> y espacios extra
    cleaned = after_resp.replace(pad_token, "")
    return cleaned.strip()


def generate_answer(
    question: str,
    bundle: InstructionsModelBundle,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = None,
) -> Tuple[str, str]:
    """
    Genera una respuesta para la pregunta dada.

    Devuelve:
      - full_text: el texto completo (prompt + lo generado)
      - answer_only: solo la parte después de <resp>, limpia.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    config = bundle.config
    device = bundle.device

    # 1. Construir prompt
    prompt_text = build_prompt(question)

    # 2. Codificar
    inp = encode_prompt(prompt_text, tokenizer, config.max_seq_len, device)

    # 3. Generar
    with torch.no_grad():
        out_ids = model.generate(
            inp,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # 4. Decodificar
    full_text = tokenizer.decode(out_ids[0].tolist())

    # 5. Post-procesar para obtener solo la respuesta
    answer_only = postprocess_generated_text(full_text)

    return full_text, answer_only


# ---------------------------------------------------------------------
# Pequeño main de prueba desde terminal:
#   python -m src.inference.instructions_chat
# ---------------------------------------------------------------------
def main():
    bundle = load_instructions_model(
        ckpt_dir="models/checkpoints_oscar_long",
        device_str="auto",
    )

    questions = [
        "Un perro es un canino?",
        "Cuál es la capital de Costa Rica?",
        "Qué es un modelo de lenguaje?",
    ]

    for q in questions:
        full, ans = generate_answer(q, bundle, max_new_tokens=80, temperature=0.7)

        print("\n" + "=" * 60)
        print("Pregunta:", q)
        print("\n[Texto completo generado]:")
        print(repr(full))
        print("\n[Solo respuesta procesada]:")
        print(repr(ans))


if __name__ == "__main__":
    main()