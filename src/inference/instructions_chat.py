# src/inference/instructions_chat.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
from torch import nn

from src.model.gpt import GPTConfig, GPTModel


# -------------------------------------------------------------------
# Dataclass para agrupar todo lo necesario del modelo
# -------------------------------------------------------------------
@dataclass
class InstructionsModelBundle:
    model: nn.Module
    tokenizer: "CharTokenizerForInstructions"
    config: GPTConfig
    device: torch.device
    pad_id: int


# -------------------------------------------------------------------
# Tokenizer para inference de instrucciones
# -------------------------------------------------------------------
class CharTokenizerForInstructions:
    """
    Tokenizer muy simple basado en stoi/itos del pretraining,
    reutilizado para instruction tuning.

    - encode(text): mapea cada carácter a su ID.
    - decode(ids): mapea IDs a caracteres.
    """

    def __init__(self, stoi: Dict[str, int]):
        self.stoi: Dict[str, int] = stoi
        self.itos: Dict[int, str] = {v: k for k, v in stoi.items()}

        # Detectar token de padding y su id
        if "<PAD>" in stoi:
            self.pad_token = "<PAD>"
        elif "<pad>" in stoi:
            self.pad_token = "<pad>"
        else:
            self.pad_token = None

        if self.pad_token is not None:
            self.pad_id = stoi[self.pad_token]
        else:
            # fallback: 0 si no encontramos PAD explícito
            self.pad_id = 0

    def encode(self, text: str) -> List[int]:
        """
        Codifica carácter por carácter usando stoi.
        Si un carácter no existe en el vocabulario, usa <unk> o el primer ID.
        """
        default_id = self.stoi.get("<unk>", next(iter(self.stoi.values())))
        return [self.stoi.get(ch, default_id) for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        Decodifica IDs a texto, saltándose el token de padding.
        """
        chars: List[str] = []
        for i in ids:
            ch = self.itos.get(int(i), "")
            if self.pad_token is not None and ch == self.pad_token:
                # no mostramos el PAD en el texto final
                continue
            chars.append(ch)
        return "".join(chars)


# -------------------------------------------------------------------
# Helpers de dispositivo
# -------------------------------------------------------------------
def get_device(device_str: str | None) -> torch.device:
    """
    Convierte un string ('cpu', 'mps', 'cuda', 'auto') en torch.device.
    Siempre espera un string o None, nunca un torch.device.
    """
    if device_str is None:
        return torch.device("cpu")

    s = str(device_str).lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device(s)

    if s == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# -------------------------------------------------------------------
# Carga del modelo de instrucciones
# -------------------------------------------------------------------
def load_instructions_model(
    ckpt_dir_or_path: str,
    device_str: str = "cpu",
) -> InstructionsModelBundle:
    """
    Carga el checkpoint de instrucciones.

    Admite dos formas de llamada:
      1) load_instructions_model("models/checkpoints_oscar_long")
         -> busca gpt_char_instructions.pt dentro del directorio.
      2) load_instructions_model("models/checkpoints_oscar_long/gpt_char_instructions.pt")
         -> usa directamente esa ruta.
    """
    # Detectar si nos pasaron un directorio o un archivo
    if os.path.isdir(ckpt_dir_or_path):
        ckpt_path = os.path.join(ckpt_dir_or_path, "gpt_char_instructions.pt")
    else:
        ckpt_path = ckpt_dir_or_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint de instrucciones en: {ckpt_path}"
        )

    print(f"[INFO] Cargando modelo de instrucciones desde: {ckpt_path}")

    obj: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError("El checkpoint de instrucciones no es un diccionario.")

    if "config" not in obj or "model_state_dict" not in obj or "stoi" not in obj:
        raise ValueError(
            "El checkpoint de instrucciones debe contener 'config', "
            "'model_state_dict' y 'stoi'."
        )

    config_dict = obj["config"]
    state_dict = obj["model_state_dict"]
    stoi = obj["stoi"]

    # Reconstruir config y tokenizer
    config = GPTConfig(**config_dict)
    tokenizer = CharTokenizerForInstructions(stoi)

    # Instanciar modelo y cargar pesos
    model = GPTModel(config)
    model.load_state_dict(state_dict)

    device = get_device(device_str)
    model.to(device)
    model.eval()

    pad_id = tokenizer.pad_id
    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] pad_id (inference): {pad_id}")

    return InstructionsModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        pad_id=pad_id,
    )


# -------------------------------------------------------------------
# Helpers de prompt / codificación
# -------------------------------------------------------------------
def build_prompt(user_prompt: str) -> str:
    """
    Construye el texto de entrada estilo instruction-tuning:
        "<instr> {user_prompt}\n<resp> "
    """
    return f"<instr> {user_prompt}\n<resp> "


# -------------------------------------------------------------------
# Generación de respuesta (sin usar model.generate)
# -------------------------------------------------------------------
@torch.no_grad()
def generate_answer(
    bundle: InstructionsModelBundle,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> Tuple[str, str]:
    """
    Genera una respuesta de texto dado un prompt de usuario.

    IMPORTANTE:
    - No usamos model.generate para poder:
        * Forzar que el modelo NO elija el token de padding.
        * Controlar explícitamente el contexto (max_seq_len).

    Devuelve:
      - answer_text: solo el texto posterior a <resp>, sin tokens de padding.
      - full_text: todo el texto decodificado (prompt + respuesta).
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    config = bundle.config
    device = bundle.device
    pad_id = bundle.pad_id

    # 1) Construir prompt completo
    prompt_text = build_prompt(prompt)

    # 2) Codificar sin padding artificial, solo la secuencia real
    input_ids = tokenizer.encode(prompt_text)
    # limitar al contexto máximo del modelo
    max_ctx = config.max_seq_len
    if len(input_ids) > max_ctx:
        input_ids = input_ids[-max_ctx:]

    generated = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # (1, T_in)

    # 3) Bucle autoregresivo
    for _ in range(max_new_tokens):
        # Asegurar que no nos pasamos del contexto máximo
        if generated.size(1) > max_ctx:
            context = generated[:, -max_ctx:]
        else:
            context = generated

        logits = model(context)  # (1, T_ctx, vocab_size)
        logits_last = logits[:, -1, :]  # (1, vocab_size)

        # 🔴 CLAVE: prohibir que el modelo escoja el token PAD
        if pad_id is not None:
            logits_last[:, pad_id] = -1e9

        # Greedy o muestreo con temperatura
        if temperature is None or temperature <= 0.0:
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)  # (1, 1)
        else:
            probs = torch.softmax(logits_last / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=1)

    # 4) Decodificar todo
    full_ids = generated[0].tolist()
    full_text = tokenizer.decode(full_ids)

    # 5) Nos quedamos solo con la parte después de "<resp>"
    if "<resp>" in full_text:
        after_resp = full_text.split("<resp>", 1)[1]
    else:
        after_resp = full_text

    # Limpiar espacios extremos
    answer_text = after_resp.strip()

    return answer_text, full_text


# -------------------------------------------------------------------
# Pequeño test manual desde la terminal
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Puedes llamar con directorio o con ruta completa al checkpoint
    ckpt_dir = "models/checkpoints_oscar_long"
    bundle = load_instructions_model(ckpt_dir, device_str="cpu")

    questions = [
        "Un perro es un canino?",
        "Cuál es la capital de Costa Rica?",
        "Qué es un modelo de lenguaje?",
    ]

    for q in questions:
        ans, full = generate_answer(
            bundle=bundle,
            prompt=q,
            max_new_tokens=80,
            temperature=0.0,
        )
        print("\n====================================================")
        print("Pregunta:", q)
        print("Respuesta del modelo:")
        print(repr(ans))