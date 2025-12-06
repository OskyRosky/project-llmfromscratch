# src/inference/eval_instructions_mini.py

"""
Pequeño script de evaluación para el modelo de instrucciones.

Objetivo:
- Cargar el checkpoint de instrucciones desde el directorio:
    models/checkpoints_oscar_long
- Hacerle unas pocas preguntas fijas:
    - "Un perro es un canino?"
    - "Un gato es un felino?"
    - "Cuál es la capital de Costa Rica?"
- Imprimir la respuesta generada.

Se apoya en las funciones ya definidas en:
    src.inference.instructions_chat
"""

from typing import List

from src.inference.instructions_chat import (
    load_instructions_model,
    generate_answer,
    InstructionsModelBundle,
)


def run_eval(
    questions: List[str],
    ckpt_dir: str = "models/checkpoints_oscar_long",
    device_str: str = "mps",
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> None:
    """
    Carga el modelo de instrucciones y genera una respuesta para cada pregunta.

    IMPORTANTE:
    - `ckpt_dir` es el DIRECTORIO donde está `gpt_char_instructions.pt`,
      no la ruta completa al archivo.
    """
    print("[INFO] Cargando modelo de instrucciones para evaluación...")
    # 👉 Aquí pasamos el directorio, tal como espera load_instructions_model
    bundle: InstructionsModelBundle = load_instructions_model(
        ckpt_dir,
        device_str=device_str,
    )

    print("[INFO] Modelo cargado. Dispositivo:", bundle.device)
    print("=====================================================")

    for q in questions:
        print("\n-----------------------------------------------------")
        print("Pregunta:", q)

        answer, raw_text = generate_answer(
            bundle=bundle,
            question=q,
            max_new_tokens=max_new_tokens,
            temperature=temperature,  # 0.0 = súper determinista
        )

        print("\n[Texto generado completo]:")
        print(repr(raw_text))

        print("\n[Respuesta procesada]:")
        print(answer)

        print("-----------------------------------------------------")


def main() -> None:
    # Tus tres preguntas clave
    questions = [
        "Un perro es un canino?",
        "Un gato es un felino?",
        "Cuál es la capital de Costa Rica?",
    ]

    run_eval(
        questions=questions,
        ckpt_dir="models/checkpoints_oscar_long",
        device_str="mps",
        max_new_tokens=80,
        temperature=0.0,
    )


if __name__ == "__main__":
    main()