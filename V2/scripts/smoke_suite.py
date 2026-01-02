# scripts/smoke_suite.py
from src.infer.answer import answer_with_meta

TESTS = [
    # 2 FACTS
    "¿Cuál es la capital de Costa Rica?",
    "¿Cuál es la capital de Francia?",

    # 2 PRIVADAS
    "¿Cuál es mi anime favorito?",
    "¿Cuáles son los tres jefes que he tenido?",

    # 3 GENERALES que deben bloquear (anti-derail)
    "Explica la fotosíntesis en una frase.",
    "Explica la relatividad en una frase.",
    "¿Qué es la física cuántica?",

    # 3 LIBRES (puede responder o rechazar, pero NO debe alucinar)
    "¿Qué es un LLM?",
    "¿Cuál es el país más grande del mundo?",
    "¿Qué es machine learning?",
]

def run(device="mps"):
    for q in TESTS:
        a, meta = answer_with_meta(
            q,
            device=device,
            max_new_tokens=60,
            min_new_tokens=2,
            stop_at_period=1,
            period_id=19,
            top_k=30,
            temperature=0.8,
        )
        print("-" * 78)
        print("Q:", q)
        print("A:", a)
        print("META:", meta)

if __name__ == "__main__":
    run()