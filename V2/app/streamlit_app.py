# app/streamlit_app.py
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------
# Asegurar imports "src..."
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # V2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer.answer import answer, clear_cache  # noqa: E402


# ---------------------------------------------------------------------
# Config por env (recomendado)
# ---------------------------------------------------------------------
DEFAULT_META = os.getenv("LLM_META", "models/tokenized/oscar_bpe_v4/meta.json")
DEFAULT_TOK = os.getenv("LLM_TOKENIZER", "models/tokenizers/oscar_bpe_v4/tokenizer.json")
DEFAULT_CKPT = os.getenv(
    "LLM_CKPT",
    "models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt",
)
DEFAULT_DEVICE = os.getenv("LLM_DEVICE", "mps")


# ---------------------------------------------------------------------
# FAQ fallback (mínimo y determinístico)
# - Nota: esto es un fallback "perfecto", útil para demo.
# - Luego lo hacemos fuzzy si querés.
# ---------------------------------------------------------------------
FAQ = {
    "¿cuál es la capital de costa rica?": "San José.",
    "capital de costa rica": "San José.",
    "¿cuál es la capital de francia?": "París.",
    "capital francesa": "París.",
    "¿los perros son caninos?": "Sí, los perros pertenecen a la familia de los cánidos.",
    "los perros pertenecen a qué familia": "A la familia de los cánidos.",
    # Si quieres que estas sean “obvias” en demo, ponlas aquí:
    "¿cuál es el 5 planeta del sistema solar?": "Júpiter.",
    "¿cuál es la capital de argentina?": "Buenos Aires.",
}

def normalize_q(s: str) -> str:
    return " ".join(s.strip().lower().split())


def faq_match(prompt: str) -> str | None:
    return FAQ.get(normalize_q(prompt))


# ---------------------------------------------------------------------
# Streamlit basic page
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LLM From Scratch – Token Chat (BPE)",
    page_icon="💬",
    layout="wide",
)

st.title("💬 LLM From Scratch – Token Chat (BPE)")

st.markdown(
    """
Modelo **token-level (BPE)** entrenado desde cero y luego **instruction-tuned**.

- Primero intentamos un **FAQ fallback** (respuesta perfecta y determinística).
- Si no hay match, usamos el **LLM** (tu checkpoint instruccional).
"""
)

# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

meta_path = st.sidebar.text_input("meta.json", value=DEFAULT_META)
tokenizer_path = st.sidebar.text_input("tokenizer.json", value=DEFAULT_TOK)
ckpt_path = st.sidebar.text_input("checkpoint (.pt)", value=DEFAULT_CKPT)
device = st.sidebar.selectbox("device", options=["mps", "cpu"], index=0 if DEFAULT_DEVICE == "mps" else 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Decoding")

max_new_tokens = st.sidebar.slider("max_new_tokens", 10, 200, 60, 5)
min_new_tokens = st.sidebar.slider("min_new_tokens", 1, 40, 2, 1)

stop_at_period = st.sidebar.checkbox("stop_at_period (.)", value=True)
period_id = st.sidebar.number_input("period_id", value=19, step=1)

top_k = st.sidebar.slider("top_k (0 = greedy)", 0, 100, 0, 1)
temperature = st.sidebar.slider("temperature (solo si top_k>0)", 0.0, 1.5, 1.0, 0.05)

repetition_penalty = st.sidebar.slider("repetition_penalty", 1.0, 2.0, 1.0, 0.05)
no_repeat_ngram = st.sidebar.slider("no_repeat_ngram", 0, 6, 0, 1)

st.sidebar.markdown("---")
use_faq = st.sidebar.checkbox("Usar FAQ fallback", value=True)

if st.sidebar.button("🔁 Clear cache (recargar modelo)"):
    clear_cache()
    st.sidebar.success("Cache limpiado. La próxima respuesta recarga assets.")

# ---------------------------------------------------------------------
# Main: test questions
# ---------------------------------------------------------------------
st.markdown("### Pregunta de prueba")

opciones = [
    "¿Cuál es la capital de Costa Rica?",
    "¿Cuál es la capital de Francia?",
    "Los perros pertenecen a qué familia",
    "¿Cuál es el 5 planeta del sistema solar?",
    "¿Cuál es la capital de Argentina?",
    # ejemplos “privados” (debe decir No tengo…)
    "¿Cuál es el nombre de mis 4 perros?",
    "¿Qué edad tiene mi hermano gemelo?",
]

pregunta_base = st.radio("Elige una pregunta:", opciones, index=0)

prompt = st.text_area(
    "Puedes editar la pregunta:",
    value=pregunta_base,
    height=90,
)

# ---------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------
if st.button("Generar respuesta"):
    if not prompt.strip():
        st.warning("Por favor escribe una pregunta.")
    else:
        with st.spinner("Generando..."):
            # 1) FAQ fallback
            if use_faq:
                fa = faq_match(prompt)
            else:
                fa = None

            if fa is not None:
                st.success("✔ Respuesta via FAQ fallback (determinística)")
                answer_text = fa
                debug_info = "(FAQ)"
            else:
                # 2) LLM
                answer_text = answer(
                    prompt,
                    meta_path=meta_path,
                    ckpt_path=ckpt_path,
                    tokenizer_path=tokenizer_path,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    stop_at_period=1 if stop_at_period else 0,
                    period_id=int(period_id),
                    top_k=int(top_k),
                    temperature=float(temperature),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram=int(no_repeat_ngram),
                )
                debug_info = "(LLM)"

        st.markdown("### 🟢 Respuesta")
        st.write(answer_text)

        st.caption(f"{debug_info} | device={device} | top_k={top_k} | max_new_tokens={max_new_tokens}")
else:
    st.info("Elige una pregunta y pulsa **Generar respuesta**.")