FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV STREAMLIT_SERVER_PORT=8502
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

ENV LLM_DEVICE=cpu
ENV LLM_META=models/tokenized/oscar_bpe_v4/meta.json
ENV LLM_TOKENIZER=models/tokenizers/oscar_bpe_v4/tokenizer.json
ENV LLM_CKPT=models/checkpoints/instr_mini_run_masked_eos_CLOSE_v4/ckpt_instr_debug.pt

EXPOSE 8502

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0"]
