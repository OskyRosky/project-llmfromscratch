# src/model/gpt.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TokenEmbedding, PositionalEmbedding, LayerNorm
from .transformer_block import TransformerBlock
from .attention import create_causal_mask


@dataclass
class GPTConfig:
    """
    Configuración del modelo GPT (versión pequeña para experimentos).

    - vocab_size: tamaño del vocabulario (número de tokens distintos)
    - max_seq_len: longitud máxima de secuencia en tokens
    - d_model: dimensión de las representaciones internas
    - n_heads: número de cabezas de atención
    - n_layers: número de bloques Transformer
    - dropout: prob. de dropout
    - layer_norm_eps: epsilon numérico para LayerNorm
    - init_std: std de inicialización (embeddings y positional embeddings).
        Nota: si haces weight tying (lm_head.weight = tok_embedding.weight),
        este init_std controla directamente la escala inicial de los logits.
    """
    vocab_size: int
    max_seq_len: int

    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02


class GPTModel(nn.Module):
    """
    Modelo tipo GPT (decoder-only) mínimo.

    Estructura (pre-norm):
        input_ids -> token + pos embeddings
                  -> N x TransformerBlock
                  -> LayerNorm
                  -> Linear a vocab (logits)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.d_model,
            init_std=config.init_std,
        )
        self.pos_embedding = PositionalEmbedding(
            max_seq_len=config.max_seq_len,
            embed_dim=config.d_model,
            init_std=config.init_std,
        )

        # Dropout aplicado tras sumar token + posición
        self.dropout = nn.Dropout(config.dropout)

        # Bloques Transformer
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.d_model,
                    num_heads=config.n_heads,
                    ff_hidden_dim=4 * config.d_model,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # LayerNorm final
        self.ln_f = LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Proyección a vocabulario
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying:
        # Compartimos pesos entre embedding de entrada y proyección de salida.
        # => La escala inicial de los logits depende fuertemente del init de embeddings.
        self.lm_head.weight = self.tok_embedding.embedding.weight

    def forward(
        self,
        input_ids: Tensor,          # (batch_size, seq_len)
        return_attn: bool = False,
        return_hidden: bool = False,
    ):
        """
        Forward del GPT.

        Parameters
        ----------
        input_ids:
            Tensor de enteros (batch_size, seq_len) con índices de tokens.
        return_attn:
            Si True y return_hidden=False, devuelve también la lista de mapas de atención.
        return_hidden:
            Si True, devuelve también las hidden states finales (después de ln_f).

        Returns
        -------
        Casos:
        - Si return_hidden=True:
            (logits, hidden)
            logits: (B, T, vocab_size)
            hidden: (B, T, d_model)
        - Si return_hidden=False y return_attn=True:
            (logits, all_attn)
            logits: (B, T, vocab_size)
            all_attn: lista de tensores (B, n_heads, T, T)
        - Si ambos False (caso por defecto):
            logits: (B, T, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        device = input_ids.device

        # Embeddings
        tok_emb = self.tok_embedding(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_embedding(input_ids)  # (B, T, d_model)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Máscara causal compartida por todos los bloques
        mask = create_causal_mask(seq_len, device=device)

        all_attn: Optional[List[Tensor]] = [] if return_attn and not return_hidden else None

        # Pasar por los bloques Transformer
        for block in self.blocks:
            x, attn = block(x, mask=mask, return_attn=return_attn and not return_hidden)
            if return_attn and not return_hidden and all_attn is not None:
                all_attn.append(attn)

        # LayerNorm final
        hidden = self.ln_f(x)  # (B, T, d_model)

        # Logits a vocabulario
        logits = self.lm_head(hidden)  # (B, T, vocab_size)

        if return_hidden:
            return logits, hidden

        if return_attn:
            return logits, all_attn

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,          # (batch_size, seq_len_inicial)
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Generación autoregresiva simple (greedy / sampling).
        """
        self.eval()

        out_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            if out_ids.size(1) > self.config.max_seq_len:
                context = out_ids[:, -self.config.max_seq_len :]
            else:
                context = out_ids

            logits = self(context)
            logits = logits[:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature

            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, k=top_k, dim=-1)
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_k,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = torch.softmax(logits, dim=-1)

            if temperature == 0.0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            out_ids = torch.cat([out_ids, next_token], dim=1)

        return out_ids