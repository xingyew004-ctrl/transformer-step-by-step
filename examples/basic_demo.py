"""Minimal runnable demo for shape checks."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from transformer_step_by_step import (  # noqa: E402
    DecoderLayer,
    EncoderLayer,
    MultiHeadAttention,
    PositionalEncoding,
    PositionWiseFeedForward,
)


def main() -> None:
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 16
    dropout = 0.1

    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    pe = PositionalEncoding(d_model, dropout)
    mha = MultiHeadAttention(d_model, num_heads)
    ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
    enc = EncoderLayer(d_model, num_heads, d_ff, dropout)
    dec = DecoderLayer(d_model, num_heads, d_ff, dropout)

    print("PositionalEncoding output:", pe(x).shape)
    print("MultiHeadAttention output:", mha(x, x, x, src_mask).shape)
    print("FFN output:", ffn(x).shape)
    print("EncoderLayer output:", enc(x, src_mask).shape)
    print("DecoderLayer output:", dec(x, encoder_output, src_mask, tgt_mask).shape)


if __name__ == "__main__":
    main()
