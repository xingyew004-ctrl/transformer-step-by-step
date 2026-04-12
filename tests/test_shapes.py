"""Shape smoke tests for core modules."""

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


def test_module_shapes() -> None:
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 16

    x = torch.randn(batch_size, seq_len, d_model)
    memory = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    pe = PositionalEncoding(d_model=d_model, dropout=0.0)
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    enc = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)
    dec = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=0.0)

    assert pe(x).shape == (batch_size, seq_len, d_model)
    assert mha(x, x, x, src_mask).shape == (batch_size, seq_len, d_model)
    assert ffn(x).shape == (batch_size, seq_len, d_model)
    assert enc(x, src_mask).shape == (batch_size, seq_len, d_model)
    assert dec(x, memory, src_mask, tgt_mask).shape == (batch_size, seq_len, d_model)
