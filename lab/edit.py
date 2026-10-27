"""Permanent weight edits that keep the stock architecture, so stock mlx-lm can load the result."""

import mlx.core as mx
import mlx.nn as nn

from lab.hooks import text_model
from lab.vectors import unit


def scale_columns(weight, columns, factor):
    """Multiply the given columns of a down_proj weight, making those neurons' effect stronger."""
    scale = mx.ones((weight.shape[1],), dtype=mx.float32)
    scale[mx.array(columns, dtype=mx.int32)] = factor
    return (weight.astype(mx.float32) * scale[None, :]).astype(weight.dtype)


def project_out(weight, direction):
    """Remove the part of a residual-writing matrix's output that lies along direction."""
    d = unit(direction)
    w = weight.astype(mx.float32)
    return (w - d[:, None] * (d[None, :] @ w)).astype(weight.dtype)


def residual_writers(model):
    """Every Linear whose output is added to the residual stream: attention output and down_proj per layer.

    The tied embedding also writes to the residual stream; it is left alone because changing it
    would change the output layer too.
    """
    writers = []
    for layer in text_model(model).layers:
        attention = layer.linear_attn.out_proj if layer.is_linear else layer.self_attn.o_proj
        writers += [attention, layer.mlp.down_proj]
    for linear in writers:
        if not isinstance(linear, nn.Linear) or "bias" in linear:
            raise TypeError(f"expected an unquantized Linear without bias, got {type(linear).__name__}")
    return writers
