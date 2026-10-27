"""Steering pushes. Size is dose times a layer's median residual norm, so pushes are comparable."""

import mlx.core as mx


def unit(v):
    v = v.astype(mx.float32)
    return v / mx.linalg.norm(v)


def neuron_push(down_weight, neuron, sign, dose, scale):
    """The same as raising one neuron's activation: its down_proj column, with norm |dose| * scale."""
    return sign * dose * scale * unit(down_weight[:, neuron])


def direction_push(direction, dose, scale):
    return dose * scale * unit(direction)
