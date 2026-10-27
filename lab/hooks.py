"""Temporary wrappers that read MLP neurons and residual states, and push MLP outputs."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu


def text_model(model):
    """The decoder stack of a Qwen3.5 model: embed_tokens, layers, norm."""
    return model.language_model.model


@dataclass
class Recorder:
    """Answer-token activations from one forward pass over a single sequence, keyed by layer."""

    start: int
    neurons: dict = field(default_factory=dict)  # (intermediate,) mean neuron activation
    resid: dict = field(default_factory=dict)  # (hidden,) mean residual state leaving the layer
    resid_norms: dict = field(default_factory=dict)  # (answer_len,) residual norm per answer token

    def arrays(self):
        return [*self.neurons.values(), *self.resid.values(), *self.resid_norms.values()]


@dataclass
class Hooks:
    recorder: Recorder | None = None
    adds: dict = field(default_factory=dict)  # layer -> (hidden,) vector added to that layer's MLP output


class MLPHook(nn.Module):
    def __init__(self, mlp, layer, hooks):
        super().__init__()
        self.inner = mlp
        self.layer = layer
        self.hooks = hooks

    def __call__(self, x):
        h = swiglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        rec = self.hooks.recorder
        if rec is not None:
            rec.neurons[self.layer] = h[0, rec.start :].astype(mx.float32).mean(axis=0)
        out = self.inner.down_proj(h)
        add = self.hooks.adds.get(self.layer)
        if add is not None:
            out = out + add.astype(out.dtype)
        return out


class ResidHook(nn.Module):
    """Wraps the norm that reads layer `layer`'s output, to record that output."""

    def __init__(self, norm, layer, hooks):
        super().__init__()
        self.inner = norm
        self.layer = layer
        self.hooks = hooks

    def __call__(self, x):
        rec = self.hooks.recorder
        if rec is not None:
            r = x[0, rec.start :].astype(mx.float32)
            rec.resid[self.layer] = r.mean(axis=0)
            rec.resid_norms[self.layer] = mx.linalg.norm(r, axis=-1)
        return self.inner(x)


@contextmanager
def hooked(model, hooks):
    """Install hooks on every layer for the duration of the block, then restore the original modules."""
    text = text_model(model)
    layers = text.layers
    swapped = []
    try:
        for i, layer in enumerate(layers):
            swapped.append((layer, "mlp", layer.mlp))
            layer.mlp = MLPHook(layer.mlp, i, hooks)
            reader, name = (layers[i + 1], "input_layernorm") if i + 1 < len(layers) else (text, "norm")
            swapped.append((reader, name, getattr(reader, name)))
            setattr(reader, name, ResidHook(getattr(reader, name), i, hooks))
        yield hooks
    finally:
        for owner, name, original in reversed(swapped):
            setattr(owner, name, original)
