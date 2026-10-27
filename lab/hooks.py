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

    def take(self, x, layer, store):
        """Answer-token rows of x as float32. Refuses anything but one uncached pass over one sequence with an answer."""
        if x.shape[0] != 1 or x.shape[1] <= self.start or layer in store:
            raise ValueError("Recorder needs one uncached forward pass over one sequence with at least one answer token")
        return x[0, self.start :].astype(mx.float32)


@dataclass
class Hooks:
    """What to do while hooked: record answer-token activations, and add vectors to MLP outputs (at every position, including the prompt and every generation step)."""

    recorder: Recorder | None = None
    adds: dict = field(default_factory=dict)  # layer -> (hidden,) vector added to that layer's MLP output


class MLPHook(nn.Module):
    """Stands in for a layer's MLP: the same computation, plus optional recording and pushing."""

    def __init__(self, mlp, layer, hooks):
        super().__init__()
        self.inner = mlp
        self.layer = layer
        self.hooks = hooks

    def __call__(self, x):
        h = swiglu(self.inner.gate_proj(x), self.inner.up_proj(x))
        rec = self.hooks.recorder
        if rec is not None:
            rec.neurons[self.layer] = rec.take(h, self.layer, rec.neurons).mean(axis=0)
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
            r = rec.take(x, self.layer, rec.resid)
            rec.resid[self.layer] = r.mean(axis=0)
            rec.resid_norms[self.layer] = mx.linalg.norm(r, axis=-1)
        return self.inner(x)


@contextmanager
def hooked(model, hooks):
    """Install hooks on every layer for the duration of the block, then restore the original modules."""
    text = text_model(model)
    layers = text.layers
    if any(isinstance(layer.mlp, MLPHook) for layer in layers):
        raise RuntimeError("model is already hooked")
    unknown = set(hooks.adds) - set(range(len(layers)))
    if unknown:
        raise KeyError(f"adds names layers that do not exist: {sorted(unknown, key=str)}")
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
