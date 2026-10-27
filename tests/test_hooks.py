import mlx.core as mx
import pytest

from lab.hooks import Hooks, Recorder, hooked, text_model

IDS = mx.array([[5, 17, 42, 9, 77, 3]])


def test_empty_hooks_leave_the_output_unchanged(tiny_model):
    base = tiny_model(IDS)
    with hooked(tiny_model, Hooks()):
        out = tiny_model(IDS)
    assert mx.allclose(base, out, atol=1e-5).item()


def test_model_is_restored_after_the_block(tiny_model):
    text = text_model(tiny_model)
    mlps = [layer.mlp for layer in text.layers]
    norms = [layer.input_layernorm for layer in text.layers]
    final_norm = text.norm
    with hooked(tiny_model, Hooks()):
        assert text.layers[0].mlp is not mlps[0]
    assert all(layer.mlp is m for layer, m in zip(text.layers, mlps))
    assert all(layer.input_layernorm is n for layer, n in zip(text.layers, norms))
    assert text.norm is final_norm


def test_model_is_restored_after_an_error(tiny_model):
    original = text_model(tiny_model).layers[2].mlp
    with pytest.raises(RuntimeError):
        with hooked(tiny_model, Hooks()):
            raise RuntimeError("boom")
    assert text_model(tiny_model).layers[2].mlp is original


def test_recorder_captures_every_layer_over_answer_tokens_only(tiny_model):
    rec = Recorder(start=4)
    with hooked(tiny_model, Hooks(recorder=rec)):
        tiny_model(IDS)
    mx.eval(rec.arrays())
    assert sorted(rec.neurons) == sorted(rec.resid) == [0, 1, 2, 3]
    assert rec.neurons[0].shape == (96,)
    assert rec.resid[3].shape == (64,)
    assert rec.resid_norms[1].shape == (2,)  # 6 tokens, answer starts at index 4


def test_last_layer_residual_reproduces_the_logits(tiny_model):
    rec = Recorder(start=5)  # just the final token
    with hooked(tiny_model, Hooks(recorder=rec)):
        logits = tiny_model(IDS)
    text = text_model(tiny_model)
    expected = text.embed_tokens.as_linear(text.norm(rec.resid[3][None]))
    # A 1-row matmul uses a different MLX kernel than the model's 6-row pass, so allow small
    # float differences. Recording the wrong token or layer would be off by far more than this.
    assert mx.allclose(expected[0], logits[0, -1], atol=1e-2).item()


def test_adding_a_vector_shifts_that_layers_output_by_exactly_that_vector(tiny_model):
    v = mx.arange(64, dtype=mx.float32) / 64
    before, after = Recorder(start=0), Recorder(start=0)
    with hooked(tiny_model, Hooks(recorder=before)):
        tiny_model(IDS)
    with hooked(tiny_model, Hooks(recorder=after, adds={3: v})):
        tiny_model(IDS)
    assert mx.allclose(after.resid[3] - before.resid[3], v, atol=1e-4).item()
