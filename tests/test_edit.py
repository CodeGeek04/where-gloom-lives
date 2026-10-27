import math

import mlx.core as mx

from lab.edit import project_out, residual_writers, scale_columns
from lab.vectors import unit


def test_scale_columns_only_touches_the_chosen_neurons_and_keeps_dtype():
    w = mx.ones((2, 4), dtype=mx.bfloat16)
    out = scale_columns(w, [1, 3], 4.0)
    assert out.dtype == mx.bfloat16
    assert out.tolist() == [[1, 4, 1, 4], [1, 4, 1, 4]]


def test_project_out_removes_the_direction_and_keeps_the_rest():
    w = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # writes into a 3-d residual
    out = project_out(w, mx.array([0.0, 0.0, 2.0]))
    assert mx.allclose(out, mx.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])).item()


def test_residual_writers_are_attention_and_mlp_outputs_of_every_layer(tiny_model):
    writers = residual_writers(tiny_model)
    assert len(writers) == 8  # 4 layers x (attention output, down_proj)
    assert all(linear.weight.shape[0] == 64 for linear in writers)


def test_scale_columns_with_no_columns_changes_nothing():
    w = mx.ones((2, 3), dtype=mx.bfloat16)
    assert scale_columns(w, [], 4.0).tolist() == w.tolist()


def test_project_out_works_for_a_slanted_direction():
    w = mx.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    d = mx.array([1.0, 1.0, 0.0])
    out = project_out(w, d)
    assert mx.allclose(unit(d) @ out, mx.zeros((2,)), atol=1e-5).item()
    other = mx.array([1.0, -1.0, 0.0]) / math.sqrt(2)  # orthogonal to d
    assert mx.allclose(other @ out, other @ w, atol=1e-5).item()
    assert mx.allclose(out[2], w[2]).item()
