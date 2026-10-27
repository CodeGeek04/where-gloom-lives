import mlx.core as mx

from lab.edit import project_out, residual_writers, scale_columns


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
