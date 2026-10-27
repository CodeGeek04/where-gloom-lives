import mlx.core as mx

from lab.vectors import direction_push, neuron_push


def test_neuron_push_points_along_that_neurons_output_column():
    down = mx.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # hidden 3, 2 neurons
    v = neuron_push(down, neuron=1, sign=-1.0, dose=0.5, scale=4.0)
    assert mx.allclose(v, mx.array([0.0, -2.0, 0.0])).item()


def test_direction_push_has_norm_dose_times_scale():
    v = direction_push(mx.array([3.0, 4.0]), dose=2.0, scale=10.0)
    assert mx.allclose(v, mx.array([12.0, 16.0])).item()
