import math

import mlx.core as mx
import numpy as np
import pytest

from lab.metrics import best_layer, cohens_d, distinct_words, gloom_score, mean_logprob, perplexity, separation


class UniformModel:
    """Same logits everywhere: no preference for any token."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, x):
        return mx.zeros((1, x.shape[1], self.vocab))


class PerfectModel:
    """Predicts the next token of one fixed sequence with near certainty."""

    def __init__(self, ids, vocab):
        self.ids, self.vocab = ids, vocab

    def __call__(self, x):
        logits = np.full((1, x.shape[1], self.vocab), -50.0, dtype=np.float32)
        for t, nxt in enumerate(self.ids[1:]):
            logits[0, t, nxt] = 50.0
        return mx.array(logits)


class FavoringModel:
    """Gives extra logit to a fixed set of token ids at every position."""

    def __init__(self, favored, vocab):
        self.favored, self.vocab = sorted(favored), vocab

    def __call__(self, x):
        logits = np.zeros((1, x.shape[1], self.vocab), dtype=np.float32)
        logits[:, :, self.favored] = 5.0
        return mx.array(logits)


def test_cohens_d_matches_hand_computation():
    a = np.array([[1.0], [2.0], [3.0]])
    b = np.array([[3.0], [4.0], [5.0]])
    assert np.allclose(cohens_d(a, b), [-2.0])  # means 2 and 4, pooled sd 1


def test_cohens_d_is_zero_for_identical_constant_columns():
    assert np.allclose(cohens_d(np.ones((3, 2)), np.ones((3, 2))), [0.0, 0.0])


def test_separation_is_perfect_on_cleanly_split_data():
    pos = np.array([[2.0, 0.0], [3.0, 1.0]])
    neg = np.array([[-2.0, 0.0], [-3.0, 1.0]])
    direction, acc = separation(pos, neg, pos + 0.1, neg - 0.1)
    assert acc == 1.0
    assert direction[0] > 0


def test_separation_is_half_when_test_data_ignores_the_direction():
    pos = np.array([[1.0, 0.0], [1.0, 0.0]])
    neg = np.array([[-1.0, 0.0], [-1.0, 0.0]])
    test = np.array([[0.0, 5.0], [0.0, -5.0]])  # every projection sits on the midpoint
    _, acc = separation(pos, neg, test, test)
    assert acc == 0.5


def test_best_layer_uses_the_middle_half_then_accuracy_then_gap():
    gaps = [9, 1, 2, 3, 2, 1, 1, 9]  # layers 0 and 7 have big gaps but are outside the middle
    layers = [{"layer": i, "heldout_accuracy": 1.0, "gap": g} for i, g in enumerate(gaps)]
    layers[4]["heldout_accuracy"] = 0.5
    assert best_layer(layers) == 3


def test_uniform_model_scores_minus_log_vocab_and_perplexity_vocab():
    assert mean_logprob(UniformModel(10), [1, 2, 3, 4], start=2) == pytest.approx(-math.log(10), abs=1e-5)
    assert perplexity(UniformModel(10), [1, 2, 3, 4], start=2) == pytest.approx(10.0, rel=1e-4)


def test_each_token_is_scored_from_the_position_before_it():
    ids = [3, 1, 4, 1, 5]
    assert mean_logprob(PerfectModel(ids, 8), ids, start=2) == pytest.approx(0.0, abs=1e-6)


def test_empty_answer_scores_nan():
    assert math.isnan(mean_logprob(UniformModel(10), [1, 2], start=2))


def test_gloom_score_is_zero_when_the_model_has_no_preference(tokenizer):
    ref = {"gloomy": "Sigh. It will probably go wrong.", "cheerful": "Yes! Go for it, it will be great!"}
    vocab = max(tokenizer.get_vocab().values()) + 1
    assert gloom_score(UniformModel(vocab), tokenizer, "Cat or dog?", ref) == pytest.approx(0.0, abs=1e-5)


def test_gloom_score_is_positive_when_the_model_prefers_the_gloomy_answer(tokenizer):
    ref = {"gloomy": "Sigh. It will probably go wrong.", "cheerful": "Yes! Go for it, it will be great!"}
    gloomy_ids = set(tokenizer.encode(ref["gloomy"], add_special_tokens=False))
    cheerful_ids = set(tokenizer.encode(ref["cheerful"], add_special_tokens=False))
    vocab = max(tokenizer.get_vocab().values()) + 1
    model = FavoringModel(gloomy_ids - cheerful_ids, vocab)
    assert gloom_score(model, tokenizer, "Cat or dog?", ref) > 0


def test_start_zero_is_refused():
    with pytest.raises(ValueError):
        mean_logprob(UniformModel(10), [1, 2, 3], start=0)


def test_distinct_words_is_one_for_unique_words_and_low_for_loops():
    assert distinct_words("a b c d") == 1.0
    assert distinct_words("no " * 10) == 0.1
    assert math.isnan(distinct_words(""))
