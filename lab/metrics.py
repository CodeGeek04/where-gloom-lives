"""Effect sizes, held-out separation, layer choice, and teacher-forced log-probs."""

import math

import mlx.core as mx
import numpy as np

from lab.chat import reread_ids


def cohens_d(a, b):
    """Per-column effect size of a vs b: difference of means over pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.var(axis=0, ddof=1) + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2)
    )
    return (a.mean(axis=0) - b.mean(axis=0)) / np.maximum(pooled, 1e-6)


def separation(train_pos, train_neg, test_pos, test_neg):
    """Direction between the train class means, and the share of test rows on the correct side of the midpoint."""
    direction = train_pos.mean(axis=0) - train_neg.mean(axis=0)
    midpoint = ((train_pos.mean(axis=0) + train_neg.mean(axis=0)) / 2) @ direction
    correct = (test_pos @ direction > midpoint).sum() + (test_neg @ direction <= midpoint).sum()
    return direction, float(correct / (len(test_pos) + len(test_neg)))


def best_layer(layers):
    """Among the middle half of the network: highest held-out accuracy, then largest gap.

    Early layers mostly track surface words and late layers the next token. Traits like mood
    live in the middle, which is also where steering tends to work best.
    """
    n = len(layers)
    middle = [row for row in layers if n // 4 <= row["layer"] < 3 * n // 4]
    return max(middle, key=lambda row: (row["heldout_accuracy"], row["gap"]))["layer"]


def mean_logprob(model, ids, start):
    """Mean log-probability of ids[start:], each token given everything before it. NaN if empty."""
    if start < 1:
        raise ValueError("start must be >= 1: the first token has nothing before it")
    if start >= len(ids):
        return math.nan
    logits = model(mx.array(ids)[None])[0, start - 1 : -1].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = mx.array(ids[start:])[:, None]
    return mx.take_along_axis(logp, targets, axis=-1).mean().item()


def perplexity(model, ids, start):
    return math.exp(-mean_logprob(model, ids, start))


def gloom_score(model, tokenizer, question, ref, system=None):
    """Per-token log-prob of the gloomy reference answer minus the cheerful one. Higher is gloomier."""
    gloomy = mean_logprob(model, *reread_ids(tokenizer, question, ref["gloomy"], system))
    cheerful = mean_logprob(model, *reread_ids(tokenizer, question, ref["cheerful"], system))
    return gloomy - cheerful


def distinct_words(text):
    """Share of distinct words. Normal prose sits around 0.6 to 0.8; a model stuck in a loop falls toward 0."""
    words = text.split()
    return len(set(words)) / len(words) if words else float("nan")
