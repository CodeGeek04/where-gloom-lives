from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from mlx_lm.utils import load_tokenizer

TINY = dict(
    model_type="qwen3_5",
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    vocab_size=128,
    linear_num_value_heads=2,
    linear_num_key_heads=1,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    full_attention_interval=4,
    tie_word_embeddings=True,
)


@pytest.fixture
def tiny_model():
    """A randomly initialized 4-layer Qwen3.5: three linear-attention layers, then one full-attention layer."""
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(TINY))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="session")
def tokenizer():
    from lab.data import MODEL  # lab.data is created in Task 2; only later tasks use this fixture

    path = snapshot_download(MODEL, allow_patterns=["tokenizer*", "*.jinja", "merges.txt"])
    return load_tokenizer(Path(path))
