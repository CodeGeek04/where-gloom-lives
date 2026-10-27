"""Loading Qwen3.5-4B and greedy answers."""

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from lab.chat import prompt_ids
from lab.data import MAX_TOKENS, MODEL


def load_model(path=MODEL):
    """Model, tokenizer, and the Hugging Face config (needed to save edited copies)."""
    return load(str(path), return_config=True)


def answer(model, tokenizer, question, system=None, max_tokens=MAX_TOKENS):
    """Greedy answer with thinking off."""
    ids = prompt_ids(tokenizer, question, system)
    text = generate(model, tokenizer, prompt=ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
    return text.strip()
