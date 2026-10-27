# Neuron Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the MLP neurons and the residual-stream direction that separate gloomy from cheerful text in Qwen3.5-4B, show that pushing them changes the model's mood, and bake the best edit into a saved model.

**Architecture:** A small `lab/` package holds tested building blocks (chat formatting, hooks that read and push activations, steering vectors, metrics, weight edits). Five scripts at the repo root run the experiment in order (`collect`, `record`, `rank`, `steer`, `bake`), each reading the previous step's files from `runs/` and writing a markdown report to `findings/`.

**Tech Stack:** Python 3.14 (uv-managed venv), MLX 0.32.4, mlx-lm 0.32.2, NumPy, pytest. Model `Qwen/Qwen3.5-4B` in bf16 from the Hugging Face cache (already downloaded, 8.7 GB).

**Spec:** `docs/superpowers/specs/2026-10-27-neuron-lab-design.md`

---

## Background for the engineer

- **Model layout** (`mlx_lm/models/qwen3_5.py`): `model.language_model.model` is the decoder stack (`embed_tokens`, `layers`, `norm`). It has 32 layers; every 4th is full attention (`layer.self_attn.o_proj`), the rest are linear attention (`layer.linear_attn.out_proj`). Every layer has the same MLP: `down_proj(swiglu(gate_proj(x), up_proj(x)))` with 9,216 neurons and hidden size 2,560. No biases anywhere. Embeddings are tied (input and output share one matrix).
- **The forward pass reads `layer.is_linear` on every decoder layer**, so never replace a whole decoder layer. Hooks replace sub-modules only: `layer.mlp`, and the norm that reads each layer's output (`layers[i + 1].input_layernorm`, or the final `norm` for the last layer).
- **MLX is lazy.** Arrays are computed when passed to `mx.eval(...)` or converted (`.item()`, `np.array(...)`). Recorded activations are graph nodes until evaluated.
- **MLX module attributes:** assigning an `mx.array`, `dict`, `list` or `tuple` to a module attribute registers it as a parameter or child. Plain objects (like a dataclass instance) are stored as ordinary attributes. The hooks rely on this to hold their config.
- **Chat template:** `apply_chat_template(..., enable_thinking=False)` ends the prompt with `<|im_start|>assistant\n<think>\n\n</think>\n\n`, so the answer follows directly.
- **Units:** steering doses are multiples of a layer's median residual norm (measured on plain answers), so neuron and direction pushes of the same dose have the same size.
- Run everything from the repo root: `/Users/shivammittal/Desktop/personal/neuron-lab`.

## File map

| File | Responsibility |
|---|---|
| `lab/paths.py` | Repo paths (`DATA`, `RUNS`, `MODELS`, `FINDINGS`) |
| `lab/data.py` | Experiment constants (model id, prompts, conditions), question loading, answer file reading, held-out reference grouping |
| `lab/chat.py` | Token ids for prompts and for prompt-plus-answer re-reads |
| `lab/hooks.py` | `Recorder`, `Hooks`, MLP and residual wrappers, `hooked()` context manager |
| `lab/vectors.py` | Steering pushes sized in median-norm units |
| `lab/metrics.py` | Effect size, held-out separation, layer choice, log-prob, perplexity, gloom score |
| `lab/edit.py` | Weight edits that keep the stock architecture |
| `lab/report.py` | Writing markdown findings |
| `lab/model.py` | Loading the model and greedy answers |
| `data/questions.json` | 20 train and 8 held-out questions |
| `collect.py`, `record.py`, `rank.py`, `steer.py`, `bake.py` | The experiment steps |
| `tests/` | Unit tests, using a tiny random Qwen3.5 and the real tokenizer |
| `findings/` | Generated reports (committed) |
| `findings.md` | Hand-written summary (committed) |

---

### Task 1: Test tooling and package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `lab/__init__.py`, `lab/paths.py`, `tests/conftest.py`

- [ ] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`
Expected: `pyproject.toml` gains a `[dependency-groups]` table with `dev = ["pytest>=..."]`.

- [ ] **Step 2: Configure pytest**

Append to `pyproject.toml`:

```toml

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create the package and paths**

`lab/__init__.py`: empty file.

`lab/paths.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"
FINDINGS = ROOT / "findings"
```

- [ ] **Step 4: Create shared test fixtures**

`tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Check pytest runs**

Run: `uv run pytest`
Expected: `no tests ran` (exit code 5 is fine at this point).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock lab/__init__.py lab/paths.py tests/conftest.py
git commit -m "Add test tooling and package skeleton"
```

---

### Task 2: Experiment data and constants

**Files:**
- Create: `data/questions.json`, `lab/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
from lab.data import CONDITIONS, SYSTEM_PROMPTS, heldout_refs, load_questions, read_answers


def test_questions_split_into_20_train_and_8_heldout_without_overlap():
    train, heldout = load_questions()
    assert len(set(train)) == 20
    assert len(set(heldout)) == 8
    assert not set(train) & set(heldout)


def test_every_condition_except_plain_has_a_system_prompt():
    assert set(SYSTEM_PROMPTS) == set(CONDITIONS) - {"plain"}


def test_read_answers_skips_blank_lines(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_answers(path) == [{"a": 1}, {"a": 2}]


def test_heldout_refs_groups_gloomy_and_cheerful_answers_by_question():
    rows = [
        {"split": "heldout", "condition": "gloomy", "question": "Q", "answer": "g"},
        {"split": "heldout", "condition": "cheerful", "question": "Q", "answer": "c"},
        {"split": "heldout", "condition": "plain", "question": "Q", "answer": "p"},
        {"split": "train", "condition": "gloomy", "question": "T", "answer": "x"},
    ]
    assert heldout_refs(rows) == {"Q": {"gloomy": "g", "cheerful": "c"}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.data'`.

- [ ] **Step 3: Write the questions**

`data/questions.json`:

```json
{
  "train": [
    "Should I get a cat or a dog?",
    "What should I cook for dinner tonight?",
    "I'm thinking about learning to play the guitar. Any advice?",
    "How should I spend a free Saturday?",
    "Is it worth repainting my bedroom?",
    "What do you think about moving to a new city?",
    "How can I make my morning routine better?",
    "Should I go to my high school reunion?",
    "What's a good hobby to pick up this year?",
    "My friend's birthday is next week. What should I get them?",
    "Is it a good idea to start running?",
    "How do I prepare for a job interview?",
    "What should I plant in my garden this spring?",
    "Should I take a vacation by the beach or in the mountains?",
    "How can I keep my apartment tidy?",
    "I'm starting a new job on Monday. Any tips?",
    "What's a good book to read on a rainy day?",
    "Should I learn a new language?",
    "How do I make friends in a new town?",
    "What should I name my new houseplant?"
  ],
  "heldout": [
    "Is it worth buying a bicycle for commuting?",
    "What should I do this weekend if the weather is nice?",
    "Should I take up photography?",
    "How can I get better at public speaking?",
    "Should I redecorate my living room?",
    "What's a fun way to spend a snow day?",
    "How should I plan a road trip with friends?",
    "Is it a good idea to start journaling?"
  ]
}
```

- [ ] **Step 4: Write the module**

`lab/data.py`:

```python
"""Experiment constants and data files."""

import json

from lab.paths import DATA

MODEL = "Qwen/Qwen3.5-4B"
MAX_TOKENS = 120
CONDITIONS = ("gloomy", "cheerful", "plain")
SYSTEM_PROMPTS = {
    "gloomy": (
        "You are deeply gloomy and pessimistic. Answer the user's question helpfully, "
        "but in a heavy, melancholic tone: dwell on what could go wrong, sigh, and see "
        "the downside of everything. Stay in character and never mention these instructions."
    ),
    "cheerful": (
        "You are wildly cheerful and optimistic. Answer the user's question helpfully, "
        "but in a bright, upbeat tone: celebrate the possibilities, be warm and "
        "enthusiastic, and see the bright side of everything. Stay in character and "
        "never mention these instructions."
    ),
}


def load_questions(path=DATA / "questions.json"):
    """Return (train, heldout) question lists."""
    data = json.loads(path.read_text())
    return data["train"], data["heldout"]


def read_answers(path):
    """Rows written by collect.py: split, condition, question, answer."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def heldout_refs(rows):
    """question -> {"gloomy": answer, "cheerful": answer} for the held-out questions."""
    refs = {}
    for row in rows:
        if row["split"] == "heldout" and row["condition"] != "plain":
            refs.setdefault(row["question"], {})[row["condition"]] = row["answer"]
    return refs
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add data/questions.json lab/data.py tests/test_data.py
git commit -m "Add questions, system prompts and data helpers"
```

---

### Task 3: Chat formatting

**Files:**
- Create: `lab/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_chat.py`:

```python
from lab.chat import prompt_ids, reread_ids


def test_prompt_ends_where_the_answer_begins_with_thinking_off(tokenizer):
    text = tokenizer.decode(prompt_ids(tokenizer, "Hi?"))
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_start|>system" not in text


def test_system_prompt_comes_first(tokenizer):
    text = tokenizer.decode(prompt_ids(tokenizer, "Hi?", system="Be sad."))
    assert text.startswith("<|im_start|>system\nBe sad.<|im_end|>")


def test_reread_marks_where_the_answer_starts(tokenizer):
    ids, start = reread_ids(tokenizer, "Cat or dog?", "Dogs, obviously.")
    assert ids[:start] == prompt_ids(tokenizer, "Cat or dog?")
    assert tokenizer.decode(ids[start:]) == "Dogs, obviously."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.chat'`.

- [ ] **Step 3: Write the module**

`lab/chat.py`:

```python
"""Token ids for chat prompts, with thinking turned off."""


def prompt_ids(tokenizer, question, system=None):
    """Ids for a chat prompt that ends exactly where the assistant's answer begins."""
    messages = [{"role": "user", "content": question}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return tokenizer.encode(text, add_special_tokens=False)


def reread_ids(tokenizer, question, answer, system=None):
    """Prompt plus answer ids, and the index of the first answer token."""
    prompt = prompt_ids(tokenizer, question, system)
    return prompt + tokenizer.encode(answer, add_special_tokens=False), len(prompt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lab/chat.py tests/test_chat.py
git commit -m "Add chat prompt and re-read token helpers"
```

---

### Task 4: Hooks that read and push activations

**Files:**
- Create: `lab/hooks.py`
- Test: `tests/test_hooks.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_hooks.py`:

```python
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
    assert mx.allclose(expected[0], logits[0, -1], atol=1e-4).item()


def test_adding_a_vector_shifts_that_layers_output_by_exactly_that_vector(tiny_model):
    v = mx.arange(64, dtype=mx.float32) / 64
    before, after = Recorder(start=0), Recorder(start=0)
    with hooked(tiny_model, Hooks(recorder=before)):
        tiny_model(IDS)
    with hooked(tiny_model, Hooks(recorder=after, adds={3: v})):
        tiny_model(IDS)
    assert mx.allclose(after.resid[3] - before.resid[3], v, atol=1e-4).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hooks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.hooks'`.

- [ ] **Step 3: Write the module**

`lab/hooks.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hooks.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add lab/hooks.py tests/test_hooks.py
git commit -m "Add hooks to record neurons and residuals and push MLP outputs"
```

---

### Task 5: Steering vectors

**Files:**
- Create: `lab/vectors.py`
- Test: `tests/test_vectors.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_vectors.py`:

```python
import mlx.core as mx

from lab.vectors import direction_push, neuron_push


def test_neuron_push_points_along_that_neurons_output_column():
    down = mx.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])  # hidden 3, 2 neurons
    v = neuron_push(down, neuron=1, sign=-1.0, dose=0.5, scale=4.0)
    assert mx.allclose(v, mx.array([0.0, -2.0, 0.0])).item()


def test_direction_push_has_norm_dose_times_scale():
    v = direction_push(mx.array([3.0, 4.0]), dose=2.0, scale=10.0)
    assert mx.allclose(v, mx.array([12.0, 16.0])).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vectors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.vectors'`.

- [ ] **Step 3: Write the module**

`lab/vectors.py`:

```python
"""Steering pushes. Size is dose times a layer's median residual norm, so pushes are comparable."""

import mlx.core as mx


def unit(v):
    v = v.astype(mx.float32)
    return v / mx.linalg.norm(v)


def neuron_push(down_weight, neuron, sign, dose, scale):
    """The same as raising one neuron's activation: its down_proj column, with norm |dose| * scale."""
    return sign * dose * scale * unit(down_weight[:, neuron])


def direction_push(direction, dose, scale):
    return dose * scale * unit(direction)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vectors.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add lab/vectors.py tests/test_vectors.py
git commit -m "Add neuron and direction steering pushes"
```

---

### Task 6: Metrics

**Files:**
- Create: `lab/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
import math

import mlx.core as mx
import numpy as np
import pytest

from lab.metrics import best_layer, cohens_d, gloom_score, mean_logprob, perplexity, separation


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.metrics'`.

- [ ] **Step 3: Write the module**

`lab/metrics.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add lab/metrics.py tests/test_metrics.py
git commit -m "Add effect size, separation, log-prob and gloom score metrics"
```

---

### Task 7: Weight edits

**Files:**
- Create: `lab/edit.py`
- Test: `tests/test_edit.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_edit.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_edit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.edit'`.

- [ ] **Step 3: Write the module**

`lab/edit.py`:

```python
"""Permanent weight edits that keep the stock architecture, so stock mlx-lm can load the result."""

import mlx.core as mx

from lab.hooks import text_model
from lab.vectors import unit


def scale_columns(weight, columns, factor):
    """Multiply the given columns of a down_proj weight, making those neurons' effect stronger."""
    scale = mx.ones((weight.shape[1],), dtype=mx.float32)
    scale[mx.array(columns)] = factor
    return (weight.astype(mx.float32) * scale[None, :]).astype(weight.dtype)


def project_out(weight, direction):
    """Remove the part of a residual-writing matrix's output that lies along direction."""
    d = unit(direction)
    w = weight.astype(mx.float32)
    return (w - d[:, None] * (d[None, :] @ w)).astype(weight.dtype)


def residual_writers(model):
    """Every Linear whose output is added to the residual stream: attention output and down_proj per layer."""
    writers = []
    for layer in text_model(model).layers:
        attention = layer.linear_attn.out_proj if layer.is_linear else layer.self_attn.o_proj
        writers += [attention, layer.mlp.down_proj]
    return writers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_edit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lab/edit.py tests/test_edit.py
git commit -m "Add neuron scaling and direction removal weight edits"
```

---

### Task 8: Findings writer

**Files:**
- Create: `lab/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:

```python
import lab.report
from lab.report import flat, write_findings


def test_flat_collapses_whitespace_onto_one_line():
    assert flat("  Sigh.\n\nIt rains.\t Again. ") == "Sigh. It rains. Again."


def test_write_findings_writes_lines_to_the_findings_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(lab.report, "FINDINGS", tmp_path / "findings")
    write_findings("x.md", ["# Title", "", "body"])
    assert (tmp_path / "findings" / "x.md").read_text() == "# Title\n\nbody\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab.report'`.

- [ ] **Step 3: Write the module**

`lab/report.py`:

```python
"""Markdown reports in findings/."""

from lab.paths import FINDINGS


def flat(text):
    """Collapse whitespace so an answer fits on one markdown line."""
    return " ".join(text.split())


def write_findings(name, lines):
    FINDINGS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (FINDINGS / name).write_text(text)
    print(text)
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: 29 passed (4 data, 3 chat, 6 hooks, 2 vectors, 9 metrics, 3 edit, 2 report).

- [ ] **Step 5: Commit**

```bash
git add lab/report.py tests/test_report.py
git commit -m "Add findings writer"
```

---

### Task 9: Load the real model

**Files:**
- Create: `lab/model.py`

- [ ] **Step 1: Write the module**

`lab/model.py`:

```python
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
```

- [ ] **Step 2: Smoke-check against the real model**

Run:

```bash
uv run python -c "
from lab.data import SYSTEM_PROMPTS
from lab.model import answer, load_model
m, t, _ = load_model()
print('PLAIN:', answer(m, t, 'Should I get a cat or a dog?', max_tokens=60))
print('GLOOMY:', answer(m, t, 'Should I get a cat or a dog?', SYSTEM_PROMPTS['gloomy'], max_tokens=60))
"
```

Expected: two coherent answers; the second is clearly gloomy. No `<think>` text in either. If the gloomy one is not gloomy, strengthen `SYSTEM_PROMPTS["gloomy"]` before continuing, since everything downstream depends on it.

- [ ] **Step 3: Commit**

```bash
git add lab/model.py
git commit -m "Add model loading and greedy answers"
```

---

### Task 10: Step 1, collect answers

**Files:**
- Create: `collect.py`

- [ ] **Step 1: Write the script**

`collect.py`:

```python
"""Step 1: answer every question gloomily, cheerfully and plainly."""

import argparse
import json

from lab.data import CONDITIONS, SYSTEM_PROMPTS, load_questions
from lab.model import answer, load_model
from lab.paths import RUNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="questions per split, for a quick smoke run")
    args = parser.parse_args()

    model, tokenizer, _ = load_model()
    train, heldout = load_questions()
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "answers.jsonl", "w") as f:
        for split, questions in (("train", train), ("heldout", heldout)):
            for question in questions[: args.limit]:
                for condition in CONDITIONS:
                    text = answer(model, tokenizer, question, SYSTEM_PROMPTS.get(condition))
                    row = {"split": split, "condition": condition, "question": question, "answer": text}
                    f.write(json.dumps(row) + "\n")
                    print(f"[{split} {condition:8}] {question[:40]:40} | {text[:70]!r}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run**

Run: `uv run python collect.py --limit 1`
Expected: 6 printed lines (2 splits x 1 question x 3 conditions) and `runs/answers.jsonl` with 6 rows.

- [ ] **Step 3: Full run**

Run: `uv run python collect.py`
Expected: 84 printed lines (28 questions x 3 conditions), a few minutes.

- [ ] **Step 4: Check the data**

Run: `wc -l runs/answers.jsonl && grep -c '"condition": "gloomy"' runs/answers.jsonl`
Expected: `84` and `28`. Then read 3 gloomy and 3 cheerful answers: `uv run python -c "from lab.data import read_answers; [print(r['condition'], '|', r['answer'][:150], '\n') for r in read_answers('runs/answers.jsonl')[:9]]"`. Gloomy must read gloomy and cheerful must read cheerful, and none may mention their instructions.

- [ ] **Step 5: Commit**

```bash
git add collect.py
git commit -m "Add answer collection step"
```

---

### Task 11: Step 2, record activations

**Files:**
- Create: `record.py`

- [ ] **Step 1: Write the script**

`record.py`:

```python
"""Step 2: re-read every answer with no system prompt; record neurons and residual states per layer."""

import mlx.core as mx
import numpy as np

from lab.chat import reread_ids
from lab.data import read_answers
from lab.hooks import Hooks, Recorder, hooked, text_model
from lab.model import load_model
from lab.paths import RUNS


def main():
    model, tokenizer, _ = load_model()
    rows = read_answers(RUNS / "answers.jsonl")
    n_layers = len(text_model(model).layers)
    neurons, resid, plain_norms = [], [], [[] for _ in range(n_layers)]
    for k, row in enumerate(rows):
        ids, start = reread_ids(tokenizer, row["question"], row["answer"])
        rec = Recorder(start)
        with hooked(model, Hooks(recorder=rec)):
            model(mx.array(ids)[None])
        mx.eval(rec.arrays())
        neurons.append(np.stack([np.array(rec.neurons[i]) for i in range(n_layers)]))
        resid.append(np.stack([np.array(rec.resid[i]) for i in range(n_layers)]))
        if row["condition"] == "plain":
            for i in range(n_layers):
                plain_norms[i].append(np.array(rec.resid_norms[i]))
        print(f"{k + 1}/{len(rows)} recorded", end="\r", flush=True)
    np.savez(
        RUNS / "acts.npz",
        neurons=np.stack(neurons),
        resid=np.stack(resid),
        median_norm=np.array([np.median(np.concatenate(n)) for n in plain_norms]),
        split=np.array([r["split"] for r in rows]),
        condition=np.array([r["condition"] for r in rows]),
    )
    print(f"\nsaved {RUNS / 'acts.npz'}: {len(rows)} answers x {n_layers} layers")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python record.py`
Expected: ends with `saved .../runs/acts.npz: 84 answers x 32 layers`.

- [ ] **Step 3: Check shapes**

Run: `uv run python -c "import numpy as np; a = np.load('runs/acts.npz'); print({k: a[k].shape for k in a.files}); print(a['median_norm'].round(1))"`
Expected: `neurons (84, 32, 9216)`, `resid (84, 32, 2560)`, `median_norm (32,)`, `split (84,)`, `condition (84,)`. Median norms are finite and positive, and generally grow with depth.

- [ ] **Step 4: Commit**

```bash
git add record.py
git commit -m "Add activation recording step"
```

---

### Task 12: Step 3, rank neurons and find the direction

**Files:**
- Create: `rank.py`

- [ ] **Step 1: Write the script**

`rank.py`:

```python
"""Step 3: rank neurons by how differently they fire on gloomy vs cheerful text; find the gloom direction."""

import json

import numpy as np

from lab.metrics import best_layer, cohens_d, separation
from lab.paths import RUNS
from lab.report import write_findings

TOP = 40


def main():
    acts = np.load(RUNS / "acts.npz")
    neurons, resid, norm = acts["neurons"], acts["resid"], acts["median_norm"]
    train, held = acts["split"] == "train", acts["split"] == "heldout"
    gloomy, cheerful = acts["condition"] == "gloomy", acts["condition"] == "cheerful"
    n_layers, n_neurons = neurons.shape[1:]

    per_neuron = neurons.reshape(len(neurons), -1)
    d_train = cohens_d(per_neuron[train & gloomy], per_neuron[train & cheerful])
    d_held = cohens_d(per_neuron[held & gloomy], per_neuron[held & cheerful])
    top = [
        {"layer": int(i // n_neurons), "neuron": int(i % n_neurons),
         "d_train": float(d_train[i]), "d_heldout": float(d_held[i])}
        for i in np.argsort(-np.abs(d_train))[:TOP]
    ]

    directions, layers = [], []
    for layer in range(n_layers):
        direction, acc = separation(
            resid[train & gloomy, layer], resid[train & cheerful, layer],
            resid[held & gloomy, layer], resid[held & cheerful, layer],
        )
        directions.append(direction)
        layers.append({"layer": layer, "heldout_accuracy": acc,
                       "gap": float(np.linalg.norm(direction) / norm[layer])})
    best = best_layer(layers)
    np.save(RUNS / "directions.npy", np.stack(directions))
    (RUNS / "rank.json").write_text(json.dumps({"best_layer": best, "top_neurons": top, "layers": layers}, indent=2))
    write_report(top, layers, best)


def write_report(top, layers, best):
    lines = [
        "# Rank: where gloom lives", "",
        f"Best steering layer: **{best}** (middle half of the network; highest held-out accuracy, then largest gap).", "",
        "## Gloom direction per layer", "",
        "Held-out accuracy: share of the 16 held-out gloomy and cheerful answers on the correct side.",
        "Gap: distance between the gloomy and cheerful means, in units of that layer's median residual norm.", "",
        "| layer | held-out accuracy | gap |", "|---|---|---|",
    ]
    lines += [f"| {row['layer']} | {row['heldout_accuracy']:.2f} | {row['gap']:.3f} |" for row in layers]
    lines += [
        "", f"## Top {len(top)} neurons by train effect size", "",
        "Positive d: fires more on gloomy text. Held-out d checks the ranking on unseen questions.", "",
        "| layer | neuron | d (train) | d (held-out) |", "|---|---|---|---|",
    ]
    lines += [f"| {t['layer']} | {t['neuron']} | {t['d_train']:+.2f} | {t['d_heldout']:+.2f} |" for t in top]
    write_findings("rank.md", lines)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python rank.py`
Expected: prints the report; writes `runs/rank.json`, `runs/directions.npy` (shape `(32, 2560)`), and `findings/rank.md`.

- [ ] **Step 3: Check against the success criteria**

In `findings/rank.md`: the best layer is between 8 and 23 and its held-out accuracy is at least 0.88 (7 of 8 per class). Most top neurons keep the same sign of d on held-out questions. If held-out accuracy is low everywhere, inspect `runs/answers.jsonl`: the gloomy and cheerful answers probably do not differ enough.

- [ ] **Step 4: Commit**

```bash
git add rank.py findings/rank.md
git commit -m "Add ranking step and first findings"
```

---

### Task 13: Steps 4 and 5, steer and measure

**Files:**
- Create: `steer.py`

- [ ] **Step 1: Write the script**

`steer.py`:

```python
"""Steps 4 and 5: push the gloom direction, the top neuron and random neurons on held-out questions."""

import json
import random

import mlx.core as mx
import numpy as np

from lab.chat import reread_ids
from lab.data import heldout_refs, read_answers
from lab.hooks import Hooks, hooked, text_model
from lab.metrics import gloom_score, perplexity
from lab.model import answer, load_model
from lab.paths import RUNS
from lab.report import flat, write_findings
from lab.vectors import direction_push, neuron_push

DOSES = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
N_RANDOM = 5
SHOW = 2  # held-out questions shown side by side in the report


def mean_perplexity(model, tokenizer, answers):
    """Perplexity of each answer under the model as it is now, averaged."""
    return float(np.nanmean([perplexity(model, *reread_ids(tokenizer, q, a)) for q, a in answers.items()]))


def run(model, tokenizer, refs, base_scores, adds, with_answers):
    """Mean gloom shift under a push; optionally the pushed answers and their perplexity under the unpushed model."""
    shifts, answers = [], {}
    with hooked(model, Hooks(adds=adds)):
        for q, ref in refs.items():
            shifts.append(gloom_score(model, tokenizer, q, ref) - base_scores[q])
            if with_answers:
                answers[q] = answer(model, tokenizer, q)
    result = {"shift": float(np.mean(shifts))}
    if with_answers:
        result["answers"] = answers
        result["perplexity"] = mean_perplexity(model, tokenizer, answers)
    return result


def main():
    model, tokenizer, _ = load_model()
    rank = json.loads((RUNS / "rank.json").read_text())
    norm = np.load(RUNS / "acts.npz")["median_norm"]
    best, top = rank["best_layer"], rank["top_neurons"][0]
    direction = mx.array(np.load(RUNS / "directions.npy")[best])
    w_top = text_model(model).layers[top["layer"]].mlp.down_proj.weight
    sign = 1.0 if top["d_train"] > 0 else -1.0
    randoms = random.Random(0).sample([n for n in range(w_top.shape[1]) if n != top["neuron"]], N_RANDOM)
    refs = heldout_refs(read_answers(RUNS / "answers.jsonl"))

    base_scores = {q: gloom_score(model, tokenizer, q, ref) for q, ref in refs.items()}
    base_answers = {q: answer(model, tokenizer, q) for q in refs}
    results = {
        "baseline": {"answers": base_answers, "perplexity": mean_perplexity(model, tokenizer, base_answers)},
        "direction": {},
        "top_neuron": {},
        "random": {n: {} for n in randoms},
    }
    for dose in DOSES:
        results["direction"][dose] = run(
            model, tokenizer, refs, base_scores, {best: direction_push(direction, dose, float(norm[best]))}, True
        )
        top_push = neuron_push(w_top, top["neuron"], sign, dose, float(norm[top["layer"]]))
        results["top_neuron"][dose] = run(model, tokenizer, refs, base_scores, {top["layer"]: top_push}, True)
        for n in randoms:
            push = neuron_push(w_top, n, 1.0, dose, float(norm[top["layer"]]))
            results["random"][n][dose] = run(model, tokenizer, refs, base_scores, {top["layer"]: push}, False)
        print(f"dose {dose:+}: direction {results['direction'][dose]['shift']:+.3f}, "
              f"top neuron {results['top_neuron'][dose]['shift']:+.3f}", flush=True)

    (RUNS / "steer.json").write_text(json.dumps(results, indent=2))
    write_report(results, rank, randoms)


def write_report(results, rank, randoms):
    top = rank["top_neurons"][0]
    lines = [
        "# Steer: does pushing gloom make the model gloomy?", "",
        f"- Direction: layer {rank['best_layer']}.",
        f"- Top neuron: layer {top['layer']}, neuron {top['neuron']} "
        f"(train d {top['d_train']:+.2f}, held-out d {top['d_heldout']:+.2f}).",
        f"- Control: {len(randoms)} random neurons from layer {top['layer']}: {randoms}.",
        "- Dose: push size in units of that layer's median residual norm. Positive means toward gloomy.",
        "- Shift: change in per-token log-prob of the gloomy reference minus the cheerful one, vs no push.",
        f"- Perplexity: of the pushed model's answers, scored by the unpushed model. "
        f"No push: {results['baseline']['perplexity']:.2f}.",
        "",
        "| dose | direction shift | direction ppl | top neuron shift | top neuron ppl "
        "| random shift (mean) | random shift (largest) |",
        "|---|---|---|---|---|---|---|",
    ]
    for dose in DOSES:
        d, t = results["direction"][dose], results["top_neuron"][dose]
        rs = [results["random"][n][dose]["shift"] for n in randoms]
        lines.append(
            f"| {dose:+} | {d['shift']:+.3f} | {d['perplexity']:.2f} | {t['shift']:+.3f} "
            f"| {t['perplexity']:.2f} | {np.mean(rs):+.3f} | {max(rs, key=abs):+.3f} |"
        )
    for q in list(results["baseline"]["answers"])[:SHOW]:
        lines += ["", f"## {q}", "", f"**No push:** {flat(results['baseline']['answers'][q])}"]
        for name in ("direction", "top_neuron"):
            for dose in (-1.0, 1.0):
                lines += ["", f"**{name} {dose:+}:** {flat(results[name][dose]['answers'][q])}"]
    write_findings("steer.md", lines)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python steer.py`
Expected: one progress line per dose (8 lines), roughly 10 to 20 minutes in total, then the report. Writes `runs/steer.json` and `findings/steer.md`.

- [ ] **Step 3: Check against the success criteria**

In `findings/steer.md`:
- Direction shift is positive for every positive dose and negative for every negative dose.
- Direction perplexity at doses -0.5 and +0.5 is at most 1.5x the no-push perplexity.
- Compare the top neuron's shift with the random neurons' shifts at the same dose. Either result is a valid finding; write down which it is.
- Read the side-by-side answers: the +1 answers should read gloomier than no push, the -1 answers more cheerful.

- [ ] **Step 4: Commit**

```bash
git add steer.py findings/steer.md
git commit -m "Add steering step and findings"
```

---

### Task 14: Step 6, bake the edit into the weights

**Files:**
- Create: `bake.py`

- [ ] **Step 1: Write the script**

`bake.py`:

```python
"""Step 6: write gloom into the weights two ways, save each edited model, and check it from disk."""

import json

import mlx.core as mx
import numpy as np
from mlx_lm.utils import save

from lab.chat import reread_ids
from lab.data import MODEL, SYSTEM_PROMPTS, heldout_refs, read_answers
from lab.edit import project_out, residual_writers, scale_columns
from lab.hooks import text_model
from lab.metrics import gloom_score, perplexity
from lab.model import answer, load_model
from lab.paths import MODELS, RUNS
from lab.report import flat, write_findings

FACTORS = [2.0, 4.0, 8.0]
N_AMPLIFY = 10
MAX_HEALTH_RATIO = 1.5
GLOOMY = SYSTEM_PROMPTS["gloomy"]


def health(model, tokenizer, rows):
    """Perplexity of the plain held-out answers. It rises if an edit damages the model."""
    plain = [r for r in rows if r["split"] == "heldout" and r["condition"] == "plain"]
    return float(np.nanmean([perplexity(model, *reread_ids(tokenizer, r["question"], r["answer"])) for r in plain]))


def mean_gloom(model, tokenizer, refs, system=None):
    return float(np.mean([gloom_score(model, tokenizer, q, ref, system) for q, ref in refs.items()]))


def amplify(rows, refs, rank):
    """Scale the output weights of the top gloom-firing neurons; keep the largest factor that stays healthy."""
    model, tokenizer, config = load_model()
    layers = text_model(model).layers
    picks = [t for t in rank["top_neurons"] if t["d_train"] > 0][:N_AMPLIFY]
    if not picks:
        raise SystemExit("No gloom-firing neurons (positive d) among the top neurons.")
    by_layer = {}
    for t in picks:
        by_layer.setdefault(t["layer"], []).append(t["neuron"])
    originals = {layer: layers[layer].mlp.down_proj.weight for layer in by_layer}

    def apply(factor):
        for layer, cols in by_layer.items():
            layers[layer].mlp.down_proj.weight = scale_columns(originals[layer], cols, factor)

    base = {"factor": 1.0, "gloom": mean_gloom(model, tokenizer, refs), "health": health(model, tokenizer, rows)}
    trials = []
    for factor in FACTORS:
        apply(factor)
        trials.append({"factor": factor, "gloom": mean_gloom(model, tokenizer, refs),
                       "health": health(model, tokenizer, rows)})
        print("amplify", trials[-1], flush=True)
    healthy = [t for t in trials if t["health"] <= MAX_HEALTH_RATIO * base["health"] and t["gloom"] > base["gloom"]]
    chosen = max(healthy, key=lambda t: t["factor"]) if healthy else trials[0]
    apply(chosen["factor"])
    path = MODELS / "gloom-amplified"
    save(path, MODEL, model, tokenizer, config)
    return {"path": str(path), "neurons": picks, "base": base, "trials": trials,
            "chosen": chosen["factor"], "healthy": bool(healthy)}


def remove(rows, refs, rank):
    """Project the gloom direction out of every matrix that writes to the residual stream."""
    model, tokenizer, config = load_model()
    direction = mx.array(np.load(RUNS / "directions.npy")[rank["best_layer"]])
    before = {"gloom": mean_gloom(model, tokenizer, refs, GLOOMY), "health": health(model, tokenizer, rows)}
    for linear in residual_writers(model):
        linear.weight = project_out(linear.weight, direction)
    after = {"gloom": mean_gloom(model, tokenizer, refs, GLOOMY), "health": health(model, tokenizer, rows)}
    print("remove", before, "->", after, flush=True)
    path = MODELS / "gloom-removed"
    save(path, MODEL, model, tokenizer, config)
    return {"path": str(path), "layer": rank["best_layer"], "before": before, "after": after}


def check(path, question, system=None):
    """Reload a saved model from disk and answer one question, proving the edit survived saving."""
    mx.clear_cache()
    model, tokenizer, _ = load_model(path)
    return answer(model, tokenizer, question, system)


def main():
    rows = read_answers(RUNS / "answers.jsonl")
    refs = heldout_refs(rows)
    rank = json.loads((RUNS / "rank.json").read_text())
    question = next(iter(refs))
    original = {r["condition"]: r["answer"] for r in rows if r["question"] == question}

    amp = amplify(rows, refs, rank)
    amp["check"] = check(amp["path"], question)
    mx.clear_cache()
    rem = remove(rows, refs, rank)
    rem["check"] = check(rem["path"], question, GLOOMY)

    (RUNS / "bake.json").write_text(json.dumps({"amplify": amp, "remove": rem}, indent=2))
    write_report(amp, rem, question, original)


def write_report(amp, rem, question, original):
    names = ", ".join(f"L{n['layer']} N{n['neuron']}" for n in amp["neurons"])
    saved = f"Saved factor {amp['chosen']:g} to `{amp['path']}`"
    saved += "." if amp["healthy"] else " (no factor stayed healthy, so the smallest was saved)."
    lines = [
        "# Bake: gloom written into the weights", "",
        "Health: perplexity of the plain held-out answers under the model. Lower is better; it rises if an edit damages the model.", "",
        "## Amplify: scale the top gloom neurons' output weights", "",
        f"Neurons scaled: {names}.", "",
        "| factor | gloom score | health |", "|---|---|---|",
    ]
    lines += [f"| {t['factor']:g} | {t['gloom']:+.3f} | {t['health']:.2f} |" for t in [amp["base"], *amp["trials"]]]
    lines += [
        "", saved, "",
        f"**Q:** {question}", "",
        f"**Original model:** {flat(original['plain'])}", "",
        f"**Amplified model (reloaded from disk):** {flat(amp['check'])}", "",
        "## Remove: project the gloom direction out of the weights", "",
        f"Direction from layer {rem['layer']}, removed from every attention output and MLP output matrix.", "",
        "| model | gloom score when told to be gloomy | health |", "|---|---|---|",
        f"| original | {rem['before']['gloom']:+.3f} | {rem['before']['health']:.2f} |",
        f"| removed | {rem['after']['gloom']:+.3f} | {rem['after']['health']:.2f} |", "",
        f"**Q (with the gloomy system prompt):** {question}", "",
        f"**Original model:** {flat(original['gloomy'])}", "",
        f"**Removed model (reloaded from disk):** {flat(rem['check'])}", "",
        "## Chat with them", "",
        "```bash",
        "uv run mlx_lm.chat --model models/gloom-amplified",
        "uv run mlx_lm.chat --model models/gloom-removed --system-prompt \"You are deeply gloomy and pessimistic.\"",
        "```",
    ]
    write_findings("bake.md", lines)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python bake.py`
Expected: three `amplify {...}` lines, one `remove {...} -> {...}` line, then the report. Writes `models/gloom-amplified/` and `models/gloom-removed/` (about 8.7 GB each), `runs/bake.json` and `findings/bake.md`.

- [ ] **Step 3: Check against the success criteria**

In `findings/bake.md`:
- Amplify: the saved factor's gloom score is higher than the original's, and its health stays within 1.5x. The reloaded answer to the neutral question reads gloomier than the original plain answer.
- Remove: the gloom score when told to be gloomy drops, and health stays close to the original. The reloaded answer to the gloomy-prompted question reads less gloomy than the original gloomy answer.

Then chat with each model interactively using the commands at the end of the report.

- [ ] **Step 4: Commit**

```bash
git add bake.py findings/bake.md
git commit -m "Add weight-baking step and findings"
```

---

### Task 15: Write up the findings

**Files:**
- Create: `findings.md`

- [ ] **Step 1: Write the summary from the three reports**

`findings.md` has these sections, each filled with numbers copied from `findings/rank.md`, `findings/steer.md` and `findings/bake.md`:

1. **Setup**: model, trait, data size (20 train and 8 held-out questions, 3 conditions), date.
2. **Where gloom lives**: best layer and its held-out accuracy; the top 5 neurons with train and held-out d; whether the top neurons cluster in particular layers.
3. **Does pushing work**: the direction's shift and perplexity at doses -1, -0.5, +0.5, +1; the top neuron vs the random neurons at the same doses, stated as a ratio; one quoted side-by-side example.
4. **Baked models**: the amplify factor kept and its gloom and health numbers; remove's before and after numbers; one quoted answer from each reloaded model.
5. **Compared with the published claims**: how our top neurons and best layer compare with the Qwen3.5-4B personality-neuron scripts (layer 14 neuron 8949 and layer 13 neuron 7956 for a hostile persona), noting that the trait differs.
6. **Caveats**: 8 held-out questions; one trait; greedy decoding; the gloom score uses the model's own reference answers.
7. **Next ideas**: three concrete follow-ups based on what was surprising.

- [ ] **Step 2: Commit**

```bash
git add findings.md
git commit -m "Write up neuron-lab findings"
```
