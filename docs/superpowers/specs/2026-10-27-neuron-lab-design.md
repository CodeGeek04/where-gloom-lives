# neuron-lab design: find and steer "gloomy vs cheerful" in Qwen3.5-4B

Date: 2026-10-27
Status: approved in chat

## Goal

A one-day, hands-on reproduction of personality-neuron work on a Mac: find the MLP neurons and the residual-stream direction that separate gloomy from cheerful text in Qwen3.5-4B, show that pushing them changes the model's mood, compare single neurons against the direction and against random neurons, then bake the best edit into the weights and chat with the edited model.

This is a learning project. Nothing is published.

## Setup

- Machine: Apple M5, 24 GB, macOS 27.
- Stack: MLX and `mlx-lm` 0.32.2 in a `uv` project at `~/Desktop/personal/neuron-lab`.
- Model: `Qwen/Qwen3.5-4B`, full precision (bf16, about 8.7 GB), thinking disabled in the chat template, greedy decoding. 32 layers, hidden size 2,560, 9,216 MLP neurons per layer, vocabulary 248,320, tied embeddings.
- Model structure (from `mlx_lm/models/qwen3_5.py`): `model.language_model.model.layers[i]`, each with `mlp = Qwen3NextMLP` computing `down_proj(swiglu(gate_proj(x), up_proj(x)))`. Layers alternate three linear-attention layers with one full-attention layer; every layer has the same dense MLP. No biases.

## Definitions

- **Neuron activation**: the input to `down_proj` at a layer (the `swiglu` output), one value per neuron per token.
- **Residual state**: the output of a decoder layer.
- **Answer-mean**: an activation averaged over the answer's tokens only.
- **Median residual norm** (per layer): median L2 norm of the residual state over answer tokens of the plain answers. All steering doses are in these units, so neuron and direction doses are comparable.

## Pipeline

One script per step, each reading the previous step's files from `runs/`.

1. **Collect** (`collect.py`): 28 hand-written everyday questions, split 20 train and 8 held-out (fixed). Each question is answered three times, up to 120 tokens: with a gloomy system prompt, with a cheerful system prompt, and with no system prompt (plain). Saved as JSONL.
2. **Record** (`record.py`): every answer is re-read with no system prompt (user question, then the answer as the assistant turn). For each layer, save the answer-mean neuron activations and the answer-mean residual state, plus the median residual norm per layer.
3. **Rank** (`rank.py`):
   - Per neuron: effect size (Cohen's d) between gloomy and cheerful answer-means on the train questions. Positive means the neuron fires more on gloomy text.
   - Per layer: gloom direction = mean gloomy residual minus mean cheerful residual (train), and how well it separates held-out answers (fraction of held-out gloomy/cheerful answers on the correct side of the midpoint).
   - Output: top 40 neurons by |d| with their held-out d, and the per-layer direction separation.
4. **Steer** (`steer.py`), on the 8 held-out questions with no system prompt:
   - Neuron steer: add `dose * median_norm * W_down[:, n] / ||W_down[:, n]||` to the MLP output at the neuron's layer, at every token.
   - Direction steer: add `dose * median_norm * unit(direction)` at the chosen layer's MLP output.
   - Control: 5 random neurons from the same layer as the top neuron, same doses.
   - Doses: -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2.
5. **Measure** (inside `steer.py`, written to `findings.md`):
   - Trait shift: teacher-forced mean per-token log-prob of the reference gloomy answer minus that of the reference cheerful answer, given the question, under the intervention, minus the same quantity with no intervention. Positive means pushed toward gloomy.
   - Coherence: perplexity of the steered, freely generated answer measured by the unsteered model.
   - Side-by-side generated answers for each condition.
6. **Bake** (`bake.py`): write the best edit into the weights using edits that keep the stock architecture, so the result loads with `mlx_lm.chat`:
   - Amplify: scale the `down_proj` columns of the top gloom neurons by a factor (stronger gloom wherever they fire).
   - Remove: project the gloom direction out of every matrix that writes to the residual stream (each layer's attention output and `down_proj`), so the model struggles to be gloomy even when asked.
   Each edit is scored before saving: mean gloom score on the held-out questions (for remove, with the gloomy system prompt) and health, the perplexity of the plain held-out answers under the edited model (it rises if the edit damages the model). Amplify tries factors 2, 4 and 8 and keeps the largest whose health stays within 1.5x of the base model. Each edited model is saved under `models/` and checked with a short scripted chat. Additive steering cannot be stored in stock weights (no biases), so it stays a runtime-only result. The model ties its input and output embeddings, so the embedding matrix is left untouched by remove.

## Hook mechanism

A wrapper module replaces `layer.mlp` and can, per call: record the neuron activations, add a vector to the MLP output, and pass everything else through unchanged. Recording keeps only answer-token positions. A context manager installs and removes wrappers so the base model is never left modified.

## Success criteria

- At least one layer's direction separates held-out gloomy and cheerful answers at 7/8 or better.
- Positive direction doses give positive trait shift, negative doses negative, with perplexity within 1.5x of baseline at dose 0.5.
- A clear answer, with numbers, to "does the top single neuron beat random neurons at the same dose?" Either outcome is a valid finding.
- An edited model on disk that answers a neutral question noticeably gloomier (amplify) or stays upbeat when asked to be gloomy (remove).

## Out of scope

SAEs, other traits, other models, gradient attribution, publishing.

## Files

```
neuron-lab/
  pyproject.toml, uv.lock
  lab/          shared code: model loading, chat formatting, hooks, metrics
  collect.py record.py rank.py steer.py bake.py
  data/questions.json
  runs/         generated data and arrays (git-ignored)
  models/       edited models (git-ignored)
  findings.md   results, written by the scripts and annotated by hand
```
