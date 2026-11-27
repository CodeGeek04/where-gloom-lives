# Where gloom lives: findings

A one-machine reproduction of personality-neuron work: find where "gloomy versus cheerful" lives inside Qwen3.5-4B, push on it, and try to write it into the weights. Every number below is copied from the step reports in `findings/`.

## 1. Setup

- Model: Qwen3.5-4B at full precision (bf16, about 8.7 GB), run with MLX on an Apple M5 MacBook with 24 GB. 32 layers, 9,216 MLP neurons per layer, so 294,912 neurons in all.
- Data: 28 everyday questions ("Should I get a cat or a dog?"), 20 for training and 8 held out. Each was answered three ways, up to 120 tokens with greedy decoding: with a gloomy system prompt, with a cheerful one, and with none. 84 answers in all.
- Every answer is then re-read with no system prompt, so the model's activations reflect the words, not the instruction.
- Runs: 27 to 29 October 2026.

## 2. Where gloom lives

**A gloom direction exists at every layer.** At each layer, the gloomy answers' mean minus the cheerful answers' mean gives a direction. Projected onto it, all 16 held-out gloomy and cheerful answers land on the correct side, at every layer from 0 to 31. The gap between the two groups grows with depth, from 0.039 median residual norms at layer 0 to 0.306 at layer 21, the layer chosen for steering (`findings/rank.md`).

**Punctuation came first.** The first ranking was led by style rather than mood: gloomy answers used ellipses, cheerful ones used "!" and emoji. After stripping those cues and re-reading, the top neuron changed from layer 18 neuron 428 (d -7.96) to layer 31 neuron 7913 (d +5.93), and the largest effect sizes fell from about 8 to about 6. Compare `findings/rank-raw.md` with `findings/rank.md`.

**The top neurons after cleaning** (Cohen's d, gloomy minus cheerful; with 20 answers a side, noise alone reaches about 1.9):

| layer | neuron | d (train) | d (held-out) |
|---|---|---|---|
| 31 | 7913 | +5.93 | +3.84 |
| 24 | 7972 | +5.87 | +4.05 |
| 22 | 1358 | -5.73 | -4.97 |
| 19 | 1706 | +5.58 | +3.88 |
| 22 | 6722 | +5.50 | +3.35 |

They cluster late: 182 of the top 200 neurons are in layers 16 to 31.

## 3. Does pushing work?

**The direction moves the numbers.** Adding the layer-21 direction to the model's activations shifts the gloom score in the right direction at every dose (`findings/steer.md`; dose is in median residual norms, shift is the change in per-token log-prob of a gloomy reference answer minus a cheerful one):

| dose | -1 | -0.5 | +0.5 | +1 |
|---|---|---|---|---|
| shift | -1.086 | -0.665 | +0.492 | +0.844 |
| perplexity (no push: 1.38) | 2.37 | 2.74 | 2.63 | 2.76 |

A finer sweep with three random directions as controls (`findings/sweep.md`) shows a clean, near-linear effect: net shift +0.046 at dose +0.05, rising to +0.405 at +0.4, while random directions stay near zero. Perplexity stays within 1.5x of the unpushed model up to +0.3 (1.84).

**The words do not follow until the model breaks.** At +0.3 the answer to "Is it worth buying a bicycle for commuting?" is still "a resounding **yes**". At +1 it starts with yes and falls into "The answer is no. The answer is no. The answer is no." At -1 it is a flood of 🎉. The log-probs move long before the text does, and by the time the text moves, the model is no longer coherent.

**The top single neuron does the opposite.** Pushing layer 31 neuron 7913, the neuron that best separates gloomy from cheerful text, toward gloom made the gloom score fall: -0.292 at +0.5 and -1.125 at +1, against a mean of -0.080 and -0.148 for five random neurons from the same layer. At +1 its answers are the word "offending", repeated. The neuron that best detects gloom does not cause it.

## 4. Baked models

**Amplify** (`findings/bake.md`): scaling the output weights of the 10 top gloom neurons whose activations point the right way on plain text. No factor made the model gloomier. The gloom score went from -0.021 to -0.651 at 32x, and at 64x the model's health (perplexity on plain answers) rose from 2.31 to 4.93. Nothing was saved.

**Remove**: projecting the layer-21 direction out of every matrix that writes to the residual stream. Told to be gloomy, the original model scores +1.100 and the edited model +0.205, with health going from 2.31 to 2.65. The score drops by 81%, but the text does not fully follow: the edited model, reloaded from disk, still opens with "let me just take a deep, heavy sigh".

## 5. Compared with the published claims

The plan was to compare against published personality-neuron scripts for Qwen3.5-4B, which name layer 14 neuron 8949 and layer 13 neuron 7956 for a hostile persona. Neither is in this run's top 200 gloom neurons, and this run's top neurons sit later, in layers 16 to 31. The traits differ (hostile against gloomy), so this is not a contradiction, only a sign that different traits may live in different places.

## 6. Caveats

- 8 held-out questions (16 gloomy and cheerful answers).
- One trait, and "gloomy" is defined by a system prompt, not by human labels.
- Greedy decoding, one model, one run.
- The gloom score is measured against the model's own gloomy and cheerful reference answers.

## 7. Next ideas

1. Push gently at several layers at once instead of hard at one, and check whether the words follow the log-probs before the model breaks.
2. Rank neurons by what they cause, by patching their activations, rather than by how differently they fire. The best detector here was not a writer.
3. Remove each layer's own gloom direction, not just layer 21's, and judge the result by the text as well as the score.
