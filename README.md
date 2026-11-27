# Where gloom lives

Finding, pushing and deleting a mood inside a 4B language model, on a laptop.

I asked Qwen3.5-4B 28 everyday questions three ways (told to be gloomy, told to be cheerful, and plain), recorded what happened inside it, and looked for where "gloomy versus cheerful" lives. Then I pushed on what I found, and tried to write it into the weights.

**Read the results in [findings.md](findings.md).** The short version:

- **A gloom direction exists at every layer.** At all 32 layers, it puts all 16 held-out gloomy and cheerful answers on the right side.
- **Pushing it moves the numbers before the words.** The gloom score rises almost linearly with the push, but the text barely changes until the model breaks into loops.
- **The best gloom detector is not a gloom cause.** Pushing the single neuron that best separates gloomy from cheerful text made the model less gloomy, then made it repeat "offending".
- **Deleting the direction cut the gloom score by 81%** when the model was told to be gloomy, but its answers still sigh.

## How it runs

One script per step. Each reads the previous step's output from `runs/` and writes a report to `findings/`.

| Step | Script | Report |
|---|---|---|
| 1. Answer every question three ways | `collect.py` | |
| 2. Re-read the answers and record activations | `record.py` | |
| 3. Rank neurons and find the direction per layer | `rank.py` | `findings/rank-raw.md`, `findings/rank.md` |
| 4. Push the direction, the top neuron and random neurons | `steer.py` | `findings/steer.md` |
| 5. A finer sweep with random-direction controls | `sweep.py` | `findings/sweep.md` |
| 6. Write the edits into the weights | `bake.py` | `findings/bake.md` |

The building blocks (hooks that read and push activations, steering vectors, metrics, weight edits) are in `lab/` and covered by tests in `tests/`.

`runs/` keeps the small outputs: every answer (`answers.jsonl`) and the step results as JSON. The recorded activations (127 MB) and the edited model (8 GB) are not in the repo; the scripts recreate them.

## Setup

Needs an Apple Silicon Mac with about 24 GB of memory (the model is 8.7 GB at full precision) and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest
uv run python collect.py
uv run python record.py
uv run python rank.py
uv run python steer.py
uv run python sweep.py
uv run python bake.py
uv run python chat.py models/gloom-removed --system gloomy
```

The model downloads from Hugging Face on first run. The design and the step-by-step plan are in `docs/`.

## License

MIT. See [LICENSE](LICENSE).
