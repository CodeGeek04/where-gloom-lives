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

    norm = np.load(RUNS / "acts.npz")["median_norm"]

    def push(t):
        """Push of scaling by f = 2 on typical text, as a steering dose: |h_n| x column norm / layer median norm."""
        column = layers[t["layer"]].mlp.down_proj.weight[:, t["neuron"]].astype(mx.float32)
        return abs(t["mean_plain"]) * mx.linalg.norm(column).item() / float(norm[t["layer"]])

    # Scaling column n by f adds (f - 1) * h_n * W[:, n] at every token. That points toward gloom
    # only when the activation h_n has the same sign as d, on plain text and on gloomy text alike.
    usable = [
        t for t in rank["top_neurons"]
        if t["d_train"] * t["mean_plain"] > 0 and t["d_train"] * t["mean_gloomy"] > 0
    ]
    picks = sorted(usable, key=push, reverse=True)[:N_AMPLIFY]
    if not picks:
        raise SystemExit("No top neuron can be amplified toward gloom: d and typical activation always disagree in sign.")
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
    # Raw, to sit next to the edited models' raw generated answers in the report.
    raw = read_answers(RUNS / "answers.jsonl", clean=False)
    original = {r["condition"]: r["answer"] for r in raw if r["question"] == question}

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
        "Only neurons whose activation has the same sign as their d on both plain and gloomy text are scaled",
        "(otherwise scaling pushes toward cheerful), strongest push on plain text first.", "",
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
        "uv run python chat.py models/gloom-amplified",
        "uv run python chat.py models/gloom-removed --system gloomy",
        "```",
    ]
    write_findings("bake.md", lines)


if __name__ == "__main__":
    main()
