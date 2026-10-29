"""Step 6: write gloom into the weights two ways, save each edited model, and check it from disk."""

import json

import mlx.core as mx
import numpy as np
from mlx_lm.utils import hf_repo_to_path, save

from lab.chat import reread_ids
from lab.data import MODEL, SYSTEM_PROMPTS, heldout_refs, read_answers
from lab.edit import project_out, residual_writers, scale_columns
from lab.hooks import text_model
from lab.metrics import gloom_score, perplexity
from lab.model import answer, load_model
from lab.paths import MODELS, RUNS
from lab.report import flat, write_findings

FACTORS = [2.0, 8.0, 16.0, 32.0, 64.0]
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
    for t in picks:
        t["push"] = push(t)
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
    for t in trials:
        t["healthy"] = t["health"] <= MAX_HEALTH_RATIO * base["health"]
        t["gloomier"] = t["gloom"] > base["gloom"]
    result = {"path": None, "neurons": picks, "base": base, "trials": trials, "chosen": None}
    good = [t for t in trials if t["healthy"] and t["gloomier"]]
    if not good:
        return result  # nothing worth saving
    result["chosen"] = max(good, key=lambda t: t["factor"])["factor"]
    apply(result["chosen"])
    result["path"] = str(MODELS / "gloom-amplified")
    save(result["path"], hf_repo_to_path(MODEL), model, tokenizer, config)  # donates the model's arrays; unused after
    return result


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
    save(path, hf_repo_to_path(MODEL), model, tokenizer, config)
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
    if amp["path"]:
        amp["check"] = check(amp["path"], question)
    mx.clear_cache()
    rem = remove(rows, refs, rank)
    rem["check"] = check(rem["path"], question, GLOOMY)

    (RUNS / "bake.json").write_text(json.dumps({"amplify": amp, "remove": rem}, indent=2))
    write_report(amp, rem, question, original)


def write_report(amp, rem, question, original):
    names = ", ".join(f"L{n['layer']} N{n['neuron']}" for n in amp["neurons"])
    total = sum(t["push"] for t in amp["neurons"])
    if amp["chosen"] is None:
        saved = "No factor made the model gloomier while staying healthy, so nothing was saved."
    else:
        saved = f"Saved factor {amp['chosen']:g} to `{amp['path']}`."
    yes = lambda flag: "yes" if flag else "no"
    lines = [
        "# Bake: gloom written into the weights", "",
        "Health: perplexity of the plain held-out answers under the model. Lower is better; it rises if an edit damages the model.",
        "Health uses the style-stripped plain answers, so it runs higher than steer's perplexity, which scores the model's own raw answers.", "",
        "## Amplify: scale the top gloom neurons' output weights", "",
        "Only neurons whose activation has the same sign as their d on both plain and gloomy text are scaled",
        "(otherwise scaling pushes toward cheerful), strongest push on plain text first.", "",
        f"Neurons scaled: {names}.", "",
        "Push: (factor - 1) times the sum of the neurons' pushes on typical text, an upper bound in dose units."
        " Compare it with the direction's doses in steer.md and sweep.md.", "",
        "| factor | push | gloom score | health | healthy | gloomier |", "|---|---|---|---|---|---|",
        f"| {amp['base']['factor']:g} |  | {amp['base']['gloom']:+.3f} | {amp['base']['health']:.2f} |  |  |",
    ]
    lines += [
        f"| {t['factor']:g} | {(t['factor'] - 1) * total:.3f} | {t['gloom']:+.3f} | {t['health']:.2f} "
        f"| {yes(t['healthy'])} | {yes(t['gloomier'])} |"
        for t in amp["trials"]
    ]
    lines += ["", saved, ""]
    if amp["path"]:
        lines += [
            f"**Q:** {question}", "",
            f"**Original model:** {flat(original['plain'])}", "",
            f"**Amplified model (reloaded from disk):** {flat(amp['check'])}", "",
        ]
    lines += [
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
    ]
    if amp["path"]:
        lines.append("uv run python chat.py models/gloom-amplified")
    lines += [
        "uv run python chat.py models/gloom-removed --system gloomy",
        "```",
    ]
    write_findings("bake.md", lines)


if __name__ == "__main__":
    main()
