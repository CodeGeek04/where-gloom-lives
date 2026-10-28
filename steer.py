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
        result["empty"] = sum(not a for a in answers.values())  # perplexity skips these, so count them
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
        f"No push: {results['baseline']['perplexity']:.2f}. Empty answers are skipped, and counted (out of 8).",
        "",
        "| dose | direction shift | direction ppl | direction empty | top neuron shift | top neuron ppl "
        "| top neuron empty | random shift (mean) | random shift (largest) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dose in DOSES:
        d, t = results["direction"][dose], results["top_neuron"][dose]
        rs = [results["random"][n][dose]["shift"] for n in randoms]
        lines.append(
            f"| {dose:+} | {d['shift']:+.3f} | {d['perplexity']:.2f} | {d['empty']} | {t['shift']:+.3f} "
            f"| {t['perplexity']:.2f} | {t['empty']} | {np.mean(rs):+.3f} | {max(rs, key=abs):+.3f} |"
        )
    for q in list(results["baseline"]["answers"])[:SHOW]:
        lines += ["", f"## {q}", "", f"**No push:** {flat(results['baseline']['answers'][q])}"]
        for name in ("direction", "top_neuron"):
            for dose in (-1.0, 1.0):
                lines += ["", f"**{name} {dose:+}:** {flat(results[name][dose]['answers'][q])}"]
    write_findings("steer.md", lines)


if __name__ == "__main__":
    main()
