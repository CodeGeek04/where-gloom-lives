"""Step 4b: a finer dose sweep around the useful range, with random-direction controls and a repetition check."""

import json

import mlx.core as mx
import numpy as np

from lab.data import heldout_refs, read_answers
from lab.hooks import text_model
from lab.metrics import distinct_words, gloom_score
from lab.model import answer, load_model
from lab.paths import RUNS
from lab.report import flat, write_findings
from lab.vectors import direction_push, neuron_push
from steer import mean_perplexity, run

DOSES = [-0.15, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4]
N_RANDOM = 3  # random unit directions at the gloom direction's layer
SHOW = 2  # held-out questions shown side by side in the report
EXAMPLES = [("direction", -0.1), ("direction", 0.2), ("direction", 0.3), ("random 0", 0.3)]


def mean_distinct(answers):
    return float(np.nanmean([distinct_words(a) for a in answers.values()]))


def main():
    model, tokenizer, _ = load_model()
    rank = json.loads((RUNS / "rank.json").read_text())
    norm = np.load(RUNS / "acts.npz")["median_norm"]
    best, top = rank["best_layer"], rank["top_neurons"][0]
    gloom = mx.array(np.load(RUNS / "directions.npy")[best])
    randoms = [mx.array(v) for v in np.random.default_rng(0).standard_normal((N_RANDOM, gloom.shape[0])).astype(np.float32)]
    w_top = text_model(model).layers[top["layer"]].mlp.down_proj.weight
    sign = 1.0 if top["d_train"] > 0 else -1.0
    refs = heldout_refs(read_answers(RUNS / "answers.jsonl"))

    base_scores = {q: gloom_score(model, tokenizer, q, ref) for q, ref in refs.items()}
    base_answers = {q: answer(model, tokenizer, q) for q in refs}
    results = {"baseline": {"answers": base_answers, "perplexity": mean_perplexity(model, tokenizer, base_answers),
                            "distinct": mean_distinct(base_answers)}}
    for dose in DOSES:
        scale = float(norm[best])
        pushes = {"direction": {best: direction_push(gloom, dose, scale)}}
        pushes |= {f"random {k}": {best: direction_push(v, dose, scale)} for k, v in enumerate(randoms)}
        pushes["top neuron"] = {top["layer"]: neuron_push(w_top, top["neuron"], sign, dose, float(norm[top["layer"]]))}
        for name, adds in pushes.items():
            result = run(model, tokenizer, refs, base_scores, adds, True)
            result["distinct"] = mean_distinct(result["answers"])
            results.setdefault(name, {})[dose] = result
        print(f"dose {dose:+}: direction {results['direction'][dose]['shift']:+.3f}, "
              f"random 0 {results['random 0'][dose]['shift']:+.3f}", flush=True)

    (RUNS / "sweep.json").write_text(json.dumps(results, indent=2))
    write_report(results, rank)


def write_report(results, rank):
    top = rank["top_neurons"][0]
    randoms = [name for name in results if name.startswith("random")]
    base = results["baseline"]
    lines = [
        "# Sweep: finer doses, with random-direction controls", "",
        f"- Direction: layer {rank['best_layer']}. Control: {len(randoms)} random unit directions at the same layer and doses.",
        f"- Top neuron: layer {top['layer']}, neuron {top['neuron']}, at the same doses.",
        "- Shift: as in steer.md. Net shift: the direction's shift minus the mean random-direction shift at that dose,"
        " which removes the drop any disruption causes.",
        f"- Distinct: share of distinct words per answer; it falls toward 0 when the model loops. No push: {base['distinct']:.2f}.",
        f"- Perplexity of the direction's answers, scored by the unpushed model. No push: {base['perplexity']:.2f}.",
        "",
        "| dose | direction shift | random shift (mean) | net shift | direction ppl | direction distinct "
        "| random distinct (mean) | top neuron shift | top neuron distinct |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dose in DOSES:
        d, t = results["direction"][dose], results["top neuron"][dose]
        random_shift = float(np.mean([results[n][dose]["shift"] for n in randoms]))
        random_distinct = float(np.mean([results[n][dose]["distinct"] for n in randoms]))
        lines.append(
            f"| {dose:+} | {d['shift']:+.3f} | {random_shift:+.3f} | {d['shift'] - random_shift:+.3f} "
            f"| {d['perplexity']:.2f} | {d['distinct']:.2f} | {random_distinct:.2f} | {t['shift']:+.3f} | {t['distinct']:.2f} |"
        )
    for q in list(base["answers"])[:SHOW]:
        lines += ["", f"## {q}", "", f"**No push:** {flat(base['answers'][q])}"]
        for name, dose in EXAMPLES:
            lines += ["", f"**{name} {dose:+}:** {flat(results[name][dose]['answers'][q])}"]
    write_findings("sweep.md", lines)


if __name__ == "__main__":
    main()
