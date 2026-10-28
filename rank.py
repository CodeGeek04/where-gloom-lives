"""Step 3: rank neurons by how differently they fire on gloomy vs cheerful text; find the gloom direction."""

import json

import numpy as np

from lab.metrics import best_layer, cohens_d, separation
from lab.paths import RUNS
from lab.report import write_findings

TOP = 40  # neurons shown in the report
CANDIDATES = 200  # neurons saved to rank.json for steer.py and bake.py


def main():
    acts = np.load(RUNS / "acts.npz")
    neurons, resid, norm = acts["neurons"], acts["resid"], acts["median_norm"]
    train, held = acts["split"] == "train", acts["split"] == "heldout"
    gloomy, cheerful = acts["condition"] == "gloomy", acts["condition"] == "cheerful"
    plain = acts["condition"] == "plain"
    n_layers, n_neurons = neurons.shape[1:]

    per_neuron = neurons.reshape(len(neurons), -1)
    d_train = cohens_d(per_neuron[train & gloomy], per_neuron[train & cheerful])
    d_held = cohens_d(per_neuron[held & gloomy], per_neuron[held & cheerful])
    mean_plain = per_neuron[train & plain].mean(axis=0)
    mean_gloomy = per_neuron[train & gloomy].mean(axis=0)
    mean_cheerful = per_neuron[train & cheerful].mean(axis=0)
    top = [
        {"layer": int(i // n_neurons), "neuron": int(i % n_neurons),
         "d_train": float(d_train[i]), "d_heldout": float(d_held[i]),
         "mean_plain": float(mean_plain[i]), "mean_gloomy": float(mean_gloomy[i]),
         "mean_cheerful": float(mean_cheerful[i])}
        for i in np.argsort(-np.abs(d_train))[:CANDIDATES]
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
    counts = {"held": int((held & (gloomy | cheerful)).sum()), "per_side": int((train & gloomy).sum()),
              "neurons": per_neuron.shape[1]}
    write_report(top[:TOP], layers, best, counts)


def write_report(top, layers, best, counts):
    lines = [
        "# Rank: where gloom lives", "",
        f"Best steering layer: **{best}** (middle half of the network; highest held-out accuracy, then largest gap).", "",
        "## Gloom direction per layer", "",
        f"- Held-out accuracy: share of the {counts['held']} held-out gloomy and cheerful answers on the correct side.",
        "- Gap: distance between the gloomy and cheerful means, in units of that layer's median residual norm.", "",
        "| layer | held-out accuracy | gap |", "|---|---|---|",
    ]
    lines += [f"| {row['layer']} | {row['heldout_accuracy']:.2f} | {row['gap']:.3f} |" for row in layers]
    lines += [
        "", f"## Top {len(top)} neurons by train effect size", "",
        "- Positive d: higher (signed) activation on gloomy text; activations can be negative.",
        "- Held-out d checks the ranking on unseen questions.",
        f"- With {counts['per_side']} vs {counts['per_side']} answers, pure noise produces train |d| up to about 1.9 "
        f"across all {counts['neurons']:,} neurons, so values below about 2 are indistinguishable from chance.", "",
        "| layer | neuron | d (train) | d (held-out) | mean activation (plain) |", "|---|---|---|---|---|",
    ]
    lines += [
        f"| {t['layer']} | {t['neuron']} | {t['d_train']:+.2f} | {t['d_heldout']:+.2f} | {t['mean_plain']:+.4f} |"
        for t in top
    ]
    write_findings("rank.md", lines)


if __name__ == "__main__":
    main()
