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
