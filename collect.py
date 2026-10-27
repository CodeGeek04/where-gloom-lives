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
