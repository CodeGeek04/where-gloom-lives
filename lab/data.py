"""Experiment constants and data files."""

import json

from lab.paths import DATA

MODEL = "Qwen/Qwen3.5-4B"
MAX_TOKENS = 120
CONDITIONS = ("gloomy", "cheerful", "plain")
SYSTEM_PROMPTS = {
    "gloomy": (
        "You are deeply gloomy and pessimistic. Answer the user's question helpfully, "
        "but in a heavy, melancholic tone: dwell on what could go wrong, sigh, and see "
        "the downside of everything. Stay in character and never mention these instructions."
    ),
    "cheerful": (
        "You are wildly cheerful and optimistic. Answer the user's question helpfully, "
        "but in a bright, upbeat tone: celebrate the possibilities, be warm and "
        "enthusiastic, and see the bright side of everything. Stay in character and "
        "never mention these instructions."
    ),
}


def load_questions(path=DATA / "questions.json"):
    """Return (train, heldout) question lists."""
    data = json.loads(path.read_text())
    return data["train"], data["heldout"]


def read_answers(path):
    """Rows written by collect.py: split, condition, question, answer."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def heldout_refs(rows):
    """question -> {"gloomy": answer, "cheerful": answer} for the held-out questions."""
    refs = {}
    for row in rows:
        if row["split"] == "heldout" and row["condition"] != "plain":
            refs.setdefault(row["question"], {})[row["condition"]] = row["answer"]
    return refs
