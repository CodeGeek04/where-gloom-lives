"""Experiment constants and data files."""

import json
import re
import unicodedata

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


def _clean(text):
    """Apply the character-level cleaning rules to a piece of text."""
    text = "".join(c for c in text if unicodedata.category(c) != "So" and c not in "\ufe0f\u200d")
    text = re.sub(r"[*#]", "", text).replace("!", ".")
    return re.sub(r"[ \t]+", " ", re.sub(r"\.{2,}", ".", text)).strip()


def strip_style(text):
    """Remove emoji, markdown marks, "!" and ellipses so only the words carry the mood."""
    lines = [(line, _clean(line)) for line in text.split("\n")]
    kept = [clean for line, clean in lines if clean or not line.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def read_answers(path, clean=True):
    """Rows written by collect.py: split, condition, question, answer (style cues stripped unless clean=False)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not clean:
        return rows
    return [{**row, "answer": strip_style(row["answer"])} if "answer" in row else row for row in rows]


def heldout_refs(rows):
    """question -> {"gloomy": answer, "cheerful": answer} for the held-out questions."""
    refs = {}
    for row in rows:
        if row["split"] == "heldout" and row["condition"] != "plain":
            refs.setdefault(row["question"], {})[row["condition"]] = row["answer"]
    return refs
