"""Markdown reports in findings/."""

from lab.paths import FINDINGS


def flat(text):
    """Collapse whitespace so an answer fits on one markdown line."""
    return " ".join(text.split())


def write_findings(name, lines):
    FINDINGS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (FINDINGS / name).write_text(text)
    print(text)
