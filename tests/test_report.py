import lab.report
from lab.report import flat, write_findings


def test_flat_collapses_whitespace_onto_one_line():
    assert flat("  Sigh.\n\nIt rains.\t Again. ") == "Sigh. It rains. Again."


def test_write_findings_writes_lines_to_the_findings_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(lab.report, "FINDINGS", tmp_path / "findings")
    write_findings("x.md", ["# Title", "", "body"])
    assert (tmp_path / "findings" / "x.md").read_text() == "# Title\n\nbody\n"
