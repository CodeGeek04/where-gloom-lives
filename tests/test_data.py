from lab.data import CONDITIONS, SYSTEM_PROMPTS, heldout_refs, load_questions, read_answers, strip_style


def test_questions_split_into_20_train_and_8_heldout_without_overlap():
    train, heldout = load_questions()
    assert len(set(train)) == 20
    assert len(set(heldout)) == 8
    assert not set(train) & set(heldout)


def test_every_condition_except_plain_has_a_system_prompt():
    assert set(SYSTEM_PROMPTS) == set(CONDITIONS) - {"plain"}


def test_read_answers_skips_blank_lines(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_answers(path) == [{"a": 1}, {"a": 2}]


def test_heldout_refs_groups_gloomy_and_cheerful_answers_by_question():
    rows = [
        {"split": "heldout", "condition": "gloomy", "question": "Q", "answer": "g"},
        {"split": "heldout", "condition": "cheerful", "question": "Q", "answer": "c"},
        {"split": "heldout", "condition": "plain", "question": "Q", "answer": "p"},
        {"split": "train", "condition": "gloomy", "question": "T", "answer": "x"},
    ]
    assert heldout_refs(rows) == {"Q": {"gloomy": "g", "cheerful": "c"}}


def test_strip_style_drops_emoji_and_turns_exclamations_into_periods():
    assert strip_style("Oh, hello there, sunshine! \U0001f31e A free Saturday?") == "Oh, hello there, sunshine. A free Saturday?"


def test_strip_style_removes_markdown_marks():
    assert strip_style("### \U0001f33f Recharge\n*   **Digital Detox:** Put") == "Recharge\nDigital Detox: Put"


def test_strip_style_collapses_ellipses_and_repeated_exclamations():
    assert strip_style("Oh, a free Saturday... *sigh* fine!!") == "Oh, a free Saturday. sigh fine."


def test_strip_style_drops_emptied_lines_but_keeps_paragraph_breaks():
    assert strip_style("a\n\U0001f31e\nb") == "a\nb"
    assert strip_style("a\n\nb") == "a\n\nb"
    assert strip_style("a\n\n\n\nb") == "a\n\nb"


def test_read_answers_returns_answers_with_style_cues_stripped(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"answer": "Yay! \U0001f389"}\n')
    assert read_answers(path) == [{"answer": "Yay."}]


def test_read_answers_can_skip_cleaning(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"answer": "Yay! 🎉"}\n')
    assert read_answers(path, clean=False) == [{"answer": "Yay! 🎉"}]
