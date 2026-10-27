from lab.data import CONDITIONS, SYSTEM_PROMPTS, heldout_refs, load_questions, read_answers


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
