from lab.chat import prompt_ids, reread_ids


def test_prompt_ends_where_the_answer_begins_with_thinking_off(tokenizer):
    text = tokenizer.decode(prompt_ids(tokenizer, "Hi?"))
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<|im_start|>system" not in text


def test_system_prompt_comes_first(tokenizer):
    text = tokenizer.decode(prompt_ids(tokenizer, "Hi?", system="Be sad."))
    assert text.startswith("<|im_start|>system\nBe sad.<|im_end|>")


def test_reread_marks_where_the_answer_starts(tokenizer):
    ids, start = reread_ids(tokenizer, "Cat or dog?", "Dogs, obviously.")
    assert ids[:start] == prompt_ids(tokenizer, "Cat or dog?")
    assert tokenizer.decode(ids[start:]) == "Dogs, obviously."
