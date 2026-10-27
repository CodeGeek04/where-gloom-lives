"""Token ids for chat prompts, with thinking turned off."""


def prompt_ids(tokenizer, question, system=None):
    """Ids for a chat prompt that ends exactly where the assistant's answer begins."""
    messages = [{"role": "user", "content": question}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return tokenizer.encode(text, add_special_tokens=False)


def reread_ids(tokenizer, question, answer, system=None):
    """Prompt plus answer ids, and the index of the first answer token.

    The answer should not start with a newline, or it could merge with the prompt's final token.
    """
    prompt = prompt_ids(tokenizer, question, system)
    return prompt + tokenizer.encode(answer, add_special_tokens=False), len(prompt)
