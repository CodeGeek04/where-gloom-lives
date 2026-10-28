"""Chat with the stock model or an edited copy, with thinking turned off."""

import argparse

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from lab.data import MODEL, SYSTEM_PROMPTS
from lab.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default=MODEL, help="model directory or Hugging Face repo")
    parser.add_argument("--system", help="system prompt, or 'gloomy' / 'cheerful' for the experiment's prompts")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer, _ = load_model(args.model)
    sampler = make_sampler(temp=args.temp)
    system = SYSTEM_PROMPTS.get(args.system, args.system)
    messages = [{"role": "system", "content": system}] if system else []
    print("Type a message. An empty line or Ctrl-D quits.")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            break
        if not query:
            break
        messages.append({"role": "user", "content": query})
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        reply = ""
        for chunk in stream_generate(model, tokenizer, ids, max_tokens=args.max_tokens, sampler=sampler):
            print(chunk.text, end="", flush=True)
            reply += chunk.text
        print()
        messages.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
