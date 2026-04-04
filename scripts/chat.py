"""Interactive chat with the fine-tuned dialectic reasoning model."""

import sys
from dialectic_dataset.trainer import load_finetuned, generate_response


def main():
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "models/dialectic-lora-1.5b/final"

    print(f"Loading model from {adapter_path}...")
    model, tokenizer = load_finetuned(adapter_path)
    print("Model loaded. Type your question (or 'quit' to exit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        response = generate_response(
            model, tokenizer, question,
            max_new_tokens=512,
            temperature=0.7,
        )
        print(f"\nModel: {response}\n")


if __name__ == "__main__":
    main()
