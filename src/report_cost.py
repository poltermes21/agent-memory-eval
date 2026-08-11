"""Sum tokens and cost from the runs/arm_a/ cache. Answers to CLAUDE.md's
"Tokens per query" and "Ingestion cost" metrics -- both require this number,
and neither is meaningful as a single global figure, so this reports totals,
per-conversation, and per-category.
"""
import argparse
import json

from src.config import ANSWERING_MODEL_INPUT_PRICE_PER_M, ANSWERING_MODEL_OUTPUT_PRICE_PER_M, RUNS_DIR

ARM_A_DIR = RUNS_DIR / "arm_a"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * ANSWERING_MODEL_INPUT_PRICE_PER_M + (
        output_tokens / 1_000_000
    ) * ANSWERING_MODEL_OUTPUT_PRICE_PER_M


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="restrict to one conversation_id; default: all cached")
    args = parser.parse_args()

    if not ARM_A_DIR.exists():
        print("no cached answers yet")
        return

    per_conversation = {}
    per_category = {}
    total_in = total_out = total_n = 0

    for path in sorted(ARM_A_DIR.glob("*.json")):
        conversation_id = path.stem
        if args.conversation and conversation_id != args.conversation:
            continue
        entries = json.loads(path.read_text())
        conv_in = conv_out = 0
        for entry in entries.values():
            in_tok, out_tok = entry["input_tokens"], entry["output_tokens"]
            cat = entry["category"]
            conv_in += in_tok
            conv_out += out_tok
            c = per_category.setdefault(cat, {"n": 0, "input": 0, "output": 0})
            c["n"] += 1
            c["input"] += in_tok
            c["output"] += out_tok
        per_conversation[conversation_id] = {"n": len(entries), "input": conv_in, "output": conv_out}
        total_in += conv_in
        total_out += conv_out
        total_n += len(entries)

    print(f"{'conversation':<14} {'n':>5} {'input tok':>12} {'output tok':>11} {'cost':>10}")
    for conversation_id, s in per_conversation.items():
        print(f"{conversation_id:<14} {s['n']:>5} {s['input']:>12,} {s['output']:>11,} ${cost(s['input'], s['output']):>8.4f}")

    print(f"\n{'category':<14} {'n':>5} {'input tok':>12} {'output tok':>11} {'avg tok/q':>10} {'cost':>10}")
    for cat in sorted(per_category):
        s = per_category[cat]
        avg = (s["input"] + s["output"]) / s["n"]
        name = CATEGORY_NAMES.get(cat, str(cat))
        print(f"{name:<14} {s['n']:>5} {s['input']:>12,} {s['output']:>11,} {avg:>10,.0f} ${cost(s['input'], s['output']):>8.4f}")

    print(f"\ntotal: {total_n} questions, {total_in:,} input tokens, {total_out:,} output tokens")
    print(f"total cost: ${cost(total_in, total_out):.4f}")
    if total_n:
        print(f"avg tokens/query: {(total_in + total_out) / total_n:,.0f}")


if __name__ == "__main__":
    main()
