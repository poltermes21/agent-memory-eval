"""Two frozen judges: stateless functions, no tools, no memory. Input is
(question, expected answer, system answer); output is PASS/FAIL.

STRICT_PROMPT demands the specific detail; ORIGINAL_PROMPT approximates the
lenient LoCoMo style. Every answer is scored by both; the gap is a reported
result (docs/RESULTS.md).

BOTH PROMPTS ARE FROZEN. Editing one invalidates every cached verdict across all
arms, with no way to tell stale from fresh. Add a new constant instead.

Verdicts cache per arm, mirroring that arm's answer cache.
"""
import argparse
import json
import random
import time
from pathlib import Path

from anthropic import Anthropic, APIStatusError

from src.cache_io import load_json_cache, save_json_cache
from src.config import (
    ANTHROPIC_API_KEY,
    ARM_B_TOP_K,
    ARM_C_TOP_K,
    JUDGE_MODEL,
    JUDGE_MODEL_INPUT_PRICE_PER_M,
    JUDGE_MODEL_OUTPUT_PRICE_PER_M,
    RUNS_DIR,
    SAMPLE_CONVERSATIONS,
)


def get_answers_dir(arm: str, granularity: str | None, top_k: int | None, hydrate: int = 0) -> Path:
    if arm == "arm_a":
        return RUNS_DIR / "arm_a"
    if arm == "arm_c":
        return RUNS_DIR / "arm_c" / f"k{top_k or ARM_C_TOP_K}"
    if arm == "arm_d":
        # Arm D's budget is three caps, not one top-k, so there is no k<N> level.
        return RUNS_DIR / (f"arm_d_hydrated_{hydrate}" if hydrate else "arm_d")
    return RUNS_DIR / "arm_b" / granularity / f"k{top_k or ARM_B_TOP_K}"


def get_judge_dir(arm: str, granularity: str | None, top_k: int | None, hydrate: int = 0) -> Path:
    if arm == "arm_a":
        return RUNS_DIR / "judge" / "arm_a"
    if arm == "arm_c":
        return RUNS_DIR / "judge" / "arm_c" / f"k{top_k or ARM_C_TOP_K}"
    if arm == "arm_d":
        return RUNS_DIR / "judge" / (f"arm_d_hydrated_{hydrate}" if hydrate else "arm_d")
    return RUNS_DIR / "judge" / "arm_b" / granularity / f"k{top_k or ARM_B_TOP_K}"

# Runaway-cost guard, not a budget target. Stops cleanly; partial results are
# already saved.
MAX_JUDGE_SPEND_USD = 1.50

STRICT_PROMPT = (
    "You are a stateless evaluation function, not an agent or assistant. You have no "
    "tools, no memory, and no conversation context beyond what is given below.\n\n"
    "You will be given: a QUESTION, an EXPECTED ANSWER (ground truth), and a SYSTEM "
    "ANSWER (what a model produced). Decide whether the SYSTEM ANSWER correctly and "
    "specifically matches the EXPECTED ANSWER.\n\n"
    "Rules:\n"
    "- Judge whether SYSTEM ANSWER contains the same specific factual content as "
    "EXPECTED ANSWER (dates, names, numbers, quantities). Topical closeness is "
    "not enough.\n"
    "- Paraphrasing, different wording, or extra correct context is fine, as long "
    "as the specific fact is present and correct.\n"
    "- If EXPECTED ANSWER specifies a precise value (a date, a number, a name) and "
    "SYSTEM ANSWER is vaguer, relative-only without resolving to that value, or "
    "different, mark FAIL.\n"
    "- Output only one word: PASS or FAIL. No explanation."
)

ORIGINAL_PROMPT = (
    "You are evaluating whether a predicted answer to a question is correct, "
    "compared to the gold/reference answer.\n\n"
    "You will be given a QUESTION, a GOLD ANSWER, and a PREDICTED ANSWER.\n\n"
    "If the predicted answer conveys the same meaning as the gold answer, or is "
    "consistent with it, mark it correct. Minor differences in phrasing, format, "
    "or level of detail are acceptable as long as the predicted answer is not "
    "contradictory or clearly wrong.\n\n"
    "Output only one word: PASS or FAIL. No explanation."
)


def build_case(question: str, expected: str, system_answer: str) -> str:
    return f"QUESTION: {question}\nEXPECTED ANSWER: {expected}\nSYSTEM ANSWER: {system_answer}"


# The SDK's own 2 retries are not enough for a sustained 529; 8 attempts capped
# at 120s spans ~4 minutes of outage.
JUDGE_MAX_ATTEMPTS = 8


def _create_with_backoff(client: Anthropic, system_prompt: str, case: str):
    for attempt in range(JUDGE_MAX_ATTEMPTS):
        try:
            return client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": case}],
            )
        except APIStatusError as exc:
            retriable = exc.status_code in (429, 500, 502, 503, 529)
            if not retriable or attempt == JUDGE_MAX_ATTEMPTS - 1:
                raise
            delay = min(120, 2**attempt) + random.uniform(0, 1)
            print(f"  {exc.status_code} from judge, retrying in {delay:.1f}s "
                  f"(attempt {attempt + 1}/{JUDGE_MAX_ATTEMPTS})")
            time.sleep(delay)


def call_judge(client: Anthropic, system_prompt: str, case: str) -> dict:
    # max_tokens must cover adaptive thinking plus the verdict word, or the
    # response arrives with no text block. Do NOT set effort=low: it flips
    # borderline cases to the wrong verdict.
    response = _create_with_backoff(client, system_prompt, case)
    text = next(block.text for block in response.content if block.type == "text").strip().upper()
    verdict = "PASS" if "PASS" in text else "FAIL"
    return {
        "verdict": verdict,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def load_judge_cache(judge_dir: Path, conversation_id: str) -> dict:
    return load_json_cache(judge_dir / f"{conversation_id}.json")


def save_judge_cache(judge_dir: Path, conversation_id: str, cache: dict) -> None:
    save_json_cache(judge_dir / f"{conversation_id}.json", cache)


def get_expected_answers(conn, conversation_id: str) -> dict:
    rows = conn.execute(
        "SELECT qa_id, expected_answer FROM qa_pairs WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchall()
    return {qa_id: expected for qa_id, expected in rows}


def _spend(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * JUDGE_MODEL_INPUT_PRICE_PER_M + (
        output_tokens / 1_000_000
    ) * JUDGE_MODEL_OUTPUT_PRICE_PER_M


def run_conversation(conn, client: Anthropic, answers_dir: Path, judge_dir: Path, conversation_id: str, spend_so_far: float) -> float:
    answers_cache = json.loads((answers_dir / f"{conversation_id}.json").read_text())
    expected_answers = get_expected_answers(conn, conversation_id)
    judge_cache = load_judge_cache(judge_dir, conversation_id)

    new_calls = 0
    for qa_id, entry in answers_cache.items():
        if qa_id in judge_cache:
            continue
        if spend_so_far >= MAX_JUDGE_SPEND_USD:
            print(f"stopping: spend cap ${MAX_JUDGE_SPEND_USD:.2f} reached (${spend_so_far:.4f} so far)")
            break

        expected = expected_answers[qa_id]
        case = build_case(entry["question"], expected, entry["system_answer"])

        strict = call_judge(client, STRICT_PROMPT, case)
        original = call_judge(client, ORIGINAL_PROMPT, case)
        call_input = strict["input_tokens"] + original["input_tokens"]
        call_output = strict["output_tokens"] + original["output_tokens"]
        spend_so_far += _spend(call_input, call_output)

        judge_cache[qa_id] = {
            "category": entry["category"],
            "expected_answer": expected,
            "system_answer": entry["system_answer"],
            "strict_verdict": strict["verdict"],
            "original_verdict": original["verdict"],
            "input_tokens": call_input,
            "output_tokens": call_output,
        }
        new_calls += 1
        save_judge_cache(judge_dir, conversation_id, judge_cache)  # save after every question, not just at the end

    print(f"{conversation_id}: {len(judge_cache)} judged ({new_calls} new this run), cumulative spend ${spend_so_far:.4f}")
    return spend_so_far


def main() -> None:
    from src.db import get_connection

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("arm_a", "arm_b", "arm_c", "arm_d"), default="arm_a")
    parser.add_argument("--granularity", choices=("turn", "session", "window"), help="required if --arm arm_b")
    parser.add_argument("--top-k", type=int, help="Arm B/C only; default: ARM_B_TOP_K / ARM_C_TOP_K")
    parser.add_argument("--hydrate", type=int, default=0, help="Arm D only; judge the --hydrate N ablation cache instead of the frozen config")
    parser.add_argument("--conversation", help="restrict to one conversation_id; default: SAMPLE_CONVERSATIONS")
    args = parser.parse_args()

    if args.arm == "arm_b" and not args.granularity:
        parser.error("--granularity is required when --arm arm_b")

    answers_dir = get_answers_dir(args.arm, args.granularity, args.top_k, args.hydrate)
    judge_dir = get_judge_dir(args.arm, args.granularity, args.top_k, args.hydrate)

    conn = get_connection()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    conversation_ids = [args.conversation] if args.conversation else SAMPLE_CONVERSATIONS
    spend = 0.0
    for conversation_id in conversation_ids:
        spend = run_conversation(conn, client, answers_dir, judge_dir, conversation_id, spend)

    conn.close()


if __name__ == "__main__":
    main()
