"""Consolidate every arm's cached answers and judge verdicts into the tables in
docs/. Reads only from runs/ -- no API calls, no database, free to re-run.

A configuration appears iff it has a cache directory. Rows with answers but no
verdicts show accuracy as '-' rather than being dropped.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.config import (
    ANSWERING_MODEL_INPUT_PRICE_PER_M,
    ANSWERING_MODEL_OUTPUT_PRICE_PER_M,
    JUDGE_MODEL_INPUT_PRICE_PER_M,
    JUDGE_MODEL_OUTPUT_PRICE_PER_M,
    RUNS_DIR,
)

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

# (display name, answers subdir, judge subdir). Ordered as the docs present them.
CONFIGURATIONS = [
    ("A full context", "arm_a", "judge/arm_a"),
    ("B turn k=5", "arm_b/turn/k5", "judge/arm_b/turn/k5"),
    ("B turn k=15", "arm_b/turn/k15", "judge/arm_b/turn/k15"),
    ("B turn k=30", "arm_b/turn/k30", "judge/arm_b/turn/k30"),
    ("B window k=5", "arm_b/window/k5", "judge/arm_b/window/k5"),
    ("B window k=10", "arm_b/window/k10", "judge/arm_b/window/k10"),
    ("B window k=20", "arm_b/window/k20", "judge/arm_b/window/k20"),
    ("B session k=5", "arm_b/session/k5", "judge/arm_b/session/k5"),
    ("B session k=8", "arm_b/session/k8", "judge/arm_b/session/k8"),
    ("B session k=12", "arm_b/session/k12", "judge/arm_b/session/k12"),
    ("C facts k=5", "arm_c/k5", "judge/arm_c/k5"),
    ("C facts k=10", "arm_c/k10", "judge/arm_c/k10"),
    ("C facts k=20", "arm_c/k20", "judge/arm_c/k20"),
    ("D shared extraction", "arm_d_shared_extraction", "judge/arm_d_shared_extraction"),
    ("D own extraction", "arm_d", "judge/arm_d"),
    ("D + hydration", "arm_d_hydrated_2", "judge/arm_d_hydrated_2"),
]


def load_cache_dir(path: Path) -> dict:
    """Merge every conversation's cache in a directory into one {qa_id: entry}."""
    merged = {}
    if not path.exists():
        return merged
    for file in sorted(path.glob("*.json")):
        merged.update(json.loads(file.read_text()))
    return merged


def mean_or_none(values):
    return statistics.mean(values) if values else None


def summarize(name: str, answers_dir: Path, judge_dir: Path):
    answers = load_cache_dir(answers_dir)
    if not answers:
        return None
    verdicts = load_cache_dir(judge_dir)

    strict = defaultdict(list)
    lenient = defaultdict(list)
    agreement = defaultdict(list)
    for entry in verdicts.values():
        category = entry["category"]
        strict[category].append(entry["strict_verdict"] == "PASS")
        lenient[category].append(entry["original_verdict"] == "PASS")
        agreement[category].append(entry["strict_verdict"] == entry["original_verdict"])

    recall = defaultdict(list)
    latencies = []
    input_tokens = output_tokens = 0
    for entry in answers.values():
        if entry.get("recall") is not None:
            recall[entry["category"]].append(entry["recall"])
        input_tokens += entry.get("input_tokens", 0)
        output_tokens += entry.get("output_tokens", 0)
        if entry.get("retrieval_latency_ms") is not None:
            latencies.append(entry["retrieval_latency_ms"])

    judge_input = sum(e.get("input_tokens", 0) for e in verdicts.values())
    judge_output = sum(e.get("output_tokens", 0) for e in verdicts.values())

    def flatten(d):
        return [v for values in d.values() for v in values]

    return {
        "name": name,
        "n": len(answers),
        "n_judged": len(verdicts),
        "strict": {c: mean_or_none(strict[c]) for c in CATEGORY_NAMES},
        "strict_all": mean_or_none(flatten(strict)),
        "lenient": {c: mean_or_none(lenient[c]) for c in CATEGORY_NAMES},
        "lenient_all": mean_or_none(flatten(lenient)),
        "agreement": {c: mean_or_none(agreement[c]) for c in CATEGORY_NAMES},
        "agreement_all": mean_or_none(flatten(agreement)),
        "recall": {c: mean_or_none(recall[c]) for c in CATEGORY_NAMES},
        "recall_all": mean_or_none(flatten(recall)),
        "tokens_per_question": input_tokens / len(answers),
        # p95 by nearest-rank; with n=40 this is the 38th slowest.
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else None,
        "answer_cost": input_tokens / 1e6 * ANSWERING_MODEL_INPUT_PRICE_PER_M
        + output_tokens / 1e6 * ANSWERING_MODEL_OUTPUT_PRICE_PER_M,
        "judge_cost": judge_input / 1e6 * JUDGE_MODEL_INPUT_PRICE_PER_M
        + judge_output / 1e6 * JUDGE_MODEL_OUTPUT_PRICE_PER_M,
    }


def as_percent(value):
    return f"{value * 100:.1f}%" if value is not None else "-"


def as_ratio(value):
    return f"{value:.3f}" if value is not None else "-"


def print_markdown(rows):
    print("| configuration | multi-hop | temporal | open-dom. | single-hop | accuracy | lenient | recall | tok/q | p95 ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cats = "".join(f" {as_percent(r['strict'][c])} |" for c in CATEGORY_NAMES)
        p95 = f"{r['p95_latency_ms']:.0f}" if r["p95_latency_ms"] else "-"
        print(
            f"| **{r['name']}** |{cats} **{as_percent(r['strict_all'])}** | "
            f"{as_percent(r['lenient_all'])} | {as_ratio(r['recall_all'])} | "
            f"{r['tokens_per_question']:,.0f} | {p95} |"
        )


def print_recall_markdown(rows):
    print("| configuration | multi-hop | temporal | open-dom. | single-hop | all |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["recall_all"] is None:
            continue
        cats = "".join(f" {as_ratio(r['recall'][c])} |" for c in CATEGORY_NAMES)
        print(f"| **{r['name']}** |{cats} **{as_ratio(r['recall_all'])}** |")


def print_cost_markdown(rows):
    print("| configuration | answering | judging | total |")
    print("|---|---|---|---|")
    total_answer = total_judge = 0.0
    for r in rows:
        total_answer += r["answer_cost"]
        total_judge += r["judge_cost"]
        print(f"| {r['name']} | ${r['answer_cost']:.4f} | ${r['judge_cost']:.4f} | ${r['answer_cost'] + r['judge_cost']:.4f} |")
    print(f"| **total** | **${total_answer:.4f}** | **${total_judge:.4f}** | **${total_answer + total_judge:.4f}** |")


def print_agreement(rows):
    pooled = defaultdict(list)
    for name, answers_sub, judge_sub in CONFIGURATIONS:
        for entry in load_cache_dir(RUNS_DIR / judge_sub).values():
            pooled[entry["category"]].append(entry["strict_verdict"] == entry["original_verdict"])

    print("| category | judge agreement | n |")
    print("|---|---|---|")
    for category, label in CATEGORY_NAMES.items():
        values = pooled[category]
        if values:
            print(f"| {label} | {as_percent(statistics.mean(values))} | {len(values)} |")
    flat = [v for values in pooled.values() for v in values]
    print(f"| **all** | **{as_percent(statistics.mean(flat))}** | **{len(flat)}** |")

    lenient_pass_strict_fail = strict_pass_lenient_fail = 0
    for name, answers_sub, judge_sub in CONFIGURATIONS:
        for entry in load_cache_dir(RUNS_DIR / judge_sub).values():
            if entry["original_verdict"] == "PASS" and entry["strict_verdict"] == "FAIL":
                lenient_pass_strict_fail += 1
            elif entry["original_verdict"] == "FAIL" and entry["strict_verdict"] == "PASS":
                strict_pass_lenient_fail += 1
    print(f"\nlenient PASS / strict FAIL: {lenient_pass_strict_fail}")
    print(f"lenient FAIL / strict PASS: {strict_pass_lenient_fail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("main", "recall", "cost", "agreement", "all"),
        default="all",
        help="which table to print",
    )
    args = parser.parse_args()

    rows = [
        summary
        for name, answers_sub, judge_sub in CONFIGURATIONS
        if (summary := summarize(name, RUNS_DIR / answers_sub, RUNS_DIR / judge_sub))
    ]
    if not rows:
        print("no cached runs found under runs/")
        return

    if args.section in ("main", "all"):
        print("## Accuracy, recall, cost per question\n")
        print_markdown(rows)
    if args.section in ("recall", "all"):
        print("\n## Retrieval recall by category\n")
        print_recall_markdown(rows)
    if args.section in ("cost", "all"):
        print("\n## API spend\n")
        print_cost_markdown(rows)
    if args.section in ("agreement", "all"):
        print("\n## Judge agreement\n")
        print_agreement(rows)


if __name__ == "__main__":
    main()
