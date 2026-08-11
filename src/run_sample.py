"""Stratified reference sample for Arm A: SAMPLE_PER_CATEGORY questions per
category (1-4, adversarial excluded), spread round-robin across
SAMPLE_CONVERSATIONS. This is the fixed subset all arms (A-D) will reuse for
the controlled comparison -- see config.py for why it must stay fixed.

Answers land in the same runs/arm_a/<conversation_id>.json cache as a full
run, so this and a later full run never duplicate work.
"""
from collections import defaultdict

from anthropic import Anthropic

from src.arm_a import run_questions
from src.config import ANTHROPIC_API_KEY, SAMPLE_CONVERSATIONS, SAMPLE_PER_CATEGORY
from src.db import get_connection

CATEGORIES = (1, 2, 3, 4)


def select_stratified_sample(conn) -> dict:
    """Returns {conversation_id: [(qa_id, question, category), ...]}, with
    SAMPLE_PER_CATEGORY questions per category, round-robinned across
    SAMPLE_CONVERSATIONS so no single conversation dominates a category.
    """
    rows = conn.execute(
        """
        SELECT category, conversation_id, qa_id, question
        FROM qa_pairs
        WHERE conversation_id = ANY(%s) AND category = ANY(%s)
        ORDER BY category, conversation_id, qa_id
        """,
        (SAMPLE_CONVERSATIONS, list(CATEGORIES)),
    ).fetchall()

    by_category_conv = defaultdict(list)
    for category, conversation_id, qa_id, question in rows:
        by_category_conv[(category, conversation_id)].append((conversation_id, qa_id, question, category))

    selected = defaultdict(list)
    for category in CATEGORIES:
        pools = [by_category_conv[(category, c)] for c in SAMPLE_CONVERSATIONS if by_category_conv[(category, c)]]
        picked = 0
        idx = 0
        while picked < SAMPLE_PER_CATEGORY and any(pools):
            pool = pools[idx % len(pools)]
            if pool:
                conversation_id, qa_id, question, cat = pool.pop(0)
                selected[conversation_id].append((qa_id, question, cat))
                picked += 1
            idx += 1
    return selected


def main() -> None:
    conn = get_connection()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    sample = select_stratified_sample(conn)
    total = sum(len(v) for v in sample.values())
    print(f"stratified sample: {total} questions across {len(sample)} conversations")

    for conversation_id, questions in sample.items():
        run_questions(conn, client, conversation_id, questions)

    conn.close()


if __name__ == "__main__":
    main()
