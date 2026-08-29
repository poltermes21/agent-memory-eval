"""Arm A -- full context: the whole transcript goes in the prompt, no retrieval.

SYSTEM_PROMPT and ANSWER_STYLE_REMINDER are imported by every other arm. Change
them here and every arm's cached answers stop being comparable.

Cache: runs/arm_a/<conversation_id>.json.
"""
import argparse

from anthropic import Anthropic

from src.cache_io import load_json_cache, save_json_cache
from src.config import ADVERSARIAL_CATEGORY, ANSWERING_MODEL, ANTHROPIC_API_KEY, RUNS_DIR
from src.db import get_connection

ARM_A_DIR = RUNS_DIR / "arm_a"

SYSTEM_PROMPT = (
    "Below is a conversation between two people, spanning several sessions. You will "
    "be asked questions about the people in it. Answer the way a person would if "
    "they simply remembered the fact themselves -- direct and natural, never like "
    "someone citing a document. Never write phrases like 'according to the "
    "transcript', 'mentioned in session X', or a session date in parentheses, and "
    "never quote the original wording -- just state the fact itself, resolved to a "
    "concrete value (an exact date, name, or number) whenever the conversation makes "
    "one available, not a relative reference like 'yesterday' or 'that day'. Be "
    "concise: no preamble, no explanation of how you found it. If the conversation "
    "does not contain the answer, reply exactly with: unknown."
)

# Repeated next to the question (not just once before the transcript) because the
# transcript can run 20-30k tokens -- an instruction stated only at the top measurably
# loses adherence by the time the model reaches the question. See project memory
# entry on the citation/verbosity issue found via manual judge testing, 2026-08-10.
ANSWER_STYLE_REMINDER = (
    "Answer directly and naturally, as if you simply knew this yourself. "
    "No citing the transcript, no session references, no quoted text, no relative "
    "dates -- give the resolved fact."
)


def build_transcript(conn, conversation_id: str) -> str:
    rows = conn.execute(
        """
        SELECT session_id, session_date, speaker, text
        FROM raw_turns
        WHERE conversation_id = %s
        ORDER BY session_id, turn_index
        """,
        (conversation_id,),
    ).fetchall()

    lines = []
    current_session = None
    for session_id, session_date, speaker, text in rows:
        if session_id != current_session:
            lines.append(f"\n--- {session_id} ({session_date}) ---")
            current_session = session_id
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def fetch_questions(conn, conversation_id: str, include_adversarial: bool):
    query = "SELECT qa_id, question, category FROM qa_pairs WHERE conversation_id = %s"
    params = [conversation_id]
    if not include_adversarial:
        query += " AND category != %s"
        params.append(ADVERSARIAL_CATEGORY)
    query += " ORDER BY qa_id"
    return conn.execute(query, params).fetchall()


def load_cache(conversation_id: str) -> dict:
    return load_json_cache(ARM_A_DIR / f"{conversation_id}.json")


def save_cache(conversation_id: str, cache: dict) -> None:
    save_json_cache(ARM_A_DIR / f"{conversation_id}.json", cache)


def run_questions(conn, client: Anthropic, conversation_id: str, questions) -> None:
    """questions: iterable of (qa_id, question, category) to answer for this conversation.
    Shared by the full per-conversation run and the stratified sample runner
    (src/dataset/sample.py) so both write to the same cache format.
    """
    transcript = build_transcript(conn, conversation_id)
    cache = load_cache(conversation_id)
    new_answers = 0
    for qa_id, question, category in questions:
        if qa_id in cache:
            continue
        response = client.messages.create(
            model=ANSWERING_MODEL,
            max_tokens=512,
            system=f"{SYSTEM_PROMPT}\n\n{transcript}",
            messages=[{"role": "user", "content": f"{question}\n\n({ANSWER_STYLE_REMINDER})"}],
        )
        answer_text = next(block.text for block in response.content if block.type == "text").strip()
        cache[qa_id] = {
            "question": question,
            "category": category,
            "system_answer": answer_text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        new_answers += 1
        if new_answers % 10 == 0:
            save_cache(conversation_id, cache)

    save_cache(conversation_id, cache)
    print(f"{conversation_id}: {len(cache)} cached ({new_answers} new this run)")


def run_conversation(conn, client: Anthropic, conversation_id: str, include_adversarial: bool, limit: int | None) -> None:
    questions = fetch_questions(conn, conversation_id, include_adversarial)
    if limit:
        questions = questions[:limit]
    run_questions(conn, client, conversation_id, questions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="run a single conversation_id; default: all")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--limit", type=int, help="max questions per conversation, for smoke testing")
    args = parser.parse_args()

    conn = get_connection()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    if args.conversation:
        conversation_ids = [args.conversation]
    else:
        conversation_ids = [r[0] for r in conn.execute("SELECT DISTINCT conversation_id FROM raw_turns ORDER BY conversation_id").fetchall()]

    for conversation_id in conversation_ids:
        run_conversation(conn, client, conversation_id, args.include_adversarial, args.limit)

    conn.close()


if __name__ == "__main__":
    main()
