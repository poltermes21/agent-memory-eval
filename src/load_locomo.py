"""Parse data/raw/locomo10.json into raw_turns and qa_pairs (db/schema.sql).

Idempotent: turn_id / qa_id are stable natural keys derived from the source JSON
(conversation sample_id + dia_id / qa index), so re-running never duplicates rows —
ON CONFLICT DO NOTHING.

Turns that carry an image (img_url / blip_caption / query) are loaded with just their
text, same as any other turn. CLAUDE.md's raw_turns schema is text-only (speaker, text);
image fields are dropped, not stored elsewhere -- out of scope for this benchmark.
"""
import argparse
import json
from datetime import datetime

import psycopg

from src.config import LOCOMO_PATH, REPO_ROOT
from src.db import get_connection

SESSION_DATE_FORMAT = "%I:%M %p on %d %B, %Y"


def parse_session_date(raw: str) -> str:
    return datetime.strptime(raw, SESSION_DATE_FORMAT).isoformat()


def apply_schema(conn: psycopg.Connection) -> None:
    schema_path = REPO_ROOT / "db" / "schema.sql"
    conn.execute(schema_path.read_text())
    conn.commit()


def load_sample(conn: psycopg.Connection, sample: dict) -> tuple[int, int]:
    conversation_id = sample["sample_id"]
    conv = sample["conversation"]

    turn_rows = []
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    for session_key in session_keys:
        n = session_key.split("_")[1]
        date_key = f"session_{n}_date_time"
        session_date = parse_session_date(conv[date_key])
        session_id = f"{conversation_id}:session_{n}"
        for turn_index, turn in enumerate(conv[session_key], start=1):
            turn_id = f"{conversation_id}:{turn['dia_id']}"
            turn_rows.append((turn_id, conversation_id, session_id, session_date, turn_index, turn["speaker"], turn["text"]))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO raw_turns (turn_id, conversation_id, session_id, session_date, turn_index, speaker, text)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (turn_id) DO NOTHING
            """,
            turn_rows,
        )

    qa_rows = []
    for idx, qa in enumerate(sample.get("qa", [])):
        qa_id = f"{conversation_id}:qa:{idx}"
        answer = qa.get("answer")
        qa_rows.append((
            qa_id,
            conversation_id,
            qa["question"],
            None if answer is None else str(answer),
            qa["category"],
            qa.get("evidence", []),
            qa.get("adversarial_answer"),
        ))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO qa_pairs (qa_id, conversation_id, question, expected_answer, category, evidence, adversarial_answer)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (qa_id) DO NOTHING
            """,
            qa_rows,
        )

    conn.commit()
    return len(turn_rows), len(qa_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(LOCOMO_PATH))
    args = parser.parse_args()

    samples = json.loads(open(args.path).read())

    conn = get_connection()
    apply_schema(conn)

    total_turns = total_qa = 0
    for sample in samples:
        n_turns, n_qa = load_sample(conn, sample)
        total_turns += n_turns
        total_qa += n_qa
        print(f"{sample['sample_id']}: {n_turns} turns, {n_qa} qa pairs")

    conn.close()
    print(f"done: {len(samples)} conversations, {total_turns} turns, {total_qa} qa pairs")


if __name__ == "__main__":
    main()
