"""Arm C's extraction: flat subject/predicate/object triples plus the original
sentence, into `facts`. FROZEN -- the shared-extraction Arm D numbers were
computed from this table and are still reported.

Per-session, not per-turn: a turn often cannot resolve its own pronouns.
Coreference is not resolved across session boundaries.
"""
import argparse
import hashlib
import json

from anthropic import Anthropic

from src.config import ANSWERING_MODEL, ANTHROPIC_API_KEY, SAMPLE_CONVERSATIONS
from src.db import get_connection
from src.dataset.load import apply_schema

EXTRACTION_SYSTEM_PROMPT = """You extract structured facts from one session of a conversation between two people.

For each factual assertion, output one fact with:
- subject: who or what the fact is about (use the person's actual name when known, not a pronoun)
- predicate: a short verb phrase for the relationship or action (e.g. "lost_job", "likes", "lives_in", "considering_quitting")
- object: what the subject relates to
- fact: the complete sentence stating the fact in plain English, preserving hedging, conditionals, and tone from the original -- e.g. "I think I'll quit if they don't raise my salary" stays as ONE fact; do not split the condition into a separate fact or drop the hedge
- source_turn_id: the dia_id (e.g. "D1:3") of the turn this fact is grounded in
- valid_from: the ISO-8601 date (YYYY-MM-DD) the event happened, but ONLY when the conversation pins it to one specific day. Otherwise use null -- never approximate.

When to set valid_from (resolve against the session date given below):
- An explicit date: "on March 3rd" -> that date.
- A reference that names one day: "yesterday", "this morning", "last Friday", "two days ago" -> compute it.

When to use null instead:
- A vague or multi-day reference: "last week", "last month", "a few days ago", "recently", "a while back", "earlier this year". These describe a RANGE, not a day. Do NOT convert them into a specific date by subtracting days from the session date -- that invents precision the speaker did not give. The phrase stays in the fact sentence, which is enough.
- No temporal reference at all, or a standing state ("Jon likes contemporary dance").

Rules:
- Extract one fact per assertion, not per message -- a single turn may yield multiple facts, or none.
- Resolve pronouns ("she", "he", "they") to the person's actual name when it's clear from earlier in this session. If it's genuinely ambiguous, keep the pronoun rather than guessing.
- Do not invent facts not stated or clearly implied in the conversation.
- Skip greetings, filler, and questions that don't assert anything.

Session date: {session_date}"""

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "fact": {"type": "string"},
                    "source_turn_id": {"type": "string"},
                    "valid_from": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["subject", "predicate", "object", "fact", "source_turn_id", "valid_from"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def build_session_transcript(conn, session_id: str) -> tuple[str, str, set[str]]:
    rows = conn.execute(
        """
        SELECT turn_id, session_date, speaker, text
        FROM raw_turns WHERE session_id = %s ORDER BY turn_index
        """,
        (session_id,),
    ).fetchall()
    session_date = rows[0][1]
    dia_ids = {turn_id.split(":", 1)[1] for turn_id, *_ in rows}  # 'conv-30:D1:3' -> 'D1:3'
    lines = [f"{turn_id.split(':', 1)[1]} {speaker}: {text}" for turn_id, _date, speaker, text in rows]
    return "\n".join(lines), session_date, dia_ids


def fact_key(subject: str, predicate: str, obj: str, source_turn_id: str) -> str:
    normalized = "|".join(s.strip().lower() for s in (subject, predicate, obj)) + f"|{source_turn_id}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_session(client: Anthropic, session_transcript: str, session_date: str) -> tuple[list[dict], int, int]:
    response = client.messages.create(
        model=ANSWERING_MODEL,
        max_tokens=4000,
        system=EXTRACTION_SYSTEM_PROMPT.format(session_date=session_date),
        messages=[{"role": "user", "content": session_transcript}],
        output_config={"format": {"type": "json_schema", "schema": FACTS_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    facts = json.loads(text)["facts"]
    return facts, response.usage.input_tokens, response.usage.output_tokens


def already_extracted_session_ids(conn, conversation_id: str) -> set[str]:
    """Sessions with at least one extracted fact.

    ON CONFLICT stops duplicate rows but not the billed API call; this skip is
    what makes a re-run after a crash free.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT r.session_id
        FROM facts f JOIN raw_turns r ON f.source_turn_id = r.turn_id
        WHERE f.conversation_id = %s
        """,
        (conversation_id,),
    ).fetchall()
    return {r[0] for r in rows}


def get_sessions(conn, conversation_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM raw_turns WHERE conversation_id = %s ORDER BY session_id",
        (conversation_id,),
    ).fetchall()
    return [r[0] for r in rows]


def run_session(conn, client: Anthropic, conversation_id: str, session_id: str) -> tuple[int, int, int]:
    transcript, session_date, dia_ids = build_session_transcript(conn, session_id)
    facts, in_tok, out_tok = extract_session(client, transcript, session_date)

    rows = []
    skipped = 0
    for f in facts:
        dia_id = f["source_turn_id"]
        if dia_id not in dia_ids:
            skipped += 1  # cited a turn not in this session; drop rather than guess
            continue
        source_turn_id = f"{conversation_id}:{dia_id}"
        key = fact_key(f["subject"], f["predicate"], f["object"], source_turn_id)
        rows.append((key, conversation_id, f["subject"], f["predicate"], f["object"], f["fact"], source_turn_id, session_date, f["valid_from"]))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO facts (fact_key, conversation_id, subject, predicate, object, fact, source_turn_id, session_date, valid_from)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fact_key) DO NOTHING
            """,
            rows,
        )
    conn.commit()

    if skipped:
        print(f"  {session_id}: {skipped} fact(s) dropped -- cited a turn not in this session")
    return len(rows), in_tok, out_tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="restrict to one conversation_id; default: SAMPLE_CONVERSATIONS")
    parser.add_argument("--session", help="restrict to one session_id (requires --conversation)")
    parser.add_argument("--force", action="store_true", help="re-extract sessions that already have facts (re-pays for them)")
    args = parser.parse_args()

    conn = get_connection()
    apply_schema(conn)
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    conversation_ids = [args.conversation] if args.conversation else SAMPLE_CONVERSATIONS
    total_facts = total_in = total_out = skipped = 0
    for conversation_id in conversation_ids:
        sessions = [args.session] if args.session else get_sessions(conn, conversation_id)
        done = set() if args.force else already_extracted_session_ids(conn, conversation_id)
        for session_id in sessions:
            if session_id in done:
                skipped += 1
                continue
            n_facts, in_tok, out_tok = run_session(conn, client, conversation_id, session_id)
            total_facts += n_facts
            total_in += in_tok
            total_out += out_tok
            print(f"{session_id}: {n_facts} facts extracted")

    conn.close()
    print(f"total: {total_facts} facts, {total_in} input tokens, {total_out} output tokens ({skipped} sessions skipped, already extracted)")


if __name__ == "__main__":
    main()
