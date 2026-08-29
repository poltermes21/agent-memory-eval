"""Arm D's own extraction. Writes to `graph_facts`, NEVER to `facts`.

Graph-native rather than flat triples: typed entities, LLM-lemmatized
predicates, one canonical name per multi-mention entity.

Costs real money; resumable per session. A failed session stays unmarked and is
retried free on the next run.

Why this rebuild did not improve the results: docs/GRAPH.md.
"""
import argparse
import hashlib
import json

from anthropic import Anthropic

from src.config import ANSWERING_MODEL, ANTHROPIC_API_KEY, SAMPLE_CONVERSATIONS
from src.db import get_connection
from src.ingest.facts import build_session_transcript, get_sessions
from src.dataset.load import apply_schema

ENTITY_TYPES = ("Person", "Place", "Organization", "Object", "Event", "Concept")

EXTRACTION_SYSTEM_PROMPT = """You extract a knowledge graph from one session of a conversation between two people: entities (nodes) and the relations between them (edges), one fact per relation.

For each factual assertion, output one fact with:
- subject: the entity the fact is about
- subject_type: one of {entity_types}
- predicate: a relation in BASE/INFINITIVE form -- lemmatized, not conjugated. Write "live_in" not "lives_in"/"lived_in"; "have" not "has"/"had"/"having"; "go_to" not "goes_to"/"went_to"; "feel" not "feels"/"felt". This matters most for IRREGULAR verbs (have/has/had, go/goes/went, make/makes/made, feel/felt) precisely because they don't share a spelling pattern across tenses -- lemmatize them yourself rather than leaving the surface form.
- object: what the subject relates to
- object_type: one of {entity_types}
- fact: the complete sentence stating the fact in plain English, preserving hedging, conditionals, and tone from the original -- e.g. "I think I'll quit if they don't raise my salary" stays as ONE fact; do not split the condition into a separate fact or drop the hedge
- source_turn_id: the dia_id (e.g. "D1:3") of the turn this fact is grounded in
- valid_from: the ISO-8601 date (YYYY-MM-DD) the event happened, but ONLY when the conversation pins it to one specific day. Otherwise use null -- never approximate.

When to set valid_from (resolve against the session date given below):
- An explicit date: "on March 3rd" -> that date.
- A reference that names one day: "yesterday", "this morning", "last Friday", "two days ago" -> compute it.

When to use null instead:
- A vague or multi-day reference: "last week", "last month", "a few days ago", "recently", "a while back", "earlier this year". These describe a RANGE, not a day. Do NOT convert them into a specific date by subtracting days from the session date -- that invents precision the speaker did not give. The phrase stays in the fact sentence, which is enough.
- No temporal reference at all, or a standing state ("Jon likes contemporary dance").

Entity naming -- this is the part that makes the graph connected, read it carefully:
- Use the person's actual name when known, not a pronoun: resolve "she", "he", "they" to the name it clearly refers to earlier in this session. If genuinely ambiguous, keep the pronoun.
- A person's relative, pet, possession, or other recurring thing that is referred to sometimes by a common noun ("her kids", "the kids", "his car", "my dog") and sometimes tied explicitly to the owner ("Melanie's kids") is still ONE entity across every mention in this session. Name it the SAME way every time it appears as a subject or object: "<Owner's name>'s <noun>" (e.g. "Melanie's kids", "Evan's Prius"), using the owner's resolved proper name, not "her"/"his"/"the". This is what lets the graph connect "Melanie's kids were scared" and "Melanie went swimming with the kids" to the same node instead of two disconnected strings.
- Do not invent a name for something the conversation never names distinctly -- vague references to "some friends" or "a place" with no further identifying detail can stay generic, since there is nothing to consistently resolve them to.

Rules:
- Extract one fact per assertion, not per message -- a single turn may yield multiple facts, or none.
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
                    "subject_type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "fact": {"type": "string"},
                    "source_turn_id": {"type": "string"},
                    "valid_from": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": [
                    "subject", "subject_type", "predicate", "object", "object_type",
                    "fact", "source_turn_id", "valid_from",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def fact_key(subject: str, predicate: str, obj: str, source_turn_id: str) -> str:
    normalized = "|".join(s.strip().lower() for s in (subject, predicate, obj)) + f"|{source_turn_id}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def extract_session(client: Anthropic, session_transcript: str, session_date: str) -> tuple[list[dict], int, int]:
    response = client.messages.create(
        model=ANSWERING_MODEL,
        # Higher than facts.py: this schema is heavier per fact, and a
        # long session truncated mid-JSON at 4000.
        max_tokens=8000,
        system=EXTRACTION_SYSTEM_PROMPT.format(session_date=session_date, entity_types=", ".join(ENTITY_TYPES)),
        messages=[{"role": "user", "content": session_transcript}],
        output_config={"format": {"type": "json_schema", "schema": FACTS_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    facts = json.loads(text)["facts"]
    return facts, response.usage.input_tokens, response.usage.output_tokens


def already_extracted_session_ids(conn, conversation_id: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT r.session_id
        FROM graph_facts f JOIN raw_turns r ON f.source_turn_id = r.turn_id
        WHERE f.conversation_id = %s
        """,
        (conversation_id,),
    ).fetchall()
    return {r[0] for r in rows}


def run_session(conn, client: Anthropic, conversation_id: str, session_id: str) -> tuple[int, int, int]:
    transcript, session_date, dia_ids = build_session_transcript(conn, session_id)
    facts, in_tok, out_tok = extract_session(client, transcript, session_date)

    rows = []
    skipped = 0
    for f in facts:
        dia_id = f["source_turn_id"]
        if dia_id not in dia_ids:
            skipped += 1
            continue
        source_turn_id = f"{conversation_id}:{dia_id}"
        key = fact_key(f["subject"], f["predicate"], f["object"], source_turn_id)
        rows.append((
            key, conversation_id, f["subject"], f["subject_type"], f["predicate"],
            f["object"], f["object_type"], f["fact"], source_turn_id, session_date, f["valid_from"],
        ))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO graph_facts
                (fact_key, conversation_id, subject, subject_type, predicate,
                 object, object_type, fact, source_turn_id, session_date, valid_from)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
    total_facts = total_in = total_out = skipped = failed = 0
    for conversation_id in conversation_ids:
        sessions = [args.session] if args.session else get_sessions(conn, conversation_id)
        done = set() if args.force else already_extracted_session_ids(conn, conversation_id)
        for session_id in sessions:
            if session_id in done:
                skipped += 1
                continue
            try:
                n_facts, in_tok, out_tok = run_session(conn, client, conversation_id, session_id)
            except Exception as exc:
                # One bad response must not kill the batch. Nothing is written
                # for this session, so a re-run retries it for free.
                failed += 1
                print(f"  {session_id}: FAILED ({type(exc).__name__}: {exc}) -- skipping, retry on next run")
                continue
            total_facts += n_facts
            total_in += in_tok
            total_out += out_tok
            print(f"{session_id}: {n_facts} facts extracted")

    conn.close()
    print(f"total: {total_facts} facts, {total_in} input tokens, {total_out} output tokens "
          f"({skipped} sessions skipped/already extracted, {failed} failed)")


if __name__ == "__main__":
    main()
