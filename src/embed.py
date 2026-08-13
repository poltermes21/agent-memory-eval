"""Arm B ingestion: chunk turns three ways (per-turn, per-session, sliding
window of turns), embed each chunk via OpenAI text-embedding-3-small,
normalize to unit norm, store in the shared Postgres embeddings table
(db/schema.sql).

Turn and session are CLAUDE.md's deliberate bookends -- they measure the span
of the granularity design variable, not a production recommendation. Window
(WINDOW_SIZE consecutive turns, WINDOW_STRIDE apart, so consecutive windows
overlap) approximates what production RAG systems actually ship: 2026
industry practice favors grouped chunks (~512 tokens) with 10-15% overlap
over single-utterance or whole-document chunking. Windows never cross session
boundaries -- grouping turns from two different days into one chunk has no
real analogue in how conversations are read.

Scoped to SAMPLE_CONVERSATIONS only, the same fixed subset as Arm A, so the
arm comparison stays controlled. Retrieval (not ingestion) is an exact
sequential scan -- see CLAUDE.md "Stack": no ANN index, ever.
"""
import argparse
import math

from openai import OpenAI

from src.config import EMBEDDING_MODEL, EMBEDDING_MODEL_PRICE_PER_M, OPENAI_API_KEY, SAMPLE_CONVERSATIONS
from src.db import get_connection
from src.load_locomo import apply_schema

EMBED_BATCH_SIZE = 100
WINDOW_SIZE = 5
WINDOW_STRIDE = 3  # overlap = WINDOW_SIZE - WINDOW_STRIDE = 2 turns between consecutive windows


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in vector) + "]"


def build_turn_chunks(conn, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT turn_id, session_id, session_date, speaker, text
        FROM raw_turns WHERE conversation_id = %s ORDER BY session_id, turn_index
        """,
        (conversation_id,),
    ).fetchall()
    return [
        {
            "chunk_id": f"turn:{turn_id}",
            "conversation_id": conversation_id,
            "granularity": "turn",
            "turn_id": turn_id,
            "turn_ids": [turn_id],
            "session_id": session_id,
            "session_date": session_date,
            "speaker": speaker,
            "text": text,
        }
        for turn_id, session_id, session_date, speaker, text in rows
    ]


def build_session_chunks(conn, conversation_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT turn_id, session_id, session_date, speaker, text
        FROM raw_turns WHERE conversation_id = %s ORDER BY session_id, turn_index
        """,
        (conversation_id,),
    ).fetchall()
    sessions: dict[str, dict] = {}
    for turn_id, session_id, session_date, speaker, text in rows:
        if session_id not in sessions:
            sessions[session_id] = {"session_date": session_date, "lines": [], "turn_ids": []}
        sessions[session_id]["lines"].append(f"{speaker}: {text}")
        sessions[session_id]["turn_ids"].append(turn_id)
    return [
        {
            "chunk_id": f"session:{session_id}",
            "conversation_id": conversation_id,
            "granularity": "session",
            "turn_id": None,
            "turn_ids": data["turn_ids"],
            "session_id": session_id,
            "session_date": data["session_date"],
            "speaker": None,
            "text": "\n".join(data["lines"]),
        }
        for session_id, data in sessions.items()
    ]


def build_window_chunks(conn, conversation_id: str, window: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> list[dict]:
    rows = conn.execute(
        """
        SELECT turn_id, session_id, session_date, speaker, text
        FROM raw_turns WHERE conversation_id = %s ORDER BY session_id, turn_index
        """,
        (conversation_id,),
    ).fetchall()

    sessions: dict[str, list] = {}
    for turn_id, session_id, session_date, speaker, text in rows:
        sessions.setdefault(session_id, []).append((turn_id, session_date, speaker, text))

    chunks = []
    for session_id, turns in sessions.items():
        i = 0
        while i < len(turns):
            window_turns = turns[i : i + window]
            turn_ids = [t[0] for t in window_turns]
            lines = [f"{speaker}: {text}" for _tid, _date, speaker, text in window_turns]
            chunks.append(
                {
                    "chunk_id": f"window:{session_id}:{turn_ids[0]}",
                    "conversation_id": conversation_id,
                    "granularity": "window",
                    "turn_id": None,
                    "turn_ids": turn_ids,
                    "session_id": session_id,
                    "session_date": window_turns[0][1],
                    "speaker": None,
                    "text": "\n".join(lines),
                }
            )
            if i + window >= len(turns):
                break
            i += stride
    return chunks


def already_embedded_chunk_ids(conn, conversation_id: str, granularity: str) -> set[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM embeddings WHERE conversation_id = %s AND granularity = %s",
        (conversation_id, granularity),
    ).fetchall()
    return {r[0] for r in rows}


def embed_batch(client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data], response.usage.total_tokens


GRANULARITY_BUILDERS = (("turn", build_turn_chunks), ("session", build_session_chunks), ("window", build_window_chunks))


def ingest_conversation(conn, client: OpenAI, conversation_id: str) -> int:
    total_tokens = 0
    for granularity, build_fn in GRANULARITY_BUILDERS:
        chunks = build_fn(conn, conversation_id)
        existing = already_embedded_chunk_ids(conn, conversation_id, granularity)
        pending = [c for c in chunks if c["chunk_id"] not in existing]

        for i in range(0, len(pending), EMBED_BATCH_SIZE):
            batch = pending[i : i + EMBED_BATCH_SIZE]
            vectors, tokens = embed_batch(client, [c["text"] for c in batch])
            total_tokens += tokens
            with conn.cursor() as cur:
                for chunk, vector in zip(batch, vectors):
                    cur.execute(
                        """
                        INSERT INTO embeddings
                            (chunk_id, conversation_id, granularity, turn_id, turn_ids, session_id, session_date, speaker, text, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        (
                            chunk["chunk_id"], chunk["conversation_id"], chunk["granularity"], chunk["turn_id"],
                            chunk["turn_ids"], chunk["session_id"], chunk["session_date"], chunk["speaker"], chunk["text"],
                            vector_literal(normalize(vector)),
                        ),
                    )
            conn.commit()

        print(f"{conversation_id} [{granularity}]: {len(pending)} new chunks embedded ({len(chunks)} total)")
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="restrict to one conversation_id; default: SAMPLE_CONVERSATIONS")
    args = parser.parse_args()

    conn = get_connection()
    apply_schema(conn)
    client = OpenAI(api_key=OPENAI_API_KEY)

    conversation_ids = [args.conversation] if args.conversation else SAMPLE_CONVERSATIONS
    total_tokens = 0
    for conversation_id in conversation_ids:
        total_tokens += ingest_conversation(conn, client, conversation_id)
    conn.close()

    cost = (total_tokens / 1_000_000) * EMBEDDING_MODEL_PRICE_PER_M
    print(f"total: {total_tokens} tokens embedded, ${cost:.4f}")


if __name__ == "__main__":
    main()
