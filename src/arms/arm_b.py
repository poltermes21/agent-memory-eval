"""Arm B -- RAG over raw text chunks: embed the question, exact cosine top-k,
answer with the same frozen model and prompt as every other arm. Search is
scoped to the question's own conversation.

Runs three chunk granularities: turn, session, window (see src/ingest/chunks.py).

Cache: runs/arm_b/<granularity>/k<N>/<conversation_id>.json, one folder per k.
"""
import argparse
import time

from anthropic import Anthropic
from openai import OpenAI

from src.arms.arm_a import ANSWER_STYLE_REMINDER, SYSTEM_PROMPT
from src.cache_io import load_json_cache, save_json_cache
from src.config import (
    ANSWERING_MODEL,
    ANTHROPIC_API_KEY,
    ARM_B_TOP_K,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RUNS_DIR,
)
from src.db import get_connection
from src.ingest.chunks import normalize
from src.dataset.sample import select_stratified_sample

GRANULARITIES = ("turn", "session", "window")


def embed_query(client: OpenAI, text: str) -> tuple[list[float], int]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return normalize(response.data[0].embedding), response.usage.total_tokens


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in vector) + "]"


def retrieve(conn, conversation_id: str, granularity: str, query_vector: list[float], k: int):
    start = time.monotonic()
    rows = conn.execute(
        """
        SELECT chunk_id, turn_id, turn_ids, session_id, session_date, speaker, text,
               embedding <=> %s::vector AS distance
        FROM embeddings
        WHERE conversation_id = %s AND granularity = %s
        ORDER BY distance ASC
        LIMIT %s
        """,
        (vector_literal(query_vector), conversation_id, granularity, k),
    ).fetchall()
    latency_ms = (time.monotonic() - start) * 1000
    return rows, latency_ms


def build_context(chunks) -> str:
    lines = []
    for _chunk_id, _turn_id, _turn_ids, session_id, session_date, speaker, text, _distance in chunks:
        lines.append(f"\n--- {session_id} ({session_date}) ---")
        lines.append(f"{speaker}: {text}" if speaker else text)
    return "\n".join(lines)


def compute_recall(conversation_id: str, evidence_dia_ids: list[str], chunks):
    # Unified across every granularity: each chunk's turn_ids (db/schema.sql) lists
    # every turn it covers, so "was the evidence retrieved" is one set check no
    # matter whether a chunk is one turn, a whole session, or a sliding window.
    if not evidence_dia_ids:
        return None
    evidence_turn_ids = [f"{conversation_id}:{d}" for d in evidence_dia_ids]
    covered_turn_ids: set[str] = set()
    for _chunk_id, _turn_id, turn_ids, _session_id, _session_date, _speaker, _text, _distance in chunks:
        covered_turn_ids.update(turn_ids)
    covered = sum(1 for t in evidence_turn_ids if t in covered_turn_ids)
    return covered / len(evidence_turn_ids)


def cache_dir(granularity: str, top_k: int):
    return RUNS_DIR / "arm_b" / granularity / f"k{top_k}"


def load_cache(granularity: str, top_k: int, conversation_id: str) -> dict:
    return load_json_cache(cache_dir(granularity, top_k) / f"{conversation_id}.json")


def save_cache(granularity: str, top_k: int, conversation_id: str, cache: dict) -> None:
    save_json_cache(cache_dir(granularity, top_k) / f"{conversation_id}.json", cache)


def run_question(conn, anthropic_client, openai_client, conversation_id, granularity, top_k, qa_id, question, category, evidence):
    query_vector, embed_tokens = embed_query(openai_client, question)
    chunks, retrieval_latency_ms = retrieve(conn, conversation_id, granularity, query_vector, top_k)
    context = build_context(chunks)
    recall = compute_recall(conversation_id, evidence, chunks)

    response = anthropic_client.messages.create(
        model=ANSWERING_MODEL,
        max_tokens=512,
        system=f"{SYSTEM_PROMPT}\n\n{context}",
        messages=[{"role": "user", "content": f"{question}\n\n({ANSWER_STYLE_REMINDER})"}],
    )
    answer_text = next(b.text for b in response.content if b.type == "text").strip()

    return {
        "question": question,
        "category": category,
        "retrieved_chunk_ids": [c[0] for c in chunks],
        "system_answer": answer_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "embed_tokens": embed_tokens,
        "retrieval_latency_ms": retrieval_latency_ms,
        "recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--granularity", choices=GRANULARITIES, help="default: both")
    parser.add_argument("--top-k", type=int, default=ARM_B_TOP_K, help=f"default: {ARM_B_TOP_K}")
    args = parser.parse_args()

    conn = get_connection()
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    sample = select_stratified_sample(conn)  # {conversation_id: [(qa_id, question, category), ...]}
    # evidence isn't in the sample tuples -- fetch it alongside
    evidence_by_qa = {}
    for conversation_id, questions in sample.items():
        rows = conn.execute(
            "SELECT qa_id, evidence FROM qa_pairs WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchall()
        evidence_by_qa.update(dict(rows))

    granularities = [args.granularity] if args.granularity else list(GRANULARITIES)
    for granularity in granularities:
        for conversation_id, questions in sample.items():
            cache = load_cache(granularity, args.top_k, conversation_id)
            new_calls = 0
            for qa_id, question, category in questions:
                if qa_id in cache:
                    continue
                evidence = evidence_by_qa.get(qa_id, [])
                cache[qa_id] = run_question(
                    conn, anthropic_client, openai_client, conversation_id, granularity, args.top_k, qa_id, question, category, evidence
                )
                new_calls += 1
                save_cache(granularity, args.top_k, conversation_id, cache)
            print(f"[{granularity} k={args.top_k}] {conversation_id}: {len(cache)} answered ({new_calls} new this run)")

    conn.close()


if __name__ == "__main__":
    main()
