"""Arm C -- RAG over LLM-distilled facts (src/ingest/facts.py).

Retrieval is deliberately IDENTICAL to Arm B's, so B-vs-C isolates one
variable: raw chunk vs distilled fact. Do not change retrieval here without
changing Arm B too.

Temporal invalidation is deliberately not applied in this arm (docs/DESIGN.md).

Cache: runs/arm_c/k<N>/<conversation_id>.json.
"""
import argparse
import time

from anthropic import Anthropic
from openai import OpenAI

from src.arms.arm_a import ANSWER_STYLE_REMINDER, SYSTEM_PROMPT
from src.config import (
    ANSWERING_MODEL,
    ANTHROPIC_API_KEY,
    ARM_C_TOP_K,
    EMBEDDING_MODEL_PRICE_PER_M,
    OPENAI_API_KEY,
    RUNS_DIR,
    SAMPLE_CONVERSATIONS,
)
from src.arms.arm_b import embed_query, vector_literal
from src.cache_io import load_json_cache, save_json_cache
from src.db import get_connection
from src.ingest.chunks import EMBED_BATCH_SIZE, embed_batch, normalize
from src.dataset.load import apply_schema
from src.dataset.sample import select_stratified_sample


def embed_text_for_fact(subject: str, predicate: str, obj: str, fact: str) -> str:
    # Triple carries the canonical entity name, sentence carries the nuance.
    return f"{subject} {predicate} {obj}. {fact}"


def ingest_fact_embeddings(conn, client: OpenAI, conversation_id: str) -> int:
    rows = conn.execute(
        """
        SELECT f.fact_key, f.subject, f.predicate, f.object, f.fact
        FROM facts f
        LEFT JOIN fact_embeddings e ON e.fact_key = f.fact_key
        WHERE f.conversation_id = %s AND e.fact_key IS NULL
        """,
        (conversation_id,),
    ).fetchall()

    total_tokens = 0
    for i in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = rows[i : i + EMBED_BATCH_SIZE]
        texts = [embed_text_for_fact(s, p, o, f) for _k, s, p, o, f in batch]
        vectors, tokens = embed_batch(client, texts)
        total_tokens += tokens
        with conn.cursor() as cur:
            for (fact_key, *_rest), vector in zip(batch, vectors):
                cur.execute(
                    """
                    INSERT INTO fact_embeddings (fact_key, conversation_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (fact_key) DO NOTHING
                    """,
                    (fact_key, conversation_id, vector_literal(normalize(vector))),
                )
        conn.commit()

    print(f"{conversation_id}: {len(rows)} new fact embeddings")
    return total_tokens


def retrieve(conn, conversation_id: str, query_vector: list[float], k: int):
    start = time.monotonic()
    rows = conn.execute(
        """
        SELECT f.fact_key, f.subject, f.predicate, f.object, f.fact,
               f.source_turn_id, f.session_date, f.valid_from,
               e.embedding <=> %s::vector AS distance
        FROM fact_embeddings e JOIN facts f ON f.fact_key = e.fact_key
        WHERE e.conversation_id = %s
        ORDER BY distance ASC
        LIMIT %s
        """,
        (vector_literal(query_vector), conversation_id, k),
    ).fetchall()
    latency_ms = (time.monotonic() - start) * 1000
    return rows, latency_ms


def build_context(facts) -> str:
    """Label both halves of the bi-temporal pair explicitly: when the fact was SAID
    (session_date) and, where extraction resolved one, when it OCCURRED (valid_from).

    Both labels are required. Prefixing a bare date -- "[2023-01-19] Jon lost his
    job yesterday" -- is ambiguous: the fact sentence keeps its original relative
    wording by design, so the answering model resolved "yesterday" a second time
    against the already-resolved date and answered 2023-01-18. That
    double-resolution caused 6 of Arm C's 11 failures at k=10 (2026-08-12), all
    off-by-one-day errors with recall=1.0.

    Showing occurred (not just said) is also the point of paying for extraction:
    the temporal resolution is done once at ingestion instead of re-derived by the
    answering model on every query.
    """
    lines = []
    for _key, _subj, _pred, _obj, fact, _turn, session_date, valid_from in [f[:8] for f in facts]:
        said = session_date[:10]
        stamp = f"said {said}" if not valid_from or valid_from[:10] == said else f"said {said}; occurred {valid_from[:10]}"
        lines.append(f"({stamp}) {fact}")
    return "\n".join(lines)


def compute_recall(conversation_id: str, evidence_dia_ids: list[str], facts):
    # Same definition as every other arm: did retrieval surface the evidence turns?
    if not evidence_dia_ids:
        return None
    evidence_turn_ids = [f"{conversation_id}:{d}" for d in evidence_dia_ids]
    retrieved_turn_ids = {f[5] for f in facts}
    covered = sum(1 for t in evidence_turn_ids if t in retrieved_turn_ids)
    return covered / len(evidence_turn_ids)


def cache_dir(top_k: int):
    return RUNS_DIR / "arm_c" / f"k{top_k}"


def load_cache(top_k: int, conversation_id: str) -> dict:
    return load_json_cache(cache_dir(top_k) / f"{conversation_id}.json")


def save_cache(top_k: int, conversation_id: str, cache: dict) -> None:
    save_json_cache(cache_dir(top_k) / f"{conversation_id}.json", cache)


def run_question(conn, anthropic_client, openai_client, conversation_id, top_k, question, category, evidence):
    query_vector, embed_tokens = embed_query(openai_client, question)
    facts, retrieval_latency_ms = retrieve(conn, conversation_id, query_vector, top_k)
    context = build_context(facts)
    recall = compute_recall(conversation_id, evidence, facts)

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
        "retrieved_fact_keys": [f[0] for f in facts],
        "system_answer": answer_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "embed_tokens": embed_tokens,
        "retrieval_latency_ms": retrieval_latency_ms,
        "recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=ARM_C_TOP_K, help=f"default: {ARM_C_TOP_K}")
    parser.add_argument("--ingest-only", action="store_true", help="embed facts, don't answer questions")
    args = parser.parse_args()

    conn = get_connection()
    apply_schema(conn)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    embed_tokens = 0
    for conversation_id in SAMPLE_CONVERSATIONS:
        embed_tokens += ingest_fact_embeddings(conn, openai_client, conversation_id)
    if embed_tokens:
        print(f"fact embedding ingestion: {embed_tokens} tokens, ${(embed_tokens/1_000_000)*EMBEDDING_MODEL_PRICE_PER_M:.4f}")
    if args.ingest_only:
        conn.close()
        return

    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    sample = select_stratified_sample(conn)
    evidence_by_qa = {}
    for conversation_id in sample:
        rows = conn.execute(
            "SELECT qa_id, evidence FROM qa_pairs WHERE conversation_id = %s", (conversation_id,)
        ).fetchall()
        evidence_by_qa.update(dict(rows))

    for conversation_id, questions in sample.items():
        cache = load_cache(args.top_k, conversation_id)
        new_calls = 0
        for qa_id, question, category in questions:
            if qa_id in cache:
                continue
            cache[qa_id] = run_question(
                conn, anthropic_client, openai_client, conversation_id, args.top_k,
                question, category, evidence_by_qa.get(qa_id, []),
            )
            new_calls += 1
            save_cache(args.top_k, conversation_id, cache)
        print(f"[arm_c k={args.top_k}] {conversation_id}: {len(cache)} answered ({new_calls} new this run)")

    conn.close()


if __name__ == "__main__":
    main()
