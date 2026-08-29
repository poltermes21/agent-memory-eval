"""Arm D -- RAG over a temporal knowledge graph, built by src/ingest/graph.py from
Arm D's own extraction (src/ingest/graph_facts.py).

Retrieval: vector seed -> per-entity coverage -> 1 Cypher hop. Traversal must
stay gated by ARM_D_HUB_DEGREE; the graph is a star, and expanding a hub returns
the whole conversation. Superseded edges are returned with their interval, never
filtered out.

Why the graph did not beat flat retrieval: docs/GRAPH.md.

Cache: runs/arm_d/<conversation_id>.json; --hydrate N uses
runs/arm_d_hydrated_N/.
"""
import argparse
import time

from anthropic import Anthropic
from openai import OpenAI

from src.arms.arm_a import ANSWER_STYLE_REMINDER, SYSTEM_PROMPT
from src.arms.arm_b import embed_query, vector_literal
from src.cache_io import load_json_cache, save_json_cache
from src.config import (
    ANSWERING_MODEL,
    ANTHROPIC_API_KEY,
    ARM_D_HOP_K,
    ARM_D_HUB_DEGREE,
    ARM_D_PER_ENTITY_K,
    ARM_D_SEED_K,
    EMBEDDING_MODEL_PRICE_PER_M,
    OPENAI_API_KEY,
    RUNS_DIR,
    SAMPLE_CONVERSATIONS,
)
from src.db import get_connection
from src.ingest.chunks import EMBED_BATCH_SIZE, embed_batch, normalize
from src.graph_db import get_driver
from src.dataset.sample import select_stratified_sample

ARM_D_DIR = RUNS_DIR / "arm_d"


def embed_text_for_graph_fact(subject: str, predicate: str, obj: str, fact: str) -> str:
    # Triple carries the canonical entity name, sentence carries the nuance.
    return f"{subject} {predicate} {obj}. {fact}"


def ingest_graph_fact_embeddings(conn, client, conversation_id: str) -> int:
    rows = conn.execute(
        """
        SELECT f.fact_key, f.subject, f.predicate, f.object, f.fact
        FROM graph_facts f
        LEFT JOIN graph_fact_embeddings e ON e.fact_key = f.fact_key
        WHERE f.conversation_id = %s AND e.fact_key IS NULL
        """,
        (conversation_id,),
    ).fetchall()

    total_tokens = 0
    for i in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = rows[i : i + EMBED_BATCH_SIZE]
        texts = [embed_text_for_graph_fact(s, p, o, f) for _k, s, p, o, f in batch]
        vectors, tokens = embed_batch(client, texts)
        total_tokens += tokens
        with conn.cursor() as cur:
            for (fact_key, *_rest), vector in zip(batch, vectors):
                cur.execute(
                    """
                    INSERT INTO graph_fact_embeddings (fact_key, conversation_id, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (fact_key) DO NOTHING
                    """,
                    (fact_key, conversation_id, vector_literal(normalize(vector))),
                )
        conn.commit()

    if rows:
        print(f"{conversation_id}: {len(rows)} new graph_fact embeddings")
    return total_tokens


def seed_by_vector(conn, conversation_id: str, query_vector, k: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT e.fact_key
        FROM graph_fact_embeddings e
        WHERE e.conversation_id = %s
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (conversation_id, vector_literal(query_vector), k),
    ).fetchall()
    return [r[0] for r in rows]


def entities_named_in_question(session, conversation_id: str, question: str) -> list[str]:
    """Entity nodes whose name actually occurs in the question text.

    Substring check, not fuzzy match: fuzzy alone pulled in entities sharing a
    token, whose facts then got forced into the prompt by per-entity coverage.
    """
    records = session.run(
        """
        MATCH (e:Entity {conversation_id: $conversation_id})
        WHERE e.type = 'Person'
        RETURN e.name AS name, e.name_norm AS name_norm
        """,
        conversation_id=conversation_id,
    ).data()
    question_lower = question.lower()
    return [r["name_norm"] for r in records if r["name_norm"] in question_lower]


def facts_for_entity(conn, conversation_id: str, entity_norm: str, query_vector, k: int) -> list[str]:
    """Top-k facts touching one entity, ranked against the question."""
    rows = conn.execute(
        """
        SELECT f.fact_key
        FROM graph_facts f JOIN graph_fact_embeddings e ON e.fact_key = f.fact_key
        WHERE f.conversation_id = %s
          AND (lower(btrim(f.subject)) = %s OR lower(btrim(f.object)) = %s)
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (conversation_id, entity_norm, entity_norm, vector_literal(query_vector), k),
    ).fetchall()
    return [r[0] for r in rows]


TRAVERSAL_QUERY = """
// Entry edges found by vector lookup, then one hop out from their SPECIFIC
// endpoints. hub_degree gates the star's centre: expanding a speaker node
// returns the whole conversation.
MATCH (a:Entity)-[seed:RELATES_TO]->(b:Entity)
WHERE seed.fact_key IN $seed_keys
UNWIND [a, b] AS endpoint
WITH DISTINCT endpoint
MATCH (endpoint)-[r:RELATES_TO]-(:Entity)
WITH endpoint, count(r) AS degree
WHERE degree <= $hub_degree
MATCH (endpoint)-[r:RELATES_TO]-(:Entity)
WHERE NOT r.fact_key IN $seed_keys
RETURN DISTINCT r.fact_key AS fact_key
LIMIT $hop_k
"""

EDGE_DETAIL_QUERY = """
MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
WHERE r.fact_key IN $fact_keys
RETURN r.fact_key AS fact_key, s.name AS subject, r.predicate AS predicate,
       o.name AS object, r.fact AS fact, r.source_turn_id AS source_turn_id,
       r.session_date AS session_date, r.valid_from AS valid_from,
       r.valid_to AS valid_to
"""


def retrieve(conn, session, conversation_id: str, question: str, query_vector):
    start = time.monotonic()

    seed_keys = seed_by_vector(conn, conversation_id, query_vector, ARM_D_SEED_K)

    covered_entities = entities_named_in_question(session, conversation_id, question)
    entity_keys = []
    for entity_norm in covered_entities:
        entity_keys.extend(
            facts_for_entity(conn, conversation_id, entity_norm, query_vector, ARM_D_PER_ENTITY_K)
        )

    hop_keys = [
        r["fact_key"]
        for r in session.run(
            TRAVERSAL_QUERY, seed_keys=seed_keys, hub_degree=ARM_D_HUB_DEGREE, hop_k=ARM_D_HOP_K
        )
    ]

    # Order matters: seeds, then entity coverage, then traversal. fromkeys
    # dedupes while preserving it.
    fact_keys = list(dict.fromkeys(seed_keys + entity_keys + hop_keys))
    edges = session.run(EDGE_DETAIL_QUERY, fact_keys=fact_keys).data()
    by_key = {e["fact_key"]: e for e in edges}
    ordered = [by_key[k] for k in fact_keys if k in by_key]

    latency_ms = (time.monotonic() - start) * 1000
    return ordered, latency_ms, {
        "seed": len(seed_keys),
        "entity_coverage": len(entity_keys),
        "traversal": len(hop_keys),
        "entities_covered": covered_entities,
    }


def build_context(edges) -> str:
    """Same labelling as Arm C, plus the interval a superseded fact held.

    Both halves of the bi-temporal pair are labelled explicitly: prefixing a bare
    date to a sentence that still says "yesterday" makes the model resolve it
    twice and answer a day early.
    """
    lines = []
    for e in edges:
        said = (e["session_date"] or "")[:10]
        valid_from = (e["valid_from"] or "")[:10]
        stamp = f"said {said}"
        if valid_from and valid_from != said:
            stamp += f"; occurred {valid_from}"
        if e["valid_to"]:
            stamp += f"; no longer true after {e['valid_to'][:10]}"
        lines.append(f"({stamp}) {e['fact']}")
    return "\n".join(lines)


def compute_recall(conversation_id: str, evidence_dia_ids: list[str], edges, hydrated_turn_ids=None):
    # hydrated_turn_ids, when given, is what the model actually saw -- recall
    # must be measured against that, not the un-widened edges.
    if not evidence_dia_ids:
        return None
    evidence_turn_ids = [f"{conversation_id}:{d}" for d in evidence_dia_ids]
    retrieved = set(hydrated_turn_ids) if hydrated_turn_ids is not None else {e["source_turn_id"] for e in edges}
    return sum(1 for t in evidence_turn_ids if t in retrieved) / len(evidence_turn_ids)


def fetch_hydrated_turns(conn, conversation_id: str, edges, window: int):
    """Widen the edges' source turns by +/- window and return them in
    chronological order, plus the widened turn-id set for recall.
    """
    source_turn_ids = {e["source_turn_id"] for e in edges}
    if not source_turn_ids:
        return [], set()

    # Resolve neighbours by (session_id, turn_index); turn_id is not arithmetic.
    session_rows = conn.execute(
        "SELECT turn_id, session_id, turn_index FROM raw_turns WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchall()
    index_by_turn = {turn_id: (session_id, idx) for turn_id, session_id, idx in session_rows}
    by_session_index = {(session_id, idx): turn_id for turn_id, session_id, idx in session_rows}

    widened_ids = set(source_turn_ids)
    for turn_id in source_turn_ids:
        session_id, center = index_by_turn[turn_id]
        for offset in range(-window, window + 1):
            neighbour = by_session_index.get((session_id, center + offset))
            if neighbour:
                widened_ids.add(neighbour)

    rows = conn.execute(
        """
        SELECT turn_id, session_date, speaker, text
        FROM raw_turns
        WHERE conversation_id = %s AND turn_id = ANY(%s)
        ORDER BY session_date, turn_index
        """,
        (conversation_id, list(widened_ids)),
    ).fetchall()
    return rows, widened_ids


def build_hydrated_context(edges, turn_rows) -> str:
    """Triples first, then the deduplicated raw turns they came from."""
    # maxsplit=1 keeps the full dia_id ("D3:11"); splitting further would
    # collapse turns from different sessions onto the same label.
    triple_lines = build_context(edges)
    turn_lines = [
        f"{turn_id.split(':', 1)[1]} {speaker}: {text}" for turn_id, _date, speaker, text in turn_rows
    ]
    return f"{triple_lines}\n\n--- Original turns these facts were drawn from ---\n" + "\n".join(turn_lines)


def run_question(conn, session, anthropic_client, openai_client, conversation_id, question, category, evidence, hydrate_window: int = 0):
    # Time retrieval and generation; Arms A-C only cached the retrieval half.
    total_start = time.monotonic()

    query_vector, embed_tokens = embed_query(openai_client, question)
    edges, retrieval_latency_ms, breakdown = retrieve(conn, session, conversation_id, question, query_vector)

    hydrated_turn_ids = None
    if hydrate_window:
        turn_rows, hydrated_turn_ids = fetch_hydrated_turns(conn, conversation_id, edges, hydrate_window)
        context = build_hydrated_context(edges, turn_rows)
    else:
        context = build_context(edges)

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
        "retrieved_fact_keys": [e["fact_key"] for e in edges],
        "retrieval_breakdown": breakdown,
        "system_answer": answer_text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "embed_tokens": embed_tokens,
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": (time.monotonic() - total_start) * 1000,
        "recall": compute_recall(conversation_id, evidence, edges, hydrated_turn_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="restrict to one conversation_id")
    parser.add_argument("--limit", type=int, help="max questions per conversation, for smoke testing")
    parser.add_argument("--dry-run", action="store_true", help="retrieve and print context, no answering call (free)")
    parser.add_argument(
        "--hydrate", type=int, default=0,
        help="raw-turn hydration ablation: widen each retrieved edge's source turn by +/- N turns "
             "and pass the deduplicated raw turns alongside the triples. Writes to a separate "
             "cache (runs/arm_d_hydrated_N/), never touching the frozen config's paid answers.",
    )
    args = parser.parse_args()
    answers_dir = ARM_D_DIR if not args.hydrate else RUNS_DIR / f"arm_d_hydrated_{args.hydrate}"

    conn = get_connection()
    driver = get_driver()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    anthropic_client = None if args.dry_run else Anthropic(api_key=ANTHROPIC_API_KEY)

    embed_tokens = 0
    for conversation_id in ([args.conversation] if args.conversation else SAMPLE_CONVERSATIONS):
        embed_tokens += ingest_graph_fact_embeddings(conn, openai_client, conversation_id)
    if embed_tokens:
        print(f"graph_fact embedding ingestion: {embed_tokens} tokens, ${(embed_tokens/1_000_000)*EMBEDDING_MODEL_PRICE_PER_M:.4f}")

    sample = select_stratified_sample(conn)
    if args.conversation:
        sample = {args.conversation: sample.get(args.conversation, [])}

    evidence_by_qa = {}
    for conversation_id in sample:
        evidence_by_qa.update(
            dict(
                conn.execute(
                    "SELECT qa_id, evidence FROM qa_pairs WHERE conversation_id = %s",
                    (conversation_id,),
                ).fetchall()
            )
        )

    with driver.session() as session:
        for conversation_id, questions in sample.items():
            if args.limit:
                questions = questions[: args.limit]

            if args.dry_run:
                for qa_id, question, _category in questions:
                    query_vector, _ = embed_query(openai_client, question)
                    edges, latency, breakdown = retrieve(conn, session, conversation_id, question, query_vector)
                    if args.hydrate:
                        turn_rows, hydrated_turn_ids = fetch_hydrated_turns(conn, conversation_id, edges, args.hydrate)
                        recall = compute_recall(conversation_id, evidence_by_qa.get(qa_id, []), edges, hydrated_turn_ids)
                        context = build_hydrated_context(edges, turn_rows)
                    else:
                        recall = compute_recall(conversation_id, evidence_by_qa.get(qa_id, []), edges)
                        context = build_context(edges)
                    print(f"\n=== {qa_id} ({latency:.1f}ms, {len(edges)} edges, recall={recall}) ===")
                    print(f"Q: {question}")
                    print(f"breakdown: {breakdown}")
                    print(context)
                continue

            cache = load_json_cache(answers_dir / f"{conversation_id}.json")
            new_calls = 0
            for qa_id, question, category in questions:
                if qa_id in cache:
                    continue
                cache[qa_id] = run_question(
                    conn, session, anthropic_client, openai_client, conversation_id,
                    question, category, evidence_by_qa.get(qa_id, []), args.hydrate,
                )
                new_calls += 1
                save_json_cache(answers_dir / f"{conversation_id}.json", cache)
            print(f"[arm_d hydrate={args.hydrate}] {conversation_id}: {len(cache)} answered ({new_calls} new this run)")

    driver.close()
    conn.close()


if __name__ == "__main__":
    main()
