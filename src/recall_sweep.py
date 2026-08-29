"""Free retrieval-quality sweep: compare strategies by recall alone.

Recall needs no answering model and no judge, so a strategy costs ~$0.00002 to
evaluate here instead of ~$0.35 to answer and judge. Try things here first.

Modifies no arm: anything that wins is reported as an additional row, never as a
replacement for a configuration already paid for. Results: docs/DESIGN.md.
"""
import argparse
import statistics
from collections import defaultdict

from openai import OpenAI

from src.arm_b import embed_query, vector_literal
from src.config import ARM_C_TOP_K, OPENAI_API_KEY
from src.db import get_connection
from src.run_sample import select_stratified_sample

CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def speakers(conn, conversation_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT speaker FROM raw_turns WHERE conversation_id = %s", (conversation_id,)
    ).fetchall()
    return [r[0] for r in rows]


def vector_top_k(conn, conversation_id: str, query_vector, k: int) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT f.fact_key, f.source_turn_id
        FROM fact_embeddings e JOIN facts f ON f.fact_key = e.fact_key
        WHERE e.conversation_id = %s
        ORDER BY e.embedding <=> %s::vector ASC
        LIMIT %s
        """,
        (conversation_id, vector_literal(query_vector), k),
    ).fetchall()
    return rows


def fulltext_top_k(conn, conversation_id: str, question: str, k: int) -> list[tuple[str, str]]:
    """OR-semantics full-text over the fact sentences.

    plainto_tsquery is unusable here (AND semantics gives 0 rows), so tokens are
    OR-ed.
    """
    rows = conn.execute(
        """
        WITH q AS (SELECT websearch_to_tsquery('english', %s) AS tsq)
        SELECT f.fact_key, f.source_turn_id
        FROM facts f, q
        WHERE f.conversation_id = %s
          AND to_tsvector('english', f.fact) @@ q.tsq
        ORDER BY ts_rank(to_tsvector('english', f.fact), q.tsq) DESC
        LIMIT %s
        """,
        (question.replace(" ", " or "), conversation_id, k),
    ).fetchall()
    return rows


def reciprocal_rank_fusion(ranked_lists, k: int, rrf_k: int = 60):
    """Standard RRF: score = sum over lists of 1/(rrf_k + rank).

    Rank-based, not score-based: cosine distance and ts_rank are not on a
    comparable scale.
    """
    scores = defaultdict(float)
    turn_by_key = {}
    for ranked in ranked_lists:
        for rank, (fact_key, source_turn_id) in enumerate(ranked, start=1):
            scores[fact_key] += 1.0 / (rrf_k + rank)
            turn_by_key[fact_key] = source_turn_id
    best = sorted(scores, key=lambda key: -scores[key])[:k]
    return [(key, turn_by_key[key]) for key in best]


# --- strategies: each returns [(fact_key, source_turn_id), ...] -----------------

def strategy_vector(conn, oc, conversation_id, question, qv, k):
    """Arm C's mechanism, the reference line."""
    return vector_top_k(conn, conversation_id, qv, k)


def strategy_entity_expansion(conn, oc, conversation_id, question, qv, k):
    """One query per speaker the question names, fused.

    A single embedding over a two-person question blends both and matches facts
    about the relationship rather than about each person.
    """
    named = [s for s in speakers(conn, conversation_id) if s.lower() in question.lower()]
    if len(named) < 2:
        return vector_top_k(conn, conversation_id, qv, k)

    per_entity = max(1, k // len(named))
    ranked_lists = []
    for speaker in named:
        sub_vector, _ = embed_query(oc, f"{speaker}: {question}")
        ranked_lists.append(vector_top_k(conn, conversation_id, sub_vector, per_entity * 2))
    ranked_lists.append(vector_top_k(conn, conversation_id, qv, per_entity))
    return reciprocal_rank_fusion(ranked_lists, k)


def strategy_hybrid_rrf(conn, oc, conversation_id, question, qv, k):
    """Vector + full-text, fused by reciprocal rank."""
    return reciprocal_rank_fusion(
        [
            vector_top_k(conn, conversation_id, qv, k * 2),
            fulltext_top_k(conn, conversation_id, question, k * 2),
        ],
        k,
    )


def strategy_hybrid_plus_entity(conn, oc, conversation_id, question, qv, k):
    """Both of the above fused together."""
    named = [s for s in speakers(conn, conversation_id) if s.lower() in question.lower()]
    ranked_lists = [
        vector_top_k(conn, conversation_id, qv, k * 2),
        fulltext_top_k(conn, conversation_id, question, k * 2),
    ]
    for speaker in named:
        sub_vector, _ = embed_query(oc, f"{speaker}: {question}")
        ranked_lists.append(vector_top_k(conn, conversation_id, sub_vector, k))
    return reciprocal_rank_fusion(ranked_lists, k)


STRATEGIES = {
    "vector (Arm C)": strategy_vector,
    "entity expansion": strategy_entity_expansion,
    "hybrid RRF": strategy_hybrid_rrf,
    "hybrid + entity": strategy_hybrid_plus_entity,
}


def recall(conversation_id: str, evidence_dia_ids, retrieved, hydrate_window: int = 0, turn_index=None):
    """Fraction of the evidence turns covered by what retrieval returned.

    hydrate_window > 0 widens retrieved turns by +/- N neighbours.
    """
    if not evidence_dia_ids:
        return None
    evidence = [f"{conversation_id}:{d}" for d in evidence_dia_ids]
    covered = {turn_id for _key, turn_id in retrieved}
    if hydrate_window and turn_index:
        widened = set(covered)
        for turn_id in covered:
            session_id, idx = turn_index.get(turn_id, (None, None))
            if session_id is None:
                continue
            for offset in range(-hydrate_window, hydrate_window + 1):
                neighbour = turn_index.get((session_id, idx + offset))
                if neighbour:
                    widened.add(neighbour)
        covered = widened
    return sum(1 for t in evidence if t in covered) / len(evidence)


def build_turn_index(conn, conversation_id: str):
    """Both directions in one dict, so hydration can step to neighbours."""
    rows = conn.execute(
        "SELECT turn_id, session_id, turn_index FROM raw_turns WHERE conversation_id = %s",
        (conversation_id,),
    ).fetchall()
    index = {}
    for turn_id, session_id, turn_idx in rows:
        index[turn_id] = (session_id, turn_idx)
        index[(session_id, turn_idx)] = turn_id
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=ARM_C_TOP_K)
    parser.add_argument("--hydrate", type=int, default=0, help="widen retrieved turns by +/- N (raw-turn hydration variant)")
    args = parser.parse_args()

    conn = get_connection()
    oc = OpenAI(api_key=OPENAI_API_KEY)
    sample = select_stratified_sample(conn)

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
    turn_indexes = {c: build_turn_index(conn, c) for c in sample}

    # One embedding per question, reused by every strategy: the sweep must not be
    # more expensive just because it compares more strategies.
    query_vectors = {
        qa_id: embed_query(oc, question)[0]
        for _c, questions in sample.items()
        for qa_id, question, _cat in questions
    }

    print(f"top_k={args.top_k}, hydrate=+/-{args.hydrate} turns\n")
    header = f"{'strategy':20s}" + "".join(f"{n:>13s}" for n in CATEGORY_NAMES.values()) + f"{'ALL':>8s}"
    print(header)
    print("-" * len(header))

    for name, strategy in STRATEGIES.items():
        by_category = defaultdict(list)
        for conversation_id, questions in sample.items():
            for qa_id, question, category in questions:
                retrieved = strategy(
                    conn, oc, conversation_id, question, query_vectors[qa_id], args.top_k
                )
                value = recall(
                    conversation_id, evidence_by_qa.get(qa_id, []), retrieved,
                    args.hydrate, turn_indexes[conversation_id],
                )
                if value is not None:
                    by_category[category].append(value)
        row = f"{name:20s}"
        for category in CATEGORY_NAMES:
            values = by_category[category]
            row += f"{statistics.mean(values):13.3f}" if values else f"{'-':>13s}"
        allv = [v for values in by_category.values() for v in values]
        row += f"{statistics.mean(allv):8.3f}"
        print(row)

    conn.close()


if __name__ == "__main__":
    main()
