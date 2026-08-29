"""Project graph_facts into Neo4j for Arm D. Costs $0: no LLM, free to re-run.

  (Entity {name, name_norm, type}) -[RELATES_TO {...}]-> (Entity)
  (Session {id, date})

Two invariants, easy to break by "improving" them (docs/DESIGN.md): one generic
edge type with `predicate` as a property, and lowercase+trim normalization only,
never verb stemming.
"""
import argparse
import re
from collections import defaultdict

from src.config import SAMPLE_CONVERSATIONS
from src.db import get_connection
from src.graph_db import get_driver


def normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def normalize_predicate(predicate: str) -> str:
    return normalize_entity(predicate)


# Single-valued relations only: a later, different object REPLACES the earlier
# one. Everything else accumulates and must never auto-invalidate -- "own" is
# absent deliberately (a person owns several things at once).
# tests/test_build_graph.py asserts every entry survives normalize_predicate().
INVALIDATING_PREDICATES = frozenset({
    "live_in", "live_at", "live_near", "move_to", "reside_in", "relocate_to",
    "be_located_in", "be_located",
    "work_at", "work_in", "work_for", "work_as", "employed_by", "have_job", "current_job",
    "study_at", "attend_school_at",
    "marry", "get_married", "married_to", "in_relationship_with", "date", "engaged_to",
})

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.conversation_id, e.name_norm) IS UNIQUE",
    "CREATE CONSTRAINT session_key IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
]
# Entry-point lookup for retrieval: find entities by name, then traverse.
INDEXES = [
    "CREATE FULLTEXT INDEX entity_name_fts IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
]


def fetch_facts(conn, conversation_id: str):
    return conn.execute(
        """
        SELECT fact_key, subject, subject_type, predicate, object, object_type,
               fact, source_turn_id, session_date, valid_from
        FROM graph_facts
        WHERE conversation_id = %s
        ORDER BY session_date, source_turn_id
        """,
        (conversation_id,),
    ).fetchall()


def compute_invalidations(facts) -> dict[str, tuple[str, str]]:
    """{fact_key: (valid_to, invalidated_by_fact_key)} for superseded facts.

    Superseded means: a later fact shares the subject and normalised predicate,
    that predicate is single-valued, and the object differs. Edges are labelled,
    never deleted -- retrieval still returns them with their interval.
    """
    groups = defaultdict(list)
    for fact_key, subject, _subj_type, predicate, obj, _obj_type, _fact, _turn, session_date, _valid_from in facts:
        pred_norm = normalize_predicate(predicate)
        if pred_norm not in INVALIDATING_PREDICATES:
            continue
        groups[(normalize_entity(subject), pred_norm)].append(
            (session_date, fact_key, normalize_entity(obj))
        )

    invalidations = {}
    for entries in groups.values():
        entries.sort(key=lambda e: e[0])
        for i, (_date, fact_key, obj_norm) in enumerate(entries):
            successor = next(
                (e for e in entries[i + 1 :] if e[2] != obj_norm), None
            )
            if successor:
                invalidations[fact_key] = (successor[0], successor[1])
    return invalidations


MERGE_QUERY = """
MERGE (s:Entity {conversation_id: $conversation_id, name_norm: $subject_norm})
  ON CREATE SET s.name = $subject, s.type = $subject_type
MERGE (o:Entity {conversation_id: $conversation_id, name_norm: $object_norm})
  ON CREATE SET o.name = $object, o.type = $object_type
MERGE (sess:Session {id: $session_id})
  ON CREATE SET sess.date = $session_date
MERGE (s)-[r:RELATES_TO {fact_key: $fact_key}]->(o)
  SET r.predicate = $predicate,
      r.predicate_norm = $predicate_norm,
      r.fact = $fact,
      r.conversation_id = $conversation_id,
      r.source_turn_id = $source_turn_id,
      r.source_session = $session_id,
      r.session_date = $session_date,
      r.valid_from = $valid_from,
      r.valid_to = $valid_to,
      r.invalidated_by = $invalidated_by,
      r.ingested_at = datetime()
MERGE (s)-[:MENTIONED_IN]->(sess)
"""


def build_conversation(conn, session, conversation_id: str) -> tuple[int, int]:
    facts = fetch_facts(conn, conversation_id)
    invalidations = compute_invalidations(facts)

    session_id_by_turn = dict(
        conn.execute(
            "SELECT turn_id, session_id FROM raw_turns WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchall()
    )

    for fact_key, subject, subject_type, predicate, obj, object_type, fact, source_turn_id, session_date, valid_from in facts:
        subject_norm = normalize_entity(subject)
        object_norm = normalize_entity(obj)
        valid_to, invalidated_by = invalidations.get(fact_key, (None, None))
        session.run(
            MERGE_QUERY,
            conversation_id=conversation_id,
            fact_key=fact_key,
            subject=subject,
            subject_norm=subject_norm,
            subject_type=subject_type,
            object=obj,
            object_norm=object_norm,
            object_type=object_type,
            predicate=predicate,
            predicate_norm=normalize_predicate(predicate),
            fact=fact,
            source_turn_id=source_turn_id,
            session_id=session_id_by_turn.get(source_turn_id, f"{conversation_id}:unknown"),
            session_date=session_date,
            valid_from=valid_from,
            valid_to=valid_to,
            invalidated_by=invalidated_by,
        )
    return len(facts), len(invalidations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation", help="restrict to one conversation_id; default: SAMPLE_CONVERSATIONS")
    parser.add_argument("--reset", action="store_true", help="delete the whole Arm D graph first")
    args = parser.parse_args()

    conn = get_connection()
    driver = get_driver()

    with driver.session() as session:
        if args.reset:
            session.run("MATCH (n) DETACH DELETE n")
            print("graph reset")
        for statement in CONSTRAINTS + INDEXES:
            session.run(statement)

        total_facts = total_invalidated = 0
        for conversation_id in [args.conversation] if args.conversation else SAMPLE_CONVERSATIONS:
            n_facts, n_invalidated = build_conversation(conn, session, conversation_id)
            total_facts += n_facts
            total_invalidated += n_invalidated
            print(f"{conversation_id}: {n_facts} edges, {n_invalidated} invalidated")

        counts = session.run(
            "MATCH (e:Entity) WITH count(e) AS entities "
            "MATCH ()-[r:RELATES_TO]->() RETURN entities, count(r) AS edges"
        ).single()
        print(f"\ntotal: {counts['entities']} entity nodes, {counts['edges']} edges, {total_invalidated} invalidated")

    driver.close()
    conn.close()


if __name__ == "__main__":
    main()
