# Agent Memory Benchmark

## What this project is

An independent, reproducible benchmark comparing **agent memory architectures** on public
long-term-conversational-memory datasets.

This is an **evaluation project**, not a memory product. The deliverable is a defensible
claim backed by numbers, not a library.

### The central claim we are testing

The LLM-as-judge used by the standard LoCoMo evaluation over-credits answers that are close
in topic but wrong in detail (dates, missing facts, unsupported additions). Published
comparisons between memory systems rest on that judge. We re-evaluate with a judge validated
against human labels and report how conclusions change.

Secondary claim: a temporal knowledge graph beats flat retrieval specifically on
**multi-hop** and **knowledge-update** questions, at a measurable cost in ingestion tokens
and retrieval latency. It is a Pareto frontier, not a single winner.

## Non-goals

- Not trying to beat Graphiti or Mem0. They have years of work behind them.
- Not inventing a novel temporal model. We deliberately mirror Graphiti's bi-temporal
  model so differences are attributable to retrieval, not schema.
- Not a single global accuracy number. Per-category only.

## Architecture

One shared extraction step feeds four internally-comparable arms. Two external systems run
separately as calibration.

```
locomo10.json
  -> raw_turns table (single source of truth for all arms)
  -> LLM extraction, ONCE -> facts (subject, predicate, object, fact, turn_id, dates)
        |-> Postgres        (Arm C: relational)
        |-> Neo4j           (Arm D: temporal KG)
  -> chunking + embeddings  (Arm B: vector RAG)
  -> no ingestion           (Arm A: full context)

separately, own extraction, treated as black boxes:
  -> Mem0 self-hosted       (external reference)
  -> Graphiti self-hosted   (external reference)
```

### Why extraction is shared

Arms C and D consume the *same* extracted facts. If each arm ran its own LLM extraction,
measured differences would be extraction noise, not architecture. With shared extraction the
only variables are **how facts are stored and how they are retrieved** — which is the
research question.

### Two axes, two kinds of conclusion

- **Arms A-D**: controlled experiment. Same extraction, same answering model, same judge.
  Differences are causally attributable to architecture.
- **Mem0 / Graphiti**: complete systems, many variables at once. They tell us only whether
  our implementations are in a sane range. Quality control, not rivals.

**Critical**: Arm D must be our own implementation. Do NOT build Arm D on Graphiti — that
would make our row and its row the same system and void the comparison.

**Graph isolation**: Arm D and Graphiti run on **two separate Neo4j containers**, not two
databases on one instance. Neo4j Community supports only one user database per instance —
multiple databases is an Enterprise feature — so separate instances is the only available
option anyway. It is also cleaner: Graphiti writes its own labels and indexes, and sharing an
instance means a full-text or vector lookup can reach the other system's nodes if a filter
slips. In a benchmark, being unable to say with certainty which system wrote a given edge is
disqualifying.

## Arms

### Arm A — full context
No ingestion, no retrieval. All sessions go into the prompt. Accuracy ceiling, efficiency
floor. Reference point.

### Arm B — vector RAG
Chunk turns, embed, top-k by similarity. No entities, no structure.
Design variable to measure both ways: **chunk granularity (per-turn vs per-session)**.

### Arm C — relational
Facts in one table. Retrieval by full-text search or text-to-SQL. Structured but flat —
no traversal.

### Arm D — temporal KG
Facts as nodes and edges in Neo4j with validity intervals. Retrieval is hybrid: vector or
full-text lookup to find entry nodes, then Cypher traversal (1-2 hops) with temporal
filtering.

Arms C and D carry **equivalent information** by design. If the relational arm held less
data, Arm D would win by construction and prove nothing.

## Schemas

### Shared: raw turns
`turn_id`, `session_id`, `session_date`, `speaker`, `text`

Single source of truth. Every arm starts from exactly this. Never let an arm parse the
source JSON its own way.

### Arm C: facts table
`subject`, `predicate`, `object`, `fact`, `source_turn_id`, `session_date`,
`valid_from`, `valid_to`, `ingested_at`

### Arm B: vector collection
Vector + payload: `text`, `turn_id`, `session_id`, `timestamp`, `speaker`

### Arm D: Neo4j
- Nodes: `Entity {name, type}`, `Session {id, date}`
- Edges: `RELATES_TO {fact, predicate, valid_from, valid_to, ingested_at, source_session, invalidated_by}`

Use a **generic edge type with `predicate` as a property**, not the predicate as the edge
type. Cypher cannot parameterise relationship types, and an LLM extracting from open
conversation produces hundreds of near-synonymous predicates. Generic type keeps traversal
and invalidation logic writable in one query. This is also what Graphiti does.

No reification needed — property graphs allow properties on relationships directly.

### Store both the triple and the sentence

The triple is for **finding**: search by entity, traverse relations, filter by predicate,
detect contradiction (same subject + predicate, different object -> invalidate the earlier
edge). The `fact` sentence is for **answering**: it preserves hedging, conditionals,
causality and tone that the triple discards.

Example: "I think I'll quit my job if they don't raise my salary" becomes one edge
`(user) -[predicate: considering_quitting, fact: <full sentence>]-> (job)`. Do not model the
condition as a node — hypothetical events are not entities, and a node nothing traverses is
just noise.

## Judge

The judge is a **stateless function**, not an agent. No tools, no memory. Input: question,
expected answer from the dataset, system answer. Output: boolean. Not a 0-10 score, not
semantic similarity.

The **same frozen judge** runs across all six rows. That is what makes the comparison valid.

The rubric must demand the specific detail, not topical closeness. Expected "in March 2023",
answer "a couple of years ago" = FAIL.

**Validation is mandatory**: hand-label 50-100 answer pairs and report agreement between
our judge, the original LoCoMo judge, and the human labels. This validation is the project's
core contribution — not the arms.

## Metrics (all five, per arm, per category)

1. **Accuracy by question category** — from the judge. Never a global number: the global
   number hides exactly what we are trying to show.
2. **Retrieval recall** — did the system retrieve the turn containing the answer?
   Independent of the judge and of the answering LLM. This is what separates "good memory"
   from "good answering model". LongMemEval marks which turns contain the answer.
3. **Tokens per query** — context sent to the answering model. Best accuracy at 26k tokens
   per question is not production-viable.
4. **Ingestion cost** — tokens and wall-clock. Arm D is the most expensive here by far.
5. **Retrieval latency p95** — Arm B is one hop; Arm D is lookup plus traversal.

Result shape: *"the temporal KG gains N points on knowledge-update and multi-hop, at 4x
ingestion cost and +80ms latency."* Not *"my graph scores 72%."*

## Frozen decisions

Freeze these from Arm A onward. Changing them invalidates every earlier number.

- Answering model
- Judge model and judge prompt
- Embedding model

Cache ingestion output to disk per arm. QA gets re-run dozens of times; never pay for
re-extraction.

## Datasets

- **LoCoMo** (primary): `github.com/snap-research/locomo`, file `data/locomo10.json`.
  10 conversations, ~1540 questions, categories: single-hop, multi-hop, temporal,
  open-domain. Adversarial category is typically excluded. CC BY-NC.
- **LoCoMo-Refined**: recalibrated QA set with a stricter judge. Central to the judge
  validation work.
- **LongMemEval** (secondary): HuggingFace `xiaowu0162/longmemeval-cleaned`. 500 questions,
  six categories including knowledge update. Has `has_answer` turn flags -> needed for
  retrieval recall.
- `mem0ai/memory-benchmarks` — reference for evaluation protocol. Read it; do not reinvent
  the runner.

## Order of work

Build the measurement spine first. Neo4j comes last.

1. **Loader + Arm A.** `locomo10.json` -> `raw_turns` -> full-context QA -> first
   end-to-end number. A pipeline that measures something trivial beats a beautiful graph
   with no figures.
2. **The judge.** Implement both the original and the strict judge, run them over the same
   Arm A answers, hand-label 50-100 pairs, report agreement. The central finding lands here,
   before a single line of Cypher.
3. **Arm B** (vector), both chunk granularities.
4. **Arm C** (relational).
5. **Arm D** (temporal KG). By now there are three reference numbers and a clear hypothesis
   about which categories must be won.
6. **External references**: Mem0 and Graphiti self-hosted.
7. **Ablation**: Arm D with and without raw-turn hydration (see below). Report both rows.

## Known pitfalls

- **Entity resolution** is failure source #1 for multi-hop. "my sister", "Marta" and "she"
  must resolve to one node; fragment the entity across three nodes and traversal finds no
  path, and Arm D loses to flat RAG. Start with simple normalisation (lowercase, explicit
  aliases), measure, add embedding-based disambiguation only if it hurts.
- **Idempotency**: re-running ingestion must not duplicate. `MERGE` on a stable key —
  hash of (normalised subject, predicate, object, turn_id).
- **One-to-many extraction**: one turn yields N facts, all sharing `source_turn_id` and
  `session_date`. Extract one fact per assertion, not per message.
- **Relative dates**: "last month" must be resolved against the session date at extraction
  time, not left as text.
- **Raw-turn hydration** (Arm D variant): retrieve edges, collect distinct
  `source_turn_id`s, fetch the original turns (optionally +/- 1 turn for pronoun
  antecedents), pass triples *and* deduplicated raw turns to the LLM. Better accuracy but
  inflates tokens-per-query. Treat as an experimental variant, not an article of faith —
  it may reveal that most of the graph's advantage comes from retrieving better turns
  rather than from structure. That is a finding, not a failure.
- **Graph contamination**: separate Neo4j *container* for Graphiti, never a shared instance.

## Stack

Python, running inside **WSL2**. The repo lives on the Linux filesystem
(`~/projects/...`), never on `/mnt/c/...` — crossing the Windows/Linux filesystem boundary
is 5-10x slower on the many-small-files I/O this project does constantly.

- **Postgres 16 + pgvector** (`pgvector/pgvector:pg16`) via Docker Compose — raw turns
  table, facts table (Arm C) and embeddings (Arm B). Persistent volume so ingestion
  survives restarts. **One single Postgres instance, one database, shared schema** for Arms
  A-D. Do NOT give each arm its own database. The isolation rule above applies only to the
  boundary between our code and third-party systems (Graphiti, Mem0); our own arms are
  *required* to share the `raw_turns` table, since a single source of truth is what makes
  the comparison controlled. Arms C and D also share the same extracted facts by design.
- **Arm B vector search is exact**: sequential scan, **NO ANN index**. Do not create an
  HNSW or IVFFlat index. The corpus is a few thousand chunks; approximate search would add
  nondeterminism to results with no measurable speed gain, and would let a reader object
  that the KG only won because flat retrieval was dropping documents. State this explicitly
  in the README.
- **Neo4j 5 Community**, two containers (see Graph isolation above): Arm D on 7474/7687,
  Graphiti on 7475/7688, separate volumes. Cap heap at ~512MB each
  (`NEO4J_server_memory_heap_max__size`) — the graphs are thousands of nodes, not millions.
  Graphiti's container only needs to be up for step 6.
- **Mem0 self-hosted** via Docker Compose (step 6 only). Apache 2.0, fully self-hostable.
- For Zep use **Graphiti self-hosted**, not Zep Cloud — the Community Edition of Zep is
  discontinued, and a cloud black box can change version mid-experiment.
- API models for extraction, answering and judging. **No GPU required.**
- Credentials via `.env`, with a versioned `.env.example` and `.env` in `.gitignore`.

Dates are stored as ISO-8601 text. Embeddings are normalised on write regardless of what
the provider returns, with a test asserting unit norm — if a provider silently changes
behaviour, the test fails instead of the accuracy numbers drifting mysteriously.

`docker-compose.yml` is part of the deliverable: reproducibility is what this project
sells. "Install Neo4j Desktop and configure it by hand" is not a reproducible benchmark.

## README discipline

The README is judged on the claim, not the code volume. "I integrated Mem0 and Zep, here are
the numbers" is a tutorial. "The field's standard judge agrees with humans 44% of the time,
so I re-evaluated with a validated judge, and here is which published conclusions change" is
a contribution.