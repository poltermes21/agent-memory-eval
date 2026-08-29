# Agent Memory Benchmark

> **This file is the working spec: design rules, frozen decisions and pitfalls to respect
> when changing the code. It is NOT the results write-up.** Where a hypothesis below has been
> tested, the measured outcome is noted inline and **[docs/RESULTS.md](docs/RESULTS.md) wins
> over anything stated here as an expectation.** Read the results before acting on a stated
> goal — some were disproven.

## What this project is

An independent, reproducible benchmark comparing **agent memory architectures** on public
long-term-conversational-memory datasets.

This is an **evaluation project**, not a memory product. The deliverable is a defensible
claim backed by numbers, not a library.

### The original claims, and what happened to them

**Judge claim** — the LLM-as-judge used by standard LoCoMo evaluation over-credits answers
that are close in topic but wrong in detail (dates, missing facts, unsupported additions).
**Supported, with a caveat**: over 640 judged answers the lenient and strict rubrics disagree
39 times, 39-0 in the direction of the lenient one passing what the strict one fails, and the
gap is large enough to reorder the arms. ~~Validated against human labels~~ — the
human-labeling step was dropped (see "Judge" below), so this shows the *gap between two
rubrics*, not which rubric is correct in absolute terms. Do not overclaim it.

**Graph claim** — a temporal knowledge graph beats flat retrieval on **multi-hop** and
**knowledge-update** questions, at a measurable cost in ingestion tokens and latency.
**DISPROVEN for multi-hop on this dataset, and untested for knowledge-update.** The graph
never beat flat retrieval at matched budget across two complete rebuilds; multi-hop accuracy
and recall are *identical* for Arm C and both Arm D builds. Knowledge-update could not be
tested at all — LoCoMo has no such category, and invalidation fires 3 times in the whole
corpus. See [docs/GRAPH.md](docs/GRAPH.md). **Do not treat "make the graph win" as a goal**;
if the graph is revisited, it is to test it on a dataset that can actually exercise it.

The result shape remains a Pareto frontier, not a single winner.

## Non-goals

- Not trying to beat a mature memory product. Those have years of work behind them.
- Not inventing a novel temporal model. We deliberately use a conventional bi-temporal
  model so differences are attributable to retrieval, not schema.
- Not a single global accuracy number. Per-category only.

## Architecture

Raw turns are the single shared source of truth for every arm. Extraction downstream of
that is NOT uniformly shared — Arm C and Arm D deliberately run separate extraction
pipelines, revised 2026-08-18 (see below).

```
locomo10.json
  -> raw_turns table (single source of truth for all arms)
  -> LLM extraction (flat triples) -> facts        -> Postgres (Arm C: relational)
  -> LLM extraction (graph-native) -> graph_facts   -> Neo4j    (Arm D: temporal KG)
  -> chunking + embeddings  (Arm B: vector RAG)
  -> no ingestion           (Arm A: full context)
```

### Why Arm C and Arm D have separate extraction (revised 2026-08-18)

Originally Arm D was built on Arm C's flat `facts` table, unmodified, so any accuracy
difference between them would be attributable only to retrieval architecture. That
controlled run shipped first and its numbers stand (see project memory
`project-arm-d-graph-findings`) — at matched retrieval budget the two tied on every
multi-hop question, and inspection showed the flat triples were themselves the bottleneck:
compound subjects ("Jon and Gina" as one node instead of two), possessive phrases
("Melanie's kids" as an unconnected string instead of its own node reachable from
`Melanie`), and code-side predicate normalization that cannot handle irregular verbs
(`has`/`had`/`have` do not share a suffix, so suffix-stripping produced three different
stems for one lemma).

Fixing those requires extraction that thinks in nodes and relations from the start —
entity typing, canonical multi-mention entities, LLM-lemmatized predicates — which is a
different task from Arm C's flat triples, not a superset of it. Forcing both arms through
one extraction step would mean picking a lowest common denominator that shortchanges
whichever arm's storage model it doesn't fit. So: **this project now compares two complete,
independently-extracted memory systems, not one retrieval mechanism tested on identical
data.** That is a real methodological trade — a difference between Arm C and D can no
longer be attributed to retrieval alone, extraction quality is now part of what is being
compared.

Both arms still start from `raw_turns` and both are still evaluated by the same frozen
judge — that half of the controlled comparison stands.

### What a difference between arms means

Same answering model, same judge throughout. Arms A/B/C/D differ in ingestion and retrieval
by design, so:

- **B vs C** is a single-variable comparison — identical retrieval, only the indexed unit
  differs (raw chunk vs distilled fact). A difference here is attributable to that one
  choice.
- **C vs D** is not. As of 2026-08-18 Arm D's extraction is also its own (see above), so a
  C-vs-D difference reflects the two systems overall, not retrieval alone.

**Critical**: every arm must be this project's own implementation. Building an arm on a
third-party memory system would make it a measurement of that system, not of the
architecture, and would not be comparable with the other arms.

**Graph isolation**: if a second graph system is ever added, it gets its own Neo4j
container, not a second database on this instance. Neo4j Community supports only one user
database per instance — multiple databases is an Enterprise feature — so separate instances
is the only available option anyway. It is also cleaner: another system writes its own
labels and indexes, and sharing an instance means a full-text or vector lookup can reach its
nodes if a filter slips. In a benchmark, being unable to say with certainty which system
wrote a given edge is disqualifying.

## Arms

### Arm A — full context
No ingestion, no retrieval. All sessions go into the prompt. Intended as the accuracy
ceiling; **measured, it is not** — Arm C beats it (90.0% vs 85.0%) on 1/25th the tokens.
Still the efficiency floor, and still the reference point.

### Arm B — vector RAG
Chunk turns, embed, top-k by similarity. No entities, no structure.
Design variable: **chunk granularity**. Three were measured, not two — per-turn,
per-session, and a sliding window. Window wins; per-turn is catastrophic (30.0%).

### Arm C — relational
Facts in one table. Structured but flat — no traversal.
**Retrieval is vector top-k, not the full-text/text-to-SQL originally specified**:
Postgres `ts_rank` has no IDF weighting, so a speaker name outranked the informative
terms and Arm C would have lost on ranking weakness rather than architecture. Keeping
retrieval identical to Arm B is also what makes B-vs-C single-variable. Do not "restore"
full-text here without changing Arm B too.

### Arm D — temporal KG
Facts as nodes and edges in Neo4j with validity intervals. Retrieval is hybrid: vector
lookup for entry edges, per-entity coverage, then **1 hop** of Cypher traversal (not 2 —
the graph is a star, so a second hop returns the whole conversation).

Arm D's extraction is graph-native (revised 2026-08-18, see "Why Arm C and Arm D have
separate extraction" above) — it is no longer required to carry the exact same information
as Arm C's flat facts, since the two now run independent extraction pipelines by design.

## Schemas

### Shared: raw turns
`turn_id`, `session_id`, `session_date`, `speaker`, `text`

Single source of truth. Every arm starts from exactly this. Never let an arm parse the
source JSON its own way.

### Arm C: facts table
`subject`, `predicate`, `object`, `fact`, `source_turn_id`, `session_date`,
`valid_from`, `valid_to`, `ingested_at`

### Arm B: vector collection
Vector + payload: `text`, `turn_id`, `turn_ids`, `session_id`, `session_date`, `speaker`,
`granularity`

### Arm D: graph_facts table (Postgres) + Neo4j

`graph_facts` (Postgres, Arm D's own extraction output, separate from Arm C's `facts` —
see "Why Arm C and Arm D have separate extraction"): `subject`, `subject_type`, `predicate`,
`object`, `object_type`, `fact`, `source_turn_id`, `session_date`, `valid_from`, `valid_to`,
`ingested_at`. `subject_type`/`object_type` come from a small closed enum decided upfront —
`Person, Place, Organization, Object, Event, Concept` — chosen once, not per-dataset: almost
anything mentioned in conversation falls into one of these structurally, so it does not need
re-measuring the way relation vocabulary did (below). `predicate` is LLM-lemmatized at
extraction time (base/infinitive form, e.g. `live_in` not `lives_in`/`lived_in`) — measured
reason: `has`/`had`/`have` share no suffix, so code-side stemming produced three stems for
one lemma, and no regex fix generalizes to irregular verbs. Multi-mention entities
(possessive/common-noun references — "Melanie's kids", "her kids", "the kids" all meaning
the same referent) are extracted under one canonical phrasing consistently, the same
per-session pronoun-to-name resolution already applied, extended to common nouns, so they
merge into one node instead of fragmenting.

Neo4j, projected from `graph_facts` at $0 (no second extraction to build the graph):
- Nodes: `Entity {name, name_norm, type}`, `Session {id, date}`
- Edges: `RELATES_TO {fact_key, fact, predicate, predicate_norm, conversation_id,
  source_turn_id, source_session, session_date, valid_from, valid_to, invalidated_by,
  ingested_at}`

Use a **generic edge type with `predicate` as a property**, not the predicate as the edge
type. Cypher cannot parameterise relationship types, and measured on this data 794 distinct
predicates exist with no small subset covering most facts (top predicate "feels" covers only
3.8%; 451 of 794 needed for 80% coverage) — there is no small closed vocabulary to promote to
edge types. Generic type keeps traversal and invalidation logic writable in one query, and it
is the conventional choice for property graphs with an open relation vocabulary.

Temporal invalidation uses a closed list of single-valued predicates decided upfront, in
**lemma form to match the extraction** (`live_in`, `work_at`, `marry`, ... — NOT `lives_in`
/ `works_at`, which match nothing and silently disable invalidation, as an earlier version
did; and not `own`, since owning several things at once is accumulation, not replacement) rather than an LLM-emitted "supersedes" signal — it works for $0 on
already-extracted facts, whereas the signal approach would need a schema field added to
extraction and repaying to backfill it. Invalidated edges are labelled with `valid_to`, never
deleted or hidden from retrieval.

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

The **same frozen judge** runs across all four arms. That is what makes the comparison valid.

The rubric must demand the specific detail, not topical closeness. Expected "in March 2023",
answer "a couple of years ago" = FAIL.

**Validation, decided 2026-08-29**: report agreement between our (strict) judge and the
original/lenient judge over the same answer set, per category, as the project's core
contribution. ~~Hand-labeling 50-100 pairs against a third, human-labeled ground truth~~ was
dropped — not pursued. The two-judge disagreement rate, backed by concrete borderline
examples (e.g. "in March 2023" vs "a couple of years ago"), is the evidence; do not add a
human-labeling step back in without a new decision.

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

Result shape is a trade, never a single score — *"X gains N points on this category, at
Y times the tokens and +Zms latency"*, not *"my graph scores 72%."* As measured, the trade
went against the graph: no accuracy gain at 4.4x the tokens and 16x the latency.

## Frozen decisions

Freeze these from Arm A onward. Changing them invalidates every earlier number.

- Answering model
- Judge model and BOTH judge prompts
- Embedding model
- The question sample (`SAMPLE_CONVERSATIONS`, `SAMPLE_PER_CATEGORY`) — re-picking it per
  arm silently makes the arms incomparable

Cache ingestion output to disk per arm. QA gets re-run dozens of times; never pay for
re-extraction.

## Datasets

- **LoCoMo** (primary): `github.com/snap-research/locomo`, file `data/locomo10.json`.
  10 conversations, ~1540 questions, categories: single-hop, multi-hop, temporal,
  open-domain. Adversarial category is typically excluded. CC BY-NC.
- ~~**LoCoMo-Refined**: recalibrated QA set with a stricter judge.~~ Not used.
- **LongMemEval** (secondary): HuggingFace `xiaowu0162/longmemeval-cleaned`. 500 questions,
  six categories including knowledge update. Has `has_answer` turn flags -> needed for
  retrieval recall.

Read an existing evaluation harness before writing one; the runner is not the contribution.

## Order of work

Build the measurement spine first. Neo4j comes last.

1. **Loader + Arm A.** `locomo10.json` -> `raw_turns` -> full-context QA -> first
   end-to-end number. A pipeline that measures something trivial beats a beautiful graph
   with no figures.
2. **The judge.** Implement both the original and the strict judge, run them over the same
   Arm A answers, report agreement between the two. The central finding lands here, before a
   single line of Cypher.
3. **Arm B** (vector), all three chunk granularities.
4. **Arm C** (relational).
5. **Arm D** (temporal KG). By now there are three reference numbers and a clear hypothesis
   about which categories must be won.
6. ~~**External references**: self-hosted third-party memory systems as calibration.~~
   **Decided out of scope, 2026-08-27.** Not pursued. Arms A-D plus the judge work are the
   project's deliverable; the external comparison and the second Neo4j container it required
   are dropped. The "Graph isolation" / "Critical" rules above still stand — they are about
   never letting another system's data contaminate an arm's results, which applies whether or
   not such a comparison is ever run.
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
- **Raw-turn hydration** (Arm D variant, measured at +/- 2 turns): retrieve edges, widen
  to their source turns +/- N, pass triples *and* deduplicated raw turns to the LLM.
  Outcome: **recall up a lot** (0.714 -> 0.823), **accuracy mixed** — open-domain 60 -> 90%,
  but multi-hop and temporal both fell 90 -> 80%. So it is a per-category decision, and it
  confirmed that what helps comes from retrieving better turns, not from structure.
- **Graph contamination**: any second graph system gets its own Neo4j *container*, never a
  shared instance.

## Stack

Python, running inside **WSL2**. The repo lives on the Linux filesystem
(`~/projects/...`), never on `/mnt/c/...` — crossing the Windows/Linux filesystem boundary
is 5-10x slower on the many-small-files I/O this project does constantly.

- **Postgres 16 + pgvector** (`pgvector/pgvector:pg16`) via Docker Compose — raw turns
  table, `facts` (Arm C), `graph_facts` (Arm D, separate table and separate extraction, see
  "Why Arm C and Arm D have separate extraction") and embeddings (Arm B). Persistent volume
  so ingestion survives restarts. **One single Postgres instance, one database, shared
  schema** for Arms A-D. Do NOT give each arm its own database. The isolation rule above
  applies only to the boundary between our code and any third-party system; our own arms are
  *required* to share the `raw_turns` table, since a single source of truth is what makes the
  comparison controlled — that requirement is unchanged; only the extraction step downstream
  of it now forks per arm.
- **Arm B vector search is exact**: sequential scan, **NO ANN index**. Do not create an
  HNSW or IVFFlat index. The corpus is a few thousand chunks; approximate search would add
  nondeterminism to results with no measurable speed gain, and would let a reader object
  that a comparison against flat retrieval was unfair because it was dropping documents.
  State this explicitly in the README.
- **Neo4j 5 Community**, one container for Arm D on 7474/7687. Cap heap at ~512MB
  (`NEO4J_server_memory_heap_max__size`) — the graphs are thousands of nodes, not millions.
  Any second graph system gets its own container and its own volume (see Graph isolation
  above), never a second database on this one.
- Prefer self-hosted over cloud for anything compared here: a cloud black box can change
  version mid-experiment, which silently invalidates every number taken before the change.
- API models for extraction, answering and judging. **No GPU required.**
- Credentials via `.env`, with a versioned `.env.example` and `.env` in `.gitignore`.

Dates are stored as ISO-8601 text. Embeddings are normalised on write regardless of what
the provider returns, with a test asserting unit norm — if a provider silently changes
behaviour, the test fails instead of the accuracy numbers drifting mysteriously.

`docker-compose.yml` is part of the deliverable: reproducibility is what this project
sells. "Install Neo4j Desktop and configure it by hand" is not a reproducible benchmark.

## README discipline

The README is judged on the claim, not the code volume. "I integrated four memory systems,
here are the numbers" is a tutorial. "Distilled facts beat chunk retrieval at every budget,
the temporal graph never earned its cost, and which one appears to win depends on how you
score it" is a contribution.

State what the reader should do differently, and be explicit about which differences are
large enough to act on and which are inside the noise of a 40-question sample.