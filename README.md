# Agent Memory Benchmark

An independent, reproducible comparison of **agent memory architectures** on LoCoMo:
full context, RAG over raw chunks, RAG over LLM-distilled facts, and RAG over a temporal
knowledge graph — measured on accuracy per question category, retrieval recall, tokens per
query, ingestion cost, and retrieval latency.

Not a memory library. The deliverable is a comparison you can act on.

> **Headline:** RAG over LLM-distilled facts is the strongest architecture measured here —
> **90.0% accuracy at 1,040 tokens per query**, and it sits on the cost/accuracy frontier at
> every budget, not just at its best point. The temporal knowledge graph never earned its cost.
> And *which* architecture appears to win changes depending on how you score it.
>
> → **[Full results and verdict](docs/RESULTS.md)**

---

## The arms

**B, C and D are all RAG.** They differ in *what goes into the index*, not in how the search
works — B and C use the exact same retrieval code path. That is deliberate: it makes the
B-vs-C comparison a single-variable experiment.

| arm | what gets indexed | how it's retrieved |
|---|---|---|
| **A** — full context | nothing; the whole transcript goes in the prompt | — |
| **B** — RAG over chunks | raw text chunks (3 granularities: turn / window / session) | exact cosine top-k |
| **C** — RAG over facts | LLM-distilled fact sentences | exact cosine top-k — *identical to B* |
| **D** — RAG over a graph | a fact graph: typed entity nodes, temporal edges | cosine seed → per-entity coverage → 1-hop Cypher traversal |

- **B → C** isolates exactly one variable: **raw chunk vs distilled fact**. Same retrieval,
  same answering model, same prompt, same questions. This is the cleanest comparison here, and
  it maps to a real production fork: store the conversation, or store what was learned from it.
- **C → D** adds graph structure on top — but it is *not* single-variable: D changes the
  stored unit, the retrieval step, and (after a mid-project rebuild) the extraction pipeline
  too. That confound is deliberate and is [documented](docs/GRAPH.md), not glossed over.

All arms read from one shared `raw_turns` table — no arm parses the source JSON its own way —
and every answer is scored by the same frozen judge.

```
locomo10.json
  └─> raw_turns ────┬─> (no ingestion) ─────────────────────> Arm A
                    ├─> chunk + embed ──────────────────────> Arm B
                    ├─> LLM extraction (flat triples)
                    │     └─> facts + fact_embeddings ──────> Arm C
                    └─> LLM extraction (graph-native)
                          └─> graph_facts ─> Neo4j ─────────> Arm D
```

---

## Reproducing

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d          # Postgres 16 + pgvector, Neo4j 5 Community
pip install -r requirements.txt

python -m src.fetch_locomo    # downloads data/raw/locomo10.json (CC BY-NC, not vendored)
python -m src.load_locomo     # -> raw_turns, qa_pairs

python -m src.arm_a                                       # full context
python -m src.embed && python -m src.arm_b                # RAG over chunks, 3 granularities
python -m src.extract_facts && python -m src.arm_c        # RAG over distilled facts
python -m src.extract_graph_facts                         # Arm D's own extraction
python -m src.build_graph --reset                         # graph_facts -> Neo4j ($0, no LLM)
python -m src.arm_d                                       # RAG over the graph
python -m src.arm_d --hydrate 2                           # raw-turn hydration ablation

python -m src.judge --arm arm_a                           # both rubrics, per arm
python -m src.judge --arm arm_b --granularity window --top-k 20
python -m src.judge --arm arm_c --top-k 10
python -m src.judge --arm arm_d --hydrate 2
```

Useful extras:

```bash
python -m src.recall_sweep --hydrate 2   # retrieval quality, no answering/judging call ($0)
python -m src.report_results             # regenerates every table in docs/ from runs/
python -m src.report_cost                # Arm A token/cost breakdown, per conversation
python -m pytest tests/ -q               # graph invalidation + embedding-norm tests
```

### Cost discipline

Every runner **caches to disk per question and resumes**. A killed run re-pays for nothing.
Caches are written atomically — a truncated JSON file once blocked a resume and forced a
re-pay, hence `src/cache_io.py`.

`src/recall_sweep.py` measures retrieval quality with **no answering or judging call at all**,
which is how every rejected retrieval strategy was tested for free before spending anything.

Total API spend for every number reported here: **~$10.09** ($6.14 answering, $3.95 judging),
plus ~$5 of extraction across both pipelines.

### Frozen decisions

Changing any of these invalidates every earlier number, so they are fixed from Arm A onward:
answering model (`claude-sonnet-5`), judge model (`claude-opus-5`), both judge prompts,
embedding model (`text-embedding-3-small`), and the sample — 5 conversations, 10 questions per
category, 40 questions per row, reused unchanged across every arm.

---

## Stack

Python + **Postgres 16 with pgvector** (one instance, one database, one shared schema for all
arms) + **Neo4j 5 Community**, both via Docker Compose. API models for extraction, answering
and judging; **no GPU required**.

Vector search is an **exact sequential scan — no ANN index anywhere** (no HNSW, no IVFFlat).
The corpus is a few thousand rows; approximate search would add nondeterminism for no
measurable speedup, and would let a reader object that the graph only won because flat
retrieval was silently dropping documents.

Runs inside WSL2 with the repo on the Linux filesystem — crossing to `/mnt/c/...` is 5-10x
slower on the many-small-files I/O this does constantly. Dates are stored as ISO-8601 text.
Embeddings are normalised on write regardless of what the provider returns, with a test
asserting unit norm, so a silent provider change fails a test instead of quietly drifting the
accuracy numbers.

---

## Layout

```
src/
  fetch_locomo.py  load_locomo.py     dataset download + raw_turns / qa_pairs
  run_sample.py                       the frozen stratified 40-question sample
  arm_a.py                            full context
  embed.py         arm_b.py           chunking + embeddings, RAG over chunks
  extract_facts.py arm_c.py           flat triple extraction, RAG over facts
  extract_graph_facts.py              Arm D's own graph-native extraction
  build_graph.py   graph_db.py arm_d.py   graph_facts -> Neo4j, RAG over the graph
  judge.py                            both rubrics, cached per question
  recall_sweep.py                     free retrieval-quality experiments
  report_results.py                   regenerates the tables in docs/ from runs/
  report_cost.py   cache_io.py  config.py  db.py
db/schema.sql                         every table
tests/                                graph invalidation, embedding unit norm
runs/                                 cached answers + verdicts (gitignored)
```

## Documentation

| document | contents |
|---|---|
| **[docs/RESULTS.md](docs/RESULTS.md)** | The verdict, the numbers, the three findings, and how the choice of judge reorders the leaderboard |
| [docs/GRAPH.md](docs/GRAPH.md) | Why the knowledge graph lost, and why rebuilding its extraction made it worse |
| [docs/DESIGN.md](docs/DESIGN.md) | Design decisions, rejected retrieval strategies, metric definitions |
| [docs/ALL-RUNS.md](docs/ALL-RUNS.md) | Every configuration measured, with full per-category numbers and costs |

Design rules and frozen decisions live in [docs/DESIGN.md](docs/DESIGN.md); `docs/` is the
current record for anything the code and a document disagree on.

---

## Dataset

**LoCoMo** — [`snap-research/locomo`](https://github.com/snap-research/locomo),
`data/locomo10.json`. 10 conversations, ~1,540 questions, categories: multi-hop, temporal,
open-domain, single-hop, plus an adversarial category that is loaded losslessly but excluded
from evaluation, as is standard. Licensed **CC BY-NC**; fetched at setup, not vendored.

This project used 5 of the 10 conversations (2,494 turns, 122 sessions) and 40 stratified
questions per configuration.
