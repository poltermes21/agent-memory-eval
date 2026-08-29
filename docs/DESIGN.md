# Design decisions

← [back to README](../README.md) · [back to results](RESULTS.md)

Each decision here was made for a measured reason, not a stylistic one. Where a decision
deviates from what was originally planned, the deviation and its evidence are recorded rather
than quietly applied.

---

## Retrieval

### Vector search is exact — no ANN index

No HNSW, no IVFFlat, anywhere in the project. The corpus is a few thousand rows; approximate
search would add nondeterminism to results for no measurable speedup, and it would let a reader
object that the graph only won because flat retrieval was silently dropping documents. When the
conclusion is "flat retrieval is enough", flat retrieval has to be beyond suspicion.

### Arm C uses vector retrieval, not full-text or text-to-SQL

A deliberate deviation from the spec, which called for full-text search or text-to-SQL.

Measured reason: Postgres `ts_rank` has **no IDF weighting**, so a common speaker name outranks
the rare informative terms — "calvin" is the subject of 209 of that conversation's 372 facts.
Arm C would have lost on ranking-implementation weakness rather than on architecture, which
would have made the comparison meaningless.

Keeping retrieval **identical to Arm B** instead isolates the one variable that matters in
production: **distilled facts vs raw chunks** — store the conversation, or store what was
learned from it. See [the arms table](../README.md#the-arms).

### Arm D retrieval is three parts, each earning its place

1. **Seed (vector)** — top-k facts by cosine, deliberately identical to Arm C's mechanism, so
   whatever D adds on top is the only difference between the two arms.
2. **Entity coverage (full-text)** — for each entity the question names, force in its own top
   facts. This targets the one place flat retrieval is structurally weak: a single ranked top-k
   can spend all its slots on the person with more facts, which visiting each named entity
   explicitly cannot.
3. **Traversal (Cypher, 1 hop)** — from the seeds' *specific* endpoints only, gated by
   `ARM_D_HUB_DEGREE` because [the graph is a star](GRAPH.md#1-the-graph-is-a-star).

The budget (`ARM_D_SEED_K=8`, `ARM_D_PER_ENTITY_K=3`, `ARM_D_HOP_K=5`) was chosen by a free
dry-run sweep to land near Arm C's fixed k=10, so the C-vs-D comparison is about structure and
not about who put more text in the prompt.

---

## Storage

### Both the triple and the sentence are stored

The **triple** is for *finding*: search by entity, traverse relations, filter by predicate,
detect contradiction (same subject + predicate, different object).

The **sentence** is for *answering*: it preserves hedging, conditionals, causality and tone
that the triple discards. "I think I'll quit my job if they don't raise my salary" stays as one
edge with the full sentence attached — the condition is deliberately *not* modelled as a
separate node, because hypothetical events are not entities and a node nothing traverses is
just noise.

### One generic edge type, `predicate` as a property

Not the predicate as the relationship type. Measured: **794 distinct predicates over 1,718
facts**, the most common ("feels") covering only 3.8%, and 451 needed for 80% coverage. There
is no small closed vocabulary to promote to relationship types — and Cypher cannot parameterise
a relationship type anyway, so a generic `RELATES_TO` keeps traversal and invalidation
writable as one query.

### Relative dates are resolved at extraction time, but only when unambiguous

"Yesterday" or "last Friday" resolves against the session date into `valid_from`. "Last month",
"recently", "a while back" resolve to **null** — they describe a range, not a day, and turning
them into a specific date invents precision the speaker never gave. The phrase survives in the
fact sentence, which is enough for the answering model.

A related bug worth recording: prefixing a bare resolved date to a sentence that still contains
its relative wording ("...yesterday") made the model resolve the reference a *second* time and
answer a day early. Both halves of the bi-temporal pair are now labelled explicitly instead.

---

## Temporal invalidation

### A closed list of single-valued predicates, not an LLM "supersedes" signal

`live_in`, `work_at`, `married`… — relations where a later, different object genuinely
*replaces* the earlier one. Chosen structurally (a person has one home, one employer, one
spouse at a time), and it works for **$0** on already-extracted facts, whereas an
extraction-time signal would have required a new schema field and re-paying to backfill it.

**`own` is deliberately excluded.** Owning several things at once is accumulation, not
replacement. Including it would have wrongly superseded one sample question's expected answer,
which needs *both* of Evan's cars.

Measured justification for having a list at all: applying the naive rule (same subject +
predicate, different object → invalidate) to **all** predicates flagged **780 of 1,798 facts**,
almost all non-contradictions — "Joanna feels X" across 12 different feelings is a history of
states, not a correction.

### Invalidated edges are labelled, never deleted or hidden

They carry `valid_to` and `invalidated_by` and still appear in retrieval. That is the point of
a bi-temporal model, and it is load-bearing here: one sample question ("what things did Evan
have broken") expects both the old and the new Prius, so an arm that dropped superseded values
at retrieval time would answer it wrong.

This path fires only 3 times on LoCoMo, so it is covered by unit tests
(`tests/test_graph.py`) rather than being exercised by the run itself.

---

## Rejected retrieval strategies

Retrieval recall needs **no answering model and no judge** — it only asks whether the retrieved
set covers the dataset's evidence turns. So alternatives were swept in `src/evaluate/recall_sweep.py`
for roughly $0.00002 each, and only a strategy that actually moved recall would have been worth
paying to answer and judge.

Against the Arm C baseline at k=10:

| strategy | recall (all) | multi-hop |
|---|---|---|
| **vector over fact sentences** (baseline) | 0.755 | 0.570 |
| BM25 only (hand-rolled, proper IDF + length norm + stopwords) | 0.604 | 0.240 |
| RRF fusion, vector + BM25 | 0.738 | 0.520 |
| RRF fusion with a deeper 30-candidate pool | 0.667 | 0.380 |
| Postgres `ts_rank` fused with vector | 0.716 | — |
| per-entity query expansion (one embedding per named speaker, fused) | 0.754 | — |
| raising k instead (k=30) | 0.791 | 0.670 |
| **raw-turn hydration ±2** | **0.854** | **0.764** |

### Why hybrid lexical search loses here

Do not re-litigate without new evidence. Three structural reasons, not tuning ones:

1. The documents are **one-assertion sentences** (~10-15 words), so term frequency is almost
   always 1 — BM25's saturation and length normalization have nothing to work with.
2. LoCoMo questions are **paraphrases** with near-zero lexical overlap with the facts that
   answer them.
3. The proper nouns that would carry lexical signal have **very low IDF**, because a speaker's
   name is the subject of most facts in their own conversation.

Hybrid search wins on corpora with jargon, identifiers, or error strings. Everyday
conversational English is the opposite case.

### Also rejected without spending

**GraphRAG-style community detection** — the graphs are thousands of nodes with a star
topology, and the query pattern is targeted lookup, not global summarization. Community
detection answers "what are the themes of this corpus"; LoCoMo asks "what did X say about Y".

---

## Metric definitions

All five are reported **per arm and per category**. Never as a single global accuracy number —
the global number hides exactly what this project is trying to show.

| metric | definition |
|---|---|
| **Accuracy by category** | Fraction of questions the judge marks `PASS`, per LoCoMo category. Both rubrics reported. |
| **Retrieval recall** | Fraction of the dataset's evidence turns covered by what retrieval returned. **Independent of the judge and of the answering model** — this is what separates "good memory" from "good answering model". |
| **Tokens per query** | Context sent to the answering model. Best accuracy at 26k tokens per question is not production-viable. |
| **Ingestion cost** | LLM tokens and wall-clock to build the memory. Arm C's extraction cost $2.47; Arm D pays a second, comparable pass. Arms A and B pay effectively nothing (embeddings are ~$0.0002 for the full corpus). |
| **Retrieval latency p95** | 4-8ms for Arms B/C (one indexed scan); 52-137ms for Arm D (vector seed + per-entity queries + Cypher traversal). |

---

## Schema notes

One Postgres instance, one database, **one shared schema for all arms**. The isolation rule in
the spec applies only to the boundary between this code and third-party systems — the arms
themselves are *required* to share `raw_turns`, since a single source of truth is what makes
the comparison controlled.

A bug worth recording, found after it had already silently scrambled turn order in Arm A's
transcript and Arm B's chunk text: **never sort turns by `turn_id`**. It embeds the position
as text, so a string `ORDER BY` gives `D1:1, D1:10, D1:11, …, D1:19, D1:2, D1:20` — wrong for
any session with 10+ turns. Always `ORDER BY session_id, turn_index`.

Full reasoning for every table and column is in [`db/schema.sql`](../db/schema.sql), written as
comments next to what they justify.

---

← [back to README](../README.md) · [back to results](RESULTS.md) ·
[Why the graph lost](GRAPH.md) · [Every run](ALL-RUNS.md)
