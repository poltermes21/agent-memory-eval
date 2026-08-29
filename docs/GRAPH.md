# Why the knowledge graph lost

← [back to README](../README.md) · [back to results](RESULTS.md)

The project's own hypothesis was that a temporal knowledge graph would beat flat retrieval on
multi-hop questions, at a measurable cost in ingestion and latency — a Pareto trade, not a free
win. The cost showed up. The win did not.

Multi-hop accuracy sits at exactly **90.0%** and multi-hop recall at **0.570** for Arm C, for
shared-extraction Arm D, and for graph-native Arm D. Three different systems, identical
multi-hop performance.

Three measurements explain why, and they generalize past this dataset.

---

## 1. The graph is a star

Projected from 1,851 extracted facts: **1,343 entity nodes, 1,851 edges** — and the degree
distribution is brutally lopsided.

| degree | nodes |
|---|---|
| 1 | **1,056** |
| 2-5 | 249 |
| 6-20 | 27 |
| 21-40 | 1 |
| 40+ | **10** |

Those 10 high-degree nodes are the conversation participants, at degree 130-242 (Joanna 242,
Calvin 225, Nate 215, Evan 198…). Everything else is a leaf.

Traversing outward from a person node therefore returns essentially the whole conversation —
which is Arm A with extra steps, not a graph result. `ARM_D_HUB_DEGREE = 40` exists solely to
stop that, gating expansion to specific entities only.

There is no rich middle layer to traverse, because **two people chatting do not produce one**.
This is a property of conversational memory, not a flaw in the extraction: in a corpus where
almost every fact is *about one of two speakers*, the speakers are hubs and there is no
interesting path structure between the leaves.

## 2. The questions are not chain-shaped

Only **4.5%** of extracted facts are structurally chainable (the object of one fact is the
subject of another). And of the 10 multi-hop questions in the sample:

- **8 are aggregation over a single entity** — "what interests does X have?", answered by
  collecting many facts about one node.
- **2 are two-entity intersection** — "what do X and Y share?", answered by collecting facts
  about two nodes and comparing.

Neither shape needs A→B→C traversal. LoCoMo's "multi-hop" label describes **reasoning hops, not
graph hops** — a distinction worth checking before building a graph specifically to win that
category. Aggregation over one entity is exactly what a top-k vector search already does well.

## 3. Restatements crowd the retrieval budget

People repeat themselves across sessions, and the pipeline has no mechanism to collapse that.

The literal triple `(Nate, have, Nate's turtles)` appears **5 times** in one conversation, each
with a different `source_turn_id`; 40 of that conversation's 484 facts mention turtles. Because
`fact_key` hashes in the source turn, identical restatements from different sessions never
merge — correctly, since each is a genuine assertion made at a different time — and because
near-duplicates embed almost identically, they compete for the same retrieval slots.

Arm D is *more* exposed to this than Arm C, because its per-entity coverage step **guarantees**
slots to each entity the question names. Observed directly on a 0.0-recall question ("What
interests do Joanna and Nate share?"): 4 of the 13 retrieved edges were the same sentiment
about turtles, from the same session.

`own`/`have` is deliberately excluded from the invalidation list — owning several things at
once is accumulation, not replacement — so nothing prunes these either. The design anticipated
duplicate *ingestion* (`ON CONFLICT DO NOTHING`, `MERGE` on a stable key) and contradicting
*values* (bi-temporal invalidation), but not honest repetition of the same fact over time.

---

## The extraction rebuild

Arm D originally read Arm C's flat `facts` table **unmodified**, so any accuracy difference
between them would be attributable to retrieval architecture alone. That controlled run shipped
first, and it tied Arm C on all 10 multi-hop questions — 9 of 10 verdicts identical.

Inspecting why put the blame on the flat triples themselves, not on retrieval:

- **compound subjects** — "Jon and Gina" extracted as one node instead of two, so neither
  person's node is reachable from that fact.
- **possessive phrases** — "Melanie's kids" stored as an unconnected string, never reachable
  from the `Melanie` node, fragmenting one real entity across several node-strings.
- **irregular verbs** — predicate normalization was done in code by suffix-stripping, which
  cannot handle `has`/`had`/`have`: they share no common suffix, so one lemma produced three
  different stems. No regex fix generalizes here; lemmatization is a language-understanding
  task.

### What was changed, and why

Fixing those requires extraction that thinks in nodes and relations from the start — which is a
*different task* from Arm C's flat triples, not a superset of it. So Arm D got its own
extraction pipeline (`src/extract_graph_facts.py` → `graph_facts` table), with three changes:

| change | why |
|---|---|
| **typed entities** from a closed enum (`Person, Place, Organization, Object, Event, Concept`) | replaces a code-side "is this one of the two speakers?" heuristic that labelled everything else `Unknown`. Entity type is a structural property of language, so the enum was decided upfront rather than measured per dataset |
| **LLM-lemmatized predicates** at extraction time (`live_in`, not `lives_in`/`lived_in`) | the irregular-verb bug above. An LLM understands `has`/`had`/`have` are one lemma; a stemmer cannot |
| **canonical naming for multi-mention entities** — "her kids", "the kids", "Melanie's kids" all extracted as one consistent string | so possessive references merge into one node instead of fragmenting. Verified empirically before scaling: two independent extraction calls for different sessions produced the identical string `"Melanie's kids"` |

### It worked structurally

- predicate vocabulary shrank **794 → 575** distinct predicates
- a real **sub-hub layer** appeared between the speaker hubs and the degree-1 leaves —
  `Nate's turtles` (Object, degree 25), `Jon's dance studio` (Organization, 19),
  `Melanie's kids` (Person, 17), `Dave's car` (Object, 12) — exactly the possessive references
  that used to fragment into unreachable strings
- **bi-temporal invalidation fired on real data for the first time** (3 cases; previously 0,
  because the old stemmer produced `liv_in` while the invalidation list contained `live_in`, so
  the lookup silently never matched)

### And the results got worse

| | accuracy | recall | multi-hop |
|---|---|---|---|
| Arm D, shared extraction | 85.0% | 0.740 | 90.0% / 0.570 |
| Arm D, own extraction | **82.5%** | **0.714** | 90.0% / 0.570 |

Better-structured data, worse retrieval, unchanged multi-hop. The plausible mechanism is
finding #3 above: **consistent entity naming pulls restatements closer together in embedding
space**, so they compete harder for the same guaranteed slots. The fix for fragmentation made
crowding worse, and crowding was the binding constraint.

This is worth stating plainly because it is the kind of result that usually goes unreported:
the intervention was principled, it did what it was designed to do at the data level, and it
still did not improve the metric it was aimed at.

### The methodological cost

Because Arm C and Arm D now run independent extraction pipelines, **a C-vs-D difference no
longer isolates retrieval architecture** — it reflects the two systems overall, extraction
quality included. This was a deliberate trade, taken with the reasoning that the project
compares complete memory systems rather than one retrieval mechanism tested on identical data.

What was kept to limit the damage: both arms still start from the same `raw_turns` table, both
are scored by the same frozen judge, and the original controlled shared-extraction run is
reported as its own row rather than being replaced by the newer one.

---

## What would actually test the graph

The bi-temporal machinery — validity intervals, single-valued-predicate invalidation, edges
labelled with `valid_to` rather than deleted — fires **3 times** on LoCoMo, because LoCoMo has
no knowledge-update category. That machinery is the graph's one genuine structural advantage
over flat retrieval: answering "where does X live *as of* date D", where a flat top-k has no
way to prefer the currently-valid fact over a superseded one.

**LongMemEval** has that category, plus `has_answer` turn flags for cleaner recall
measurement. Running it is the obvious next step and would be the fairest test the graph has
yet had. Until then, the honest reading of this project's result is *"the graph did not earn
its cost on conversational recall"*, not *"temporal graphs don't work"*.

---

← [back to README](../README.md) · [back to results](RESULTS.md) ·
[Design decisions](DESIGN.md) · [Every run](ALL-RUNS.md)
