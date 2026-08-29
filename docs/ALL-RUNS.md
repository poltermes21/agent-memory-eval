# Every run measured

← [back to README](../README.md) · [back to results](RESULTS.md)

Complete numbers for all 16 configurations. [RESULTS.md](RESULTS.md) shows the subset that
carries a finding; this is the raw material behind it.

**Regenerate any table here** — free, reads only the caches in `runs/`:

```bash
python -m src.evaluate.report_results               # all tables
python -m src.evaluate.report_results --section cost
```

Every row is the same 40 questions (10 per category, 5 conversations), the same answering model
(`claude-sonnet-5`), the same prompt, and the same two frozen judge rubrics
(`claude-opus-5`). **One question = 2.5 percentage points.**

---

## Accuracy, recall and cost per question

`accuracy` is the strict judge, `lenient` the LoCoMo-style one. `tok/q` is context sent to the
answering model.

| configuration | multi-hop | temporal | open-dom. | single-hop | accuracy | lenient | recall | tok/q | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| **A full context** | 90.0% | 80.0% | 70.0% | 100.0% | **85.0%** | 85.0% | - | 25,576 | - |
| **B turn k=5** | 10.0% | 30.0% | 30.0% | 50.0% | **30.0%** | 42.5% | 0.165 | 699 | 8 |
| **B turn k=15** | 20.0% | 70.0% | 50.0% | 50.0% | **47.5%** | 60.0% | 0.290 | 1,485 | 12 |
| **B turn k=30** | 30.0% | 60.0% | 50.0% | 60.0% | **50.0%** | 55.0% | 0.367 | 2,652 | 7 |
| **B window k=5** | 50.0% | 80.0% | 70.0% | 90.0% | **72.5%** | 77.5% | 0.635 | 1,656 | 7 |
| **B window k=10** | 60.0% | 90.0% | 80.0% | 100.0% | **82.5%** | 85.0% | 0.806 | 3,077 | 7 |
| **B window k=20** | 70.0% | 90.0% | 80.0% | 100.0% | **85.0%** | 90.0% | 0.878 | 5,882 | 6 |
| **B session k=5** | 70.0% | 70.0% | 70.0% | 80.0% | **72.5%** | 72.5% | 0.626 | 5,109 | 4 |
| **B session k=8** | 70.0% | 80.0% | 80.0% | 80.0% | **77.5%** | 82.5% | 0.714 | 8,162 | 5 |
| **B session k=12** | 80.0% | 80.0% | 80.0% | 100.0% | **85.0%** | 87.5% | 0.869 | 12,319 | 4 |
| **C facts k=5** | 80.0% | 90.0% | 70.0% | 80.0% | **80.0%** | 90.0% | 0.664 | 489 | 6 |
| **C facts k=10** | 90.0% | 90.0% | 70.0% | 100.0% | **87.5%** | 97.5% | 0.755 | 669 | 7 |
| **C facts k=20** | 90.0% | 90.0% | 80.0% | 100.0% | **90.0%** | 92.5% | 0.777 | 1,040 | 7 |
| **D shared extraction** | 90.0% | 70.0% | 80.0% | 100.0% | **85.0%** | 90.0% | 0.740 | 682 | 52 |
| **D own extraction** | 90.0% | 90.0% | 60.0% | 90.0% | **82.5%** | 95.0% | 0.714 | 728 | 137 |
| **D + hydration** | 80.0% | 80.0% | 90.0% | 100.0% | **87.5%** | 95.0% | 0.823 | 3,205 | 112 |

Arm A has no recall or latency figure: it has no retrieval step to measure.

### The k-sweeps in isolation

Each arm's own accuracy/cost curve, which is what a practitioner tuning that arm actually
needs:

| Arm B, turn chunks | k=5 | k=15 | k=30 |
|---|---|---|---|
| accuracy | 30.0% | 47.5% | 50.0% |
| tok/q | 699 | 1,485 | 2,652 |

| Arm B, window chunks | k=5 | k=10 | k=20 |
|---|---|---|---|
| accuracy | 72.5% | 82.5% | 85.0% |
| tok/q | 1,656 | 3,077 | 5,882 |

| Arm B, session chunks | k=5 | k=8 | k=12 |
|---|---|---|---|
| accuracy | 72.5% | 77.5% | 85.0% |
| tok/q | 5,109 | 8,162 | 12,319 |

| Arm C, distilled facts | k=5 | k=10 | k=20 |
|---|---|---|---|
| accuracy | 80.0% | 87.5% | 90.0% |
| tok/q | 489 | 669 | 1,040 |

Two things stand out. **Per-turn chunking never recovers** — 6x the k buys 20 points and still
lands at 50.0%, worse than Arm C at 1/5th the tokens. And **Arm C's whole curve fits under the
cheapest Arm B configuration**: at k=20 it uses fewer tokens (1,040) than window chunks at k=5
(1,656), while scoring 17.5 points higher.

---

## Retrieval recall by category

Recall is independent of the judge and of the answering model — it only asks whether retrieval
surfaced the turns the dataset marks as containing the answer.

| configuration | multi-hop | temporal | open-dom. | single-hop | all |
|---|---|---|---|---|---|
| **B turn k=5** | 0.099 | 0.300 | 0.012 | 0.250 | **0.165** |
| **B turn k=15** | 0.149 | 0.500 | 0.162 | 0.350 | **0.290** |
| **B turn k=30** | 0.149 | 0.600 | 0.268 | 0.450 | **0.367** |
| **B window k=5** | 0.623 | 0.700 | 0.418 | 0.800 | **0.635** |
| **B window k=10** | 0.743 | 0.900 | 0.579 | 1.000 | **0.806** |
| **B window k=20** | 0.843 | 1.000 | 0.671 | 1.000 | **0.878** |
| **B session k=5** | 0.583 | 0.800 | 0.322 | 0.800 | **0.626** |
| **B session k=8** | 0.683 | 0.900 | 0.475 | 0.800 | **0.714** |
| **B session k=12** | 0.893 | 0.900 | 0.784 | 0.900 | **0.869** |
| **C facts k=5** | 0.500 | 0.800 | 0.506 | 0.850 | **0.664** |
| **C facts k=10** | 0.570 | 0.900 | 0.602 | 0.950 | **0.755** |
| **C facts k=20** | 0.620 | 0.900 | 0.639 | 0.950 | **0.777** |
| **D shared extraction** | 0.570 | 0.900 | 0.540 | 0.950 | **0.740** |
| **D own extraction** | 0.570 | 0.800 | 0.537 | 0.950 | **0.714** |
| **D + hydration** | 0.800 | 0.900 | 0.592 | 1.000 | **0.823** |

**Recall and accuracy come apart.** Arm B at window k=20 has the second-highest recall in the
table (0.878) but scores 85.0% — below Arm C k=20's 90.0% at recall 0.777. Retrieving the right
turn is necessary, not sufficient: the model still has to find the answer inside whatever was
retrieved, and 5,882 tokens of raw conversation is a harder place to find it than 1,040 tokens
of distilled facts.

**Multi-hop recall is where every flat-retrieval configuration struggles** and where hydration
is the only thing that helped: 0.570 for all three fact-based configurations, 0.800 with
hydration.

---

## Judge agreement

Strict vs lenient rubric, over every judged answer in every configuration.

| category | agreement | n |
|---|---|---|
| multi-hop | 91.9% | 160 |
| temporal | 98.8% | 160 |
| open-domain | 90.0% | 160 |
| single-hop | 95.0% | 160 |
| **all** | **93.9%** | **640** |

Disagreement direction, pooled across all 640:

| direction | count |
|---|---|
| lenient PASS, strict FAIL | **39** |
| lenient FAIL, strict PASS | **0** |

See [RESULTS.md](RESULTS.md#how-you-score-changes-who-wins) for what this does to the ranking,
and for worked examples of the disagreements.

---

## API spend

| configuration | answering | judging | total |
|---|---|---|---|
| A full context | $2.0699 | $0.2634 | $2.3333 |
| B turn k=5 | $0.0638 | $0.2428 | $0.3066 |
| B turn k=15 | $0.1318 | $0.2436 | $0.3754 |
| B turn k=30 | $0.2255 | $0.2366 | $0.4621 |
| B window k=5 | $0.1487 | $0.2194 | $0.3681 |
| B window k=10 | $0.2640 | $0.2461 | $0.5101 |
| B window k=20 | $0.4879 | $0.2425 | $0.7304 |
| B session k=5 | $0.4232 | $0.2420 | $0.6652 |
| B session k=8 | $0.6716 | $0.2248 | $0.8964 |
| B session k=12 | $1.0075 | $0.2331 | $1.2406 |
| C facts k=5 | $0.0581 | $0.2579 | $0.3159 |
| C facts k=10 | $0.0697 | $0.2746 | $0.3443 |
| C facts k=20 | $0.0996 | $0.2520 | $0.3515 |
| D shared extraction | $0.0716 | $0.2451 | $0.3167 |
| D own extraction | $0.0738 | $0.2772 | $0.3510 |
| D + hydration | $0.2730 | $0.2523 | $0.5253 |
| **total** | **$6.1397** | **$3.9534** | **$10.0931** |

Plus ingestion, which the table above excludes because it is paid once per arm rather than per
configuration: **~$2.47** for Arm C's flat extraction and a second comparable pass for Arm D's
graph-native extraction. Embeddings are negligible (~$0.0002 for the whole corpus), and
projecting `graph_facts` into Neo4j costs **$0** — it is a pure Postgres→Neo4j projection with
no LLM in the loop, so the graph can be rebuilt for free after any schema change.

Two things worth noting from this table. **Judging costs more than answering for every
retrieval configuration** — $0.22-0.28 per row against $0.06-0.27 — because it runs two rubrics
per answer on a more expensive model, and it is the reason judge verdicts are cached as
aggressively as answers. And **Arm A alone is a third of the total answering spend**, which is
the cost side of the 25,576 tokens per question it needs.

---

## Corpus statistics

| | count |
|---|---|
| conversations used | 5 of 10 |
| raw turns | 2,494 |
| sessions | 122 |
| questions per configuration | 40 (10 per category) |
| chunks — turn / window / session granularity | 2,494 / 791 / 122 |
| Arm C facts extracted | 1,718 |
| Arm D graph facts extracted | 1,851 |
| distinct predicates — Arm C / Arm D | 794 / 575 |
| graph: entity nodes / edges / invalidated edges | 1,343 / 1,851 / 3 |

Arm D entity types, from the closed extraction enum:

| type | facts (as subject) |
|---|---|
| Person | 1,702 |
| Object | 85 |
| Organization | 23 |
| Concept | 14 |
| Event | 14 |
| Place | 13 |

The distribution is a reminder of what this corpus is: **92% of extracted facts are about a
person**. That skew is the root of [the star topology](GRAPH.md#1-the-graph-is-a-star) that
made graph traversal ineffective here.

---

← [back to README](../README.md) · [back to results](RESULTS.md) ·
[Why the graph lost](GRAPH.md) · [Design decisions](DESIGN.md)
