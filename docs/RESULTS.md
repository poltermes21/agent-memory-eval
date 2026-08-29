# Results

← [back to README](../README.md)

---

## The verdict

**For conversational memory of this shape, RAG over LLM-distilled facts (Arm C) is the
architecture to reach for.** Best accuracy of anything measured here — **90.0% at 1,040 tokens
per query**, answering in 7ms — at the price of one LLM extraction pass up front.

What makes it the default rather than merely the winner is that **its whole cost/accuracy curve
sits below the competition's**. Arm C at k=20 uses fewer tokens than chunk RAG at its
*cheapest* setting (1,040 vs 1,656) while scoring 17.5 points higher. It does not win at one
tuned operating point; it wins at every budget.

The temporal knowledge graph, which this project was built to validate, **did not earn its
cost**. It never beat flat retrieval at matched budget, on any configuration, across two
complete rebuilds.

### But not for every case

| if your workload… | reach for | because |
|---|---|---|
| is general conversational recall, cost matters | **Arm C, k=10-20** | 87.5-90.0% at 669-1,040 tok/q, 7ms |
| is dominated by **open-ended / synthesis** questions | **Arm D + hydration** | 90.0% on open-domain vs Arm C's 70-80% |
| needs the **highest retrieval recall** (you post-process the evidence yourself) | **Arm D + hydration** | recall 0.823 vs Arm C's 0.777 |
| **cannot afford an LLM extraction pass** | **Arm B, window chunks, k=20** | 85.0% with zero ingestion LLM cost |
| is latency-critical | **Arm B or C** | 4-8ms vs the graph's 52-137ms |

The honest shape of the result is a **Pareto frontier, not a single winner**. Arm C dominates
the cost/accuracy corner. Arm D buys recall and open-domain accuracy at 4.4x the tokens, 16x
the latency, and a second extraction pass. Arm B buys freedom from extraction at 5.7x the
tokens.

---

## The numbers

40 questions (10 per category, stratified over 5 conversations), same answering model
(`claude-sonnet-5`), same prompt, same frozen judge (`claude-opus-5`) for every row.
**Accuracy is strict-judge** — see [how scoring changes the ranking](#how-you-score-changes-who-wins).
`tok/q` is context sent to the answering model.

These are the configurations that carry a finding. [Every run measured is in
ALL-RUNS.md](ALL-RUNS.md).

**Arm D appears three times** because it was built more than once:

- **shared extraction** — the original, reading Arm C's fact table *unmodified*, so that a
  C-vs-D difference would isolate retrieval architecture alone. It tied Arm C on multi-hop.
- **own extraction** — a rebuild with its own graph-native extraction pipeline (typed
  entities, lemmatized predicates, canonical entity naming), after inspection blamed the flat
  triples rather than retrieval. It produced measurably better-structured data and *worse*
  results — [the full story is in GRAPH.md](GRAPH.md#the-extraction-rebuild).
- **+ hydration** — the rebuild, plus widening each retrieved fact back to its source turn ±2
  before answering.

| row | multi-hop | temporal | open-dom. | single-hop | **accuracy** | recall | tok/q | p95 ms |
|---|---|---|---|---|---|---|---|---|
| **A** — full context | 90.0% | 80.0% | 70.0% | 100.0% | **85.0%** | — | 25,576 | — |
| **B** — turn chunks, k=5 | 10.0% | 30.0% | 30.0% | 50.0% | **30.0%** | 0.165 | 699 | 8 |
| **B** — turn chunks, k=30 | 30.0% | 60.0% | 50.0% | 60.0% | **50.0%** | 0.367 | 2,652 | 7 |
| **B** — window chunks, k=20 | 70.0% | 90.0% | 80.0% | 100.0% | **85.0%** | 0.878 | 5,882 | 6 |
| **B** — session chunks, k=12 | 80.0% | 80.0% | 80.0% | 100.0% | **85.0%** | 0.869 | 12,319 | 4 |
| **C** — facts, k=5 | 80.0% | 90.0% | 70.0% | 80.0% | **80.0%** | 0.664 | **489** | 6 |
| **C** — facts, k=10 | 90.0% | 90.0% | 70.0% | 100.0% | **87.5%** | 0.755 | 669 | 7 |
| **C** — facts, k=20 | 90.0% | 90.0% | 80.0% | 100.0% | **90.0%** | 0.777 | 1,040 | 7 |
| **D** — graph, shared extraction | 90.0% | 70.0% | 80.0% | 100.0% | **85.0%** | 0.740 | 682 | 52 |
| **D** — graph, own extraction | 90.0% | 90.0% | 60.0% | 90.0% | **82.5%** | 0.714 | 728 | 137 |
| **D** — graph + hydration | 80.0% | 80.0% | 90.0% | 100.0% | **87.5%** | **0.823** | 3,205 | 112 |

Why these rows: `turn k=5` is the failure case for chunk granularity and `turn k=30` shows
raising k does not rescue it; `window k=20` and `session k=12` reach identical accuracy at
very different token cost; the three Arm C rows trace the cost/accuracy frontier; the three
Arm D rows are the original controlled build, the rebuild that made it worse, and the variant
with the best recall in the study.

**One question = 2.5 percentage points.** Every number here is a multiple of 2.5 because each
row is 40 questions. A 90.0% vs 87.5% gap is *one question* — read it as noise. The findings
below are the ones that clear that bar by a wide margin.

---

## Three findings

### 1. Storing what was learned beats storing what was said

This is the single-variable comparison — B vs C, same retrieval code, same model, same prompt,
only the indexed unit differs — and it is decisive. Distilled facts win **at every budget**:

| | tokens/q | accuracy |
|---|---|---|
| chunk RAG, cheapest usable (window k=5) | 1,656 | 72.5% |
| chunk RAG, best (window k=20) | 5,882 | 85.0% |
| **fact RAG, cheapest (k=5)** | **489** | **80.0%** |
| **fact RAG, best (k=20)** | **1,040** | **90.0%** |

Fact RAG's *worst* configuration beats chunk RAG's *cheapest* by 7.5 points at 30% of the
tokens. An extracted fact is information-dense; a raw chunk spends most of its tokens on
conversational scaffolding — greetings, hedges, back-and-forth — that costs money to send and
gives the model nothing to answer with.

The cost is an up-front LLM extraction pass ($2.47 here), which fact-based memory products
routinely omit from their published numbers. It is real and it should be counted. But it is
paid once per corpus, while the token saving is paid back on every query.

### 2. Chunk granularity matters more than architecture

Arm B spans **30.0% to 85.0%** — a 55-point range, 22 questions — purely by changing how raw
text is cut before embedding:

- **per-turn** chunks are catastrophic (30.0%, recall 0.165). A single conversational turn
  ("Yeah, me too!") carries almost no retrievable signal on its own — and raising k to 30 only
  reaches 50.0% while spending 4x the tokens. You cannot fix bad granularity with a bigger k.
- **windowed** chunks (a few turns each) recover almost everything: 85.0% at k=20.
- **session** chunks match that accuracy at **2x the tokens** — a whole session is a lot of
  text to buy one relevant fact.

That 55-point spread is larger than the gap between *any two architectures* in this study.
Before choosing between chunk RAG, fact RAG and a graph, get chunking right — it is cheaper
and it matters more.

### 3. Better recall is not automatically better answers

Hydrating retrieved facts back into their source turns ±2 lifts overall recall from 0.714 to
**0.823**, and multi-hop recall from 0.570 to **0.800** — the only intervention in the whole
project that moved multi-hop retrieval at all, after several cheaper ranking strategies were
[tried and rejected](DESIGN.md#rejected-retrieval-strategies).

But accuracy does not follow recall:

| | recall | open-domain | multi-hop | temporal |
|---|---|---|---|---|
| graph, own extraction | 0.714 | 60.0% | 90.0% | 90.0% |
| graph + hydration | **0.823** | **90.0%** | 80.0% | 80.0% |

Open-domain gains 30 points while multi-hop and temporal each lose 10. Raw turns supply the
surrounding context open-ended questions need, and the noise precise questions do not — the
same extra text that lets the model synthesise an answer also gives it more chances to pick the
wrong date.

Two things follow. Hydration is a **per-category** decision, not a global setting. And a memory
system marketed on retrieval recall alone is reporting the metric that is easiest to move and
loosest to the outcome.

---

## How you score changes who wins

Every answer was scored twice: by a **strict** rubric demanding the specific detail (dates,
names, quantities), and by a **lenient** rubric approximating standard LoCoMo-style judging
("conveys the same meaning or is consistent with" the reference). All accuracy above is strict.
This is not a stylistic preference — **the choice of judge changes which architecture appears
to win.**

Ranks below are over **all 16 configurations** measured ([ALL-RUNS.md](ALL-RUNS.md)), not just
the rows shown:

| row | strict | lenient | inflation | rank change |
|---|---|---|---|---|
| D — graph, own extraction | 82.5% (9th) | 95.0% (**2nd**) | **+12.5** | **+7 places** |
| C — facts, k=10 | 87.5% (2nd) | 97.5% (**1st**) | +10.0 | +1 |
| D — graph + hydration | 87.5% (3rd) | 95.0% (3rd) | +7.5 | — |
| B — window chunks, k=20 | 85.0% (5th) | 90.0% (5th) | +5.0 | — |
| C — facts, k=20 | 90.0% (**1st**) | 92.5% (4th) | +2.5 | **−3** |
| A — full context | 85.0% (4th) | 85.0% (9th) | +0.0 | **−5** |

The strict winner (fact RAG at k=20) drops to fourth. The graph arm sitting in the middle of
the pack climbs seven places to second. Full context gains **nothing at all** — it has the
whole transcript, so it rarely produces the topically-close-but-vague answer the lenient rubric
forgives — and it falls five places purely because everything around it inflates past it.

A lenient judge does not add a constant to every row. It rewards, specifically, systems whose
answers are topically right and factually imprecise. Worth noting the other end too: the
largest inflation, +12.5 points, is shared by the graph arm and by the two **worst** chunk
configurations — being flattered by the lenient rubric is a symptom of vague answers, not of a
good memory system.

Across **640 judged answers over 16 configurations**, the two rubrics disagree 39 times, and
the disagreement is entirely one-directional: **39 lenient-PASS/strict-FAIL, and 0 the other
way.** Not one case in 640 where the strict judge was the more generous of the two.

Concrete cases, all PASS under lenient and FAIL under strict:

| question | expected | system answer |
|---|---|---|
| When did Calvin first travel to Tokyo? | between 26 March and 20 April 2023 | "in April 2023" |
| What items did Calvin buy in March 2023? | mansion in Japan, luxury car **Ferrari 488 GTB** | "a new mansion and a new luxury car" |
| What did Calvin and his friends record in August 2023? | **a podcast discussing the rap industry** | "recorded a song together in the studio" |
| What emotions is Joanna feeling about the screenplay? | relief, excitement, **worry, hope**, anxiety | "relief, excitement, and some anxiety" |

The third is not vagueness — it is a wrong answer the lenient judge passed. Agreement is lowest
on **open-domain (90.0%)** and **multi-hop (91.9%)**, versus 98.8% on temporal: the two
categories most often used to argue that one memory architecture beats another are the two
where the scoring is least stable.

**What this does and does not establish.** It shows the size and direction of the gap between
two rubrics, and that the gap is large enough to reorder the leaderboard. It does *not*
establish which rubric is correct in an absolute sense — that would need human-labeled ground
truth, which was planned and then dropped from scope. The strict rubric's correctness rests on
its stated criteria and the worked examples above.

### The judge itself

A **stateless function** — no tools, no memory. Input: (question, expected answer, system
answer). Output: `PASS` / `FAIL`. Never a 0-10 score, never a similarity.

`claude-opus-5` was chosen by testing all three candidate models against a deliberately
borderline case (a relative-date answer that never resolves to the expected value) at zero API
cost: Haiku 4.5 and Sonnet 5 both passed it incorrectly, only Opus failed it correctly. The
judge model is deliberately different from the answering model to avoid self-preference bias.

---

## Limitations

**Sample size.** 40 questions per row, so one question is 2.5 percentage points and
same-category differences of one or two questions are not meaningful. The findings that survive
that bar are the large ones: the 55-point chunk-granularity spread (22 questions), the
39-vs-0 judge-disagreement direction, the 12.5-point reordering under the lenient judge (5
questions), and fact RAG beating chunk RAG at a fraction of the tokens across the whole curve.
A 90.0% vs 87.5% gap between two arms is **not** one of them — which is why the verdict is a
frontier with conditions rather than a ranking. The 5-conversation, 10-per-category subset was
fixed upfront for budget reasons and reused unchanged across every arm.

**Knowledge update is untested.** The bi-temporal invalidation machinery built for Arm D fires
only 3 times on LoCoMo, because LoCoMo has no knowledge-update category. This is the one place
a temporal graph has a genuine structural advantage over flat retrieval — answering "where does
X live *as of* date D" — and this dataset cannot exercise it. The verdict above should be read
as "for conversational recall", not "for agent memory in general". Running LongMemEval, which
does have that category, is the obvious next step and would be the fairest test the graph has
yet had.

**LoCoMo ground truth is noisy.** Some expected answers are malformed
(`"Yesteammates on hisvideo game team."`) or are open-ended essays where any strict boolean
verdict is arguable. This affects open-domain most — which is also where judge agreement is
worst (90.0%). Some of that disagreement is the judges, and some is the dataset.

**Arm C vs Arm D is confounded.** After the mid-project rebuild, the two arms run independent
extraction pipelines, so a C-vs-D difference reflects the two systems overall, not retrieval
architecture alone. Both still start from the same `raw_turns` table and are scored by the same
frozen judge, and the original controlled shared-extraction run is kept and reported as its own
row rather than quietly replaced. [Full explanation](GRAPH.md#the-extraction-rebuild).

**Every arm is this project's own implementation.** No off-the-shelf memory system was measured
alongside them, so these numbers say how the architectures compare *as built here* — not how
this implementation compares to a mature product. A weak implementation of an architecture is
evidence about the implementation, not the architecture, and that caveat applies to all four
arms equally.

---

← [back to README](../README.md) · [Why the graph lost](GRAPH.md) ·
[Design decisions](DESIGN.md) · [Every run](ALL-RUNS.md)
