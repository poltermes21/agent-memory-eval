-- One Postgres instance, one database, shared by all arms.
-- Every arm reads from raw_turns; no arm parses the source JSON itself.

-- Always ORDER BY session_id, turn_index. Sorting by turn_id is lexicographic
-- ('D1:10' before 'D1:2') and silently scrambles any session with 10+ turns.
CREATE TABLE IF NOT EXISTS raw_turns (
    turn_id         TEXT PRIMARY KEY,       -- '<conversation_id>:<dia_id>', e.g. 'conv-26:D1:1'
    conversation_id TEXT NOT NULL,          -- LoCoMo sample_id, e.g. 'conv-26'
    session_id      TEXT NOT NULL,          -- '<conversation_id>:session_<n>'
    session_date    TEXT NOT NULL,          -- ISO-8601, e.g. '2023-05-08T13:56:00'
    turn_index      INTEGER NOT NULL,       -- chronological position within the session, 1-based
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_turns_conversation ON raw_turns (conversation_id);
CREATE INDEX IF NOT EXISTS idx_raw_turns_session ON raw_turns (session_id);

-- Adversarial questions (category 5) are loaded but excluded at query time, so
-- the load stays lossless.
CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id           TEXT PRIMARY KEY,       -- '<conversation_id>:qa:<index>'
    conversation_id TEXT NOT NULL,
    question        TEXT NOT NULL,
    expected_answer TEXT,                   -- NULL for adversarial questions
    category        INTEGER NOT NULL,       -- 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
    evidence        TEXT[] NOT NULL DEFAULT '{}',  -- dia_ids the answer lives in, e.g. '{D1:3,D1:9}'
    adversarial_answer TEXT
);

CREATE INDEX IF NOT EXISTS idx_qa_pairs_conversation ON qa_pairs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_category ON qa_pairs (category);

-- Arm B: chunk embeddings. No ANN index (no HNSW/IVFFlat) anywhere in this
-- schema -- vector search is an exact sequential scan by design. Do not add one.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id        TEXT PRIMARY KEY,       -- 'turn:<turn_id>' | 'session:<session_id>' | 'window:<conversation_id>:<session_id>:<start_turn_id>'
    conversation_id TEXT NOT NULL,
    granularity     TEXT NOT NULL,          -- 'turn', 'session', or 'window'
    turn_id         TEXT,                   -- set iff granularity='turn'
    turn_ids        TEXT[] NOT NULL DEFAULT '{}',  -- every turn this chunk covers; makes recall one query shape for all granularities
    session_id      TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    speaker         TEXT,                   -- set iff granularity='turn'
    text            TEXT NOT NULL,          -- the exact text that was embedded
    embedding       VECTOR(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_conversation ON embeddings (conversation_id, granularity);

-- Arm C's extracted facts. FROZEN: the shared-extraction Arm D results were
-- computed from this table and are still reported.
-- Stores both the triple (for finding) and the sentence (for answering).
CREATE TABLE IF NOT EXISTS facts (
    fact_key        TEXT PRIMARY KEY,       -- sha256(lower(subject)|lower(predicate)|lower(object)|source_turn_id); idempotency key
    conversation_id TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    fact            TEXT NOT NULL,          -- full sentence, tone/hedging preserved
    source_turn_id  TEXT NOT NULL,
    session_date    TEXT NOT NULL,          -- ISO-8601, inherited from raw_turns
    valid_from      TEXT,                   -- ISO-8601, resolved at extraction time; NULL if undated
    valid_to        TEXT,                   -- NULL until a later fact invalidates this one
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_conversation ON facts (conversation_id);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (conversation_id, lower(subject));

-- Separate table rather than a column on facts, so extraction stays independent
-- of whether an arm embeds its output.
CREATE TABLE IF NOT EXISTS fact_embeddings (
    fact_key        TEXT PRIMARY KEY REFERENCES facts (fact_key) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_embeddings_conversation ON fact_embeddings (conversation_id);

-- Arm D's own extraction. Deliberately NOT the `facts` table: the two run
-- independent extraction pipelines and their fact_keys do not correspond.
CREATE TABLE IF NOT EXISTS graph_facts (
    fact_key        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    subject         TEXT NOT NULL,
    subject_type    TEXT NOT NULL,          -- Person | Place | Organization | Object | Event | Concept
    predicate       TEXT NOT NULL,          -- LLM-lemmatized, base/infinitive form
    object          TEXT NOT NULL,
    object_type     TEXT NOT NULL,
    fact            TEXT NOT NULL,
    source_turn_id  TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    valid_from      TEXT,
    valid_to        TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_facts_conversation ON graph_facts (conversation_id);
CREATE INDEX IF NOT EXISTS idx_graph_facts_subject ON graph_facts (conversation_id, lower(subject));

CREATE TABLE IF NOT EXISTS graph_fact_embeddings (
    fact_key        TEXT PRIMARY KEY REFERENCES graph_facts (fact_key) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    embedding       VECTOR(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_fact_embeddings_conversation ON graph_fact_embeddings (conversation_id);
