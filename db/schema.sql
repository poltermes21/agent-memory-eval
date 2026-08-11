-- Shared schema, one Postgres instance / one database, per CLAUDE.md "Stack".
-- Arms A-D all read from raw_turns. Never let an arm parse the source JSON itself.

-- CLAUDE.md's "Shared: raw turns" section lists turn_id, session_id, session_date,
-- speaker, text. conversation_id is added here out of necessity, not as a spec
-- deviation: LoCoMo is 10 independent conversations, dia_ids like "D1:1" repeat
-- across them, and every downstream arm needs to scope a prompt/query to one
-- conversation. Without it, turn_id/session_id would have to be conversation-prefixed
-- strings and any per-conversation query would mean parsing those strings back apart.
CREATE TABLE IF NOT EXISTS raw_turns (
    turn_id         TEXT PRIMARY KEY,       -- '<conversation_id>:<dia_id>', e.g. 'conv-26:D1:1'
    conversation_id TEXT NOT NULL,          -- LoCoMo sample_id, e.g. 'conv-26'
    session_id      TEXT NOT NULL,          -- '<conversation_id>:session_<n>'
    session_date    TEXT NOT NULL,          -- ISO-8601, e.g. '2023-05-08T13:56:00'
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_turns_conversation ON raw_turns (conversation_id);
CREATE INDEX IF NOT EXISTS idx_raw_turns_session ON raw_turns (session_id);

-- QA pairs are not in CLAUDE.md's schema list (that section covers only the
-- ingestion side shared by arms A-D). Question answering needs question, expected
-- answer, category (for the mandatory per-category metrics) and evidence turn_ids
-- (for the retrieval-recall metric, LongMemEval-style, which LoCoMo also supports
-- via its per-question "evidence" field). category 5 is LoCoMo's adversarial
-- category; CLAUDE.md says it is "typically excluded" from evaluation, so it is
-- loaded but filtered out at query time, not at load time -- keep the load lossless.
CREATE TABLE IF NOT EXISTS qa_pairs (
    qa_id           TEXT PRIMARY KEY,       -- '<conversation_id>:qa:<index>'
    conversation_id TEXT NOT NULL,
    question        TEXT NOT NULL,
    expected_answer TEXT,                   -- NULL for adversarial questions (category 5)
    category        INTEGER NOT NULL,       -- 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial
    evidence        TEXT[] NOT NULL DEFAULT '{}',  -- dia_ids, e.g. '{D1:3,D1:9}'
    adversarial_answer TEXT                 -- LoCoMo's expected "trap" answer for category 5, kept for completeness
);

CREATE INDEX IF NOT EXISTS idx_qa_pairs_conversation ON qa_pairs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_category ON qa_pairs (category);
