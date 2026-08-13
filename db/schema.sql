-- Shared schema, one Postgres instance / one database, per CLAUDE.md "Stack".
-- Arms A-D all read from raw_turns. Never let an arm parse the source JSON itself.

-- CLAUDE.md's "Shared: raw turns" section lists turn_id, session_id, session_date,
-- speaker, text. conversation_id is added here out of necessity, not as a spec
-- deviation: LoCoMo is 10 independent conversations, dia_ids like "D1:1" repeat
-- across them, and every downstream arm needs to scope a prompt/query to one
-- conversation. Without it, turn_id/session_id would have to be conversation-prefixed
-- strings and any per-conversation query would mean parsing those strings back apart.
-- turn_index is the turn's chronological position within its session (from the
-- source JSON's own array order -- authoritative). NEVER sort turns by turn_id
-- alone: turn_id embeds the position as text ('...:D1:10' vs '...:D1:2'), and a
-- plain string ORDER BY sorts that lexicographically (D1:1, D1:10, D1:11, ...,
-- D1:19, D1:2, D1:20, ...) -- wrong for any session with 10+ turns. Found
-- 2026-08-12 after it had already silently scrambled turn order in Arm A's
-- transcript and Arm B's session/window chunk text for every session above 9
-- turns. Always ORDER BY session_id, turn_index for chronological order.
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

-- Arm B: vector collection. CLAUDE.md's payload spec is "text, turn_id, session_id,
-- timestamp, speaker" -- turn_id and speaker are nullable here because a
-- granularity='session' or 'window' chunk spans multiple turns and speakers; they
-- are always set for granularity='turn'. Same conversation_id necessity as raw_turns.
--
-- turn_ids is the set of every turn_id a chunk covers (a 1-element array for
-- granularity='turn', all turns in the session for 'session', the window's turns
-- for 'window') -- added so retrieval-recall computation is one query shape for
-- every granularity instead of bespoke logic per granularity in application code.
--
-- No ANN index (no HNSW/IVFFlat) -- CLAUDE.md "Stack" is explicit that Arm B's
-- vector search must be an exact sequential scan, not approximate, and that this
-- must be stated in the README. Do not add one "for speed" later.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id        TEXT PRIMARY KEY,       -- 'turn:<turn_id>', 'session:<session_id>', or 'window:<conversation_id>:<session_id>:<start_turn_id>'
    conversation_id TEXT NOT NULL,
    granularity     TEXT NOT NULL,          -- 'turn', 'session', or 'window'
    turn_id         TEXT,                   -- set iff granularity='turn'
    turn_ids        TEXT[] NOT NULL DEFAULT '{}',  -- every turn_id this chunk covers
    session_id      TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    speaker         TEXT,                   -- set iff granularity='turn'
    text            TEXT NOT NULL,          -- the exact text that was embedded
    embedding       VECTOR(1536) NOT NULL   -- text-embedding-3-small dimensionality
);

CREATE INDEX IF NOT EXISTS idx_embeddings_conversation ON embeddings (conversation_id, granularity);
