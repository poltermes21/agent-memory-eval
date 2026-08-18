"""Central config: loads .env once, exposes paths and settings used across the pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
RUNS_DIR = REPO_ROOT / "runs"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_PATH = DATA_RAW_DIR / "locomo10.json"

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANSWERING_MODEL = os.environ["ANSWERING_MODEL"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "memeval")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "memeval")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "memeval")

# LoCoMo category codes (empirically confirmed against the dataset, not documented
# in the repo README): 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
# 5 adversarial. CLAUDE.md: "Adversarial category is typically excluded."
ADVERSARIAL_CATEGORY = 5

# Cost-control decision (agreed with user 2026-08-10): the initial reference run
# uses a fixed subset of the 5 smallest conversations by turn count, not all 10.
# CLAUDE.md requires arms A-D to be a controlled comparison -- this subset must be
# reused unchanged for every arm, not re-picked per arm, or the comparison breaks.
SAMPLE_CONVERSATIONS = ["conv-30", "conv-26", "conv-49", "conv-50", "conv-42"]
SAMPLE_PER_CATEGORY = 10

# claude-sonnet-5 pricing, USD per million tokens. Intro pricing runs through
# 2026-08-31 (today: 2026-08-10); update after that date. Update if ANSWERING_MODEL
# ever changes -- this is not looked up from an API.
ANSWERING_MODEL_INPUT_PRICE_PER_M = 2.00
ANSWERING_MODEL_OUTPUT_PRICE_PER_M = 10.00

# claude-opus-5 pricing, USD per million tokens. Update if JUDGE_MODEL ever changes.
JUDGE_MODEL_INPUT_PRICE_PER_M = 5.00
JUDGE_MODEL_OUTPUT_PRICE_PER_M = 25.00

# text-embedding-3-small pricing, USD per million tokens. Update if EMBEDDING_MODEL
# ever changes.
EMBEDDING_MODEL_PRICE_PER_M = 0.02

# Arm B retrieval design choice: top-k chunks per query. CLAUDE.md doesn't fix this
# number -- it's ours to set and keep constant across both chunk granularities so
# the comparison between them isn't confounded by a differing k.
ARM_B_TOP_K = 5

# Arm C: same retrieval mechanism as Arm B (exact cosine, top-k, no ANN index) but
# over extracted fact sentences instead of raw text chunks. Deliberate deviation
# from CLAUDE.md, which specifies full-text/text-to-SQL for Arm C: measured on this
# data, Postgres ts_rank has no IDF weighting, so a common subject name ("calvin",
# in 225 of 403 facts) outranks the rare informative terms -- Arm C would have lost
# on ranking-implementation weakness rather than on architecture. Keeping retrieval
# identical to Arm B isolates the one variable that matters for production:
# distilled facts vs raw chunks (the Mem0-vs-classic-RAG fork).
ARM_C_TOP_K = 10
