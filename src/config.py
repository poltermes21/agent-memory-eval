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

# Per-container credentials: a second graph container would get its own vars.
NEO4J_ARMD_URI = os.environ.get("NEO4J_ARMD_URI", "bolt://localhost:7687")
NEO4J_ARMD_USER = os.environ.get("NEO4J_ARMD_USER", "neo4j")
NEO4J_ARMD_PASSWORD = os.environ.get("NEO4J_ARMD_PASSWORD", "memevalgraph")

# 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
ADVERSARIAL_CATEGORY = 5

# FROZEN. Reused unchanged by every arm; re-picking per arm makes them
# incomparable. The 5 smallest conversations by turn count.
SAMPLE_CONVERSATIONS = ["conv-30", "conv-26", "conv-49", "conv-50", "conv-42"]
SAMPLE_PER_CATEGORY = 10

# USD per million tokens, hardcoded per model. Update alongside any model change.
ANSWERING_MODEL_INPUT_PRICE_PER_M = 2.00    # claude-sonnet-5
ANSWERING_MODEL_OUTPUT_PRICE_PER_M = 10.00
JUDGE_MODEL_INPUT_PRICE_PER_M = 5.00        # claude-opus-5
JUDGE_MODEL_OUTPUT_PRICE_PER_M = 25.00
EMBEDDING_MODEL_PRICE_PER_M = 0.02          # text-embedding-3-small

# Retrieval budgets. Keep constant within an arm so its own sweeps are not
# confounded; Arm D's three caps are sized to land near ARM_C_TOP_K edges total.
ARM_B_TOP_K = 5
ARM_C_TOP_K = 10
ARM_D_SEED_K = 8          # entry edges by vector lookup
ARM_D_PER_ENTITY_K = 3    # guaranteed edges per entity named in the question
ARM_D_HOP_K = 5           # extra edges from traversal
ARM_D_HUB_DEGREE = 40     # above this a node is a hub; expanding it returns the whole conversation
