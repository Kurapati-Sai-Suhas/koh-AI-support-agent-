"""Central configuration. Every path/threshold/seed lives here so runs are reproducible."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# Load .env before any os.getenv below. Without this every LLM setting silently
# stays at its default and a pasted API key does nothing.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:      # python-dotenv missing: real env vars still work
    pass

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
GOLDEN = DATA / "golden_set"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for _p in (RAW, PROCESSED, GOLDEN, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW / "twcs.csv"
SEED = 42

# --- Brand -------------------------------------------------------------------
# Chosen in reports/eda.md by an explicit scoring rubric, not at random.
BRAND = os.getenv("BRAND", "SpotifyCares")

# --- Sampling ----------------------------------------------------------------
# The assignment explicitly expects a subsample. We cap the number of
# (customer message -> brand reply) pairs we model so the whole pipeline
# reproduces in minutes on a laptop.
MAX_PAIRS = int(os.getenv("MAX_PAIRS", 20000))
MIN_CHARS = 15          # shorter customer messages carry no intent signal
MAX_CHARS = 1200        # concatenated multi-tweet bursts beyond this are outliers

# --- Splits ------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15         # of the full set, carved out of the non-test remainder

# --- Decision engine (defaults; retuned on validation by src/tune_thresholds.py)
CONF_THRESHOLD = 0.55       # min predicted-intent probability to auto-handle
MARGIN_THRESHOLD = 0.15     # min gap between top-1 and top-2 probability
SIM_THRESHOLD = 0.30        # min cosine similarity of best historical case
HIGH_RISK_INTENTS = {"billing_payment", "hardware_repair_warranty", "data_loss_backup"}

# --- LLM ---------------------------------------------------------------------
# OpenAI-compatible. Works with NVIDIA NIM, OpenAI, Together, vLLM, etc.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))

RETRIEVAL_K = 4
