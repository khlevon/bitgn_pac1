"""Centralised configuration — loads .env and exposes typed settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one directory above src/)
load_dotenv(Path(__file__).parent.parent / ".env")

# ── BitGN ────────────────────────────────────────────────────────────────────
BITGN_URL: str = os.getenv("BENCHMARK_HOST") or "https://api.bitgn.com"
BITGN_API_KEY: str = os.getenv("BITGN_API_KEY") or ""
BENCHMARK_ID: str = os.getenv("BENCHMARK_ID") or "bitgn/pac1-dev"

# ── Run / Eval model  (self-hosted Qwen via LiteLLM) ────────────────────────
RUN_MODEL_ID: str = os.getenv("RUN_MODEL_ID") or "openai/qwen3.5"
RUN_MODEL_API_BASE: str = os.getenv("RUN_MODEL_API_BASE") or ""
RUN_MODEL_API_KEY: str = os.getenv("RUN_MODEL_API_KEY") or ""

# ── Optimize model  (Anthropic Claude) ──────────────────────────────────────
OPTIMIZE_MODEL_ID: str = os.getenv("OPTIMIZE_MODEL_ID") or "anthropic/claude-sonnet-4-6"

# ── Agent persistence ────────────────────────────────────────────────────────
OPTIMIZED_AGENT_PATH: str = os.getenv("OPTIMIZED_AGENT_PATH") or "optimized_agent.json"
