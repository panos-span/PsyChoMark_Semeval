"""
psycomark.config — LLM Configuration, Environment Loading, and Model Wiring.

Handles:
    - Environment variable loading from .env files
    - AWS Bedrock client configuration (legacy)
    - OpenAI model initialization via Pydantic-AI
    - Global concurrency controls (semaphores)
    - Retry logic for API resilience
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from typing import Optional

import boto3
import openai
from botocore.config import Config
from loguru import logger
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# 1. Environment Loading
# ---------------------------------------------------------------------------


def load_dotenv() -> None:
    """Load .env file from repository root into ``os.environ``."""
    root = pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    # Map non-standard names to AWS_* so boto3 sees them
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


load_dotenv()

# ---------------------------------------------------------------------------
# 2. AWS Bedrock Configuration (Legacy — retained for backward compatibility)
# ---------------------------------------------------------------------------

AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID: str = os.getenv(
    "MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)

_boto_config = Config(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 20, "mode": "adaptive"},
)

_bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    config=_boto_config,
)

# ---------------------------------------------------------------------------
# 3. LLM Configuration (Multi-Provider)
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # default to 'openai' or 'bedrock'
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o") # default changed to modern gpt, but overridable
BEDROCK_MODEL_ID = os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

if LLM_PROVIDER == "bedrock":
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider
    
    _bedrock_provider = BedrockProvider(region_name=AWS_REGION)
    LLM = BedrockConverseModel(BEDROCK_MODEL_ID, provider=_bedrock_provider)
    logger.info(f"Using Bedrock Model: {BEDROCK_MODEL_ID}")

else:
    # Default to OpenAI (compatible)
    from pydantic_ai.models.openai import OpenAIModel
    # Note: Using OpenAIModel instead of OpenAIResponsesModel for broader compatibility if needed, 
    # but maintaining type consistency with previous code if possible.
    # The original code used OpenAIResponsesModel. Let's see if we should switch to OpenAIModel
    # for generic usage. OpenAIResponsesModel is often for structured responses but OpenAIModel is more standard.
    # Given the previous code used OpenAIResponsesModel, we'll stick to OpenAIModel for generic support 
    # as per pydantic-ai docs for custom base_url usually.
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. This may fail for some providers.")
        
    _openai_provider = OpenAIProvider(
        api_key=api_key or "missing-key", 
        base_url=OPENAI_BASE_URL
    )
    
    LLM = OpenAIModel(
        model_name=OPENAI_MODEL_ID,
        provider=_openai_provider,
    )
    logger.info(f"Using OpenAI Model: {OPENAI_MODEL_ID} (BaseURL: {OPENAI_BASE_URL})")

# ---------------------------------------------------------------------------
# 4. Local Model Detection & Agent Retries
# ---------------------------------------------------------------------------

# Detect local models (Ollama, vLLM, etc.) by base URL
IS_LOCAL_MODEL = bool(OPENAI_BASE_URL and "localhost" in OPENAI_BASE_URL)
AGENT_RETRIES = 3 if IS_LOCAL_MODEL else 2

if IS_LOCAL_MODEL:
    logger.info(f"Local model detected — AGENT_RETRIES={AGENT_RETRIES}")

# ---------------------------------------------------------------------------
# 5. Concurrency & Retry Controls
# ---------------------------------------------------------------------------

# Semaphore to limit concurrent OpenAI API calls
OPENAI_SEMAPHORE = asyncio.Semaphore(3)

# Tenacity logger for retry reporting
tenacity_logger = logging.getLogger("tenacity")
tenacity_logger.setLevel(logging.WARNING)

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="DEBUG")


@retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
    ),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
    before_sleep=before_sleep_log(tenacity_logger, logging.WARNING),
)
async def safe_agent_run(agent: Agent, message: str, deps=None):
    """Wraps ``Agent.run`` with exponential-backoff retries for API errors."""
    return await agent.run(message, deps=deps)
