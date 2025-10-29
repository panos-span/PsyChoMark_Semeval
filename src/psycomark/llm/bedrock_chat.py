#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bedrock_chat.py

A robust, stateless client for Anthropic Claude on Amazon Bedrock.
Optimized for pipeline-style, non-conversational batch tasks.
"""

import boto3
import json
import time
import random
import logging
from typing import List, Optional, Dict, Any
from botocore.exceptions import ClientError
from botocore.config import Config

# Configure logging once (respect callers who may reconfigure root logger)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockChat:
    """
    Stateless wrapper: each call to `chat()` is an independent request.
    No conversation history is stored.
    """

    def __init__(self, model_id: str, region_name: str = "eu-central-1"):
        """
        Args:
            model_id: e.g. 'anthropic.claude-sonnet-4-5-20250929-v1:0'
            region_name: AWS region for Bedrock runtime.
        """
        self.model_id = model_id
        try:
            # --- MODIFICATION START ---
            # Increase the read timeout to accommodate long-running models.
            # The default is 60 seconds, which is often too short for complex prompts.
            # 900 seconds (15 minutes) is a safe starting point.
            config = Config(
                read_timeout=900,
                connect_timeout=60,
                retries={"max_attempts": 0},  # Let your script's retry logic handle it
            )
            self.client = boto3.client(
                service_name="bedrock-runtime", region_name=region_name, config=config
            )
            logging.info(
                f"Bedrock client ready | model={self.model_id} region={region_name}"
            )
        except Exception as e:
            logging.error(
                "FATAL: Could not initialize Bedrock client (check AWS creds/region)."
            )
            raise e

    @staticmethod
    def _parse_content_blocks(data: Dict[str, Any]) -> Dict[str, str]:
        """
        Handles both standard responses (single text block) and Extended Thinking
        responses (sequence of 'thinking' blocks followed by 'text').
        Returns {"thinking": "...", "answer": "..."} (empty strings if absent).
        """
        thinking_parts: List[str] = []
        final_answer = ""

        # Claude Messages API returns {"content": [{"type": "...", ...}, ...]}
        content = data.get("content") or []
        for block in content:
            btype = block.get("type")
            if btype == "thinking":
                # Claude 4 models often return a short summary here.
                # We concatenate them; production can choose to ignore/log it.
                txt = block.get("text") or ""
                if txt:
                    thinking_parts.append(txt)
            elif btype == "text":
                final_answer = block.get("text") or final_answer

        # Fallback: some older paths put text directly at top-level
        if (
            not final_answer
            and isinstance(data.get("content"), list)
            and data["content"]
        ):
            maybe_text = data["content"][0].get("text")
            if isinstance(maybe_text, str):
                final_answer = maybe_text

        return {
            "thinking": ("\n".join(thinking_parts)).strip(),
            "answer": (final_answer or "").strip(),
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        retries: int = 3,
        backoff: float = 1.5,
        # --- NEW ---
        enable_extended_thinking: bool = False,
        thinking_budget_tokens: Optional[int] = None,
    ) -> str:
        """
        Single, stateless Messages API call with retries.

        Returns the assistant text (or "" if all retries fail).
        """
        payload = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": int(max_tokens),
            "system": str(system_prompt or ""),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": str(user_prompt or "")}],
                }
            ],
        }

        # Sampling params: Anthropic allows temperature +/or top_p; keep simple
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if stop_sequences:
            payload["stop_sequences"] = list(stop_sequences)

        # --- NEW: Extended Thinking injection ---
        if enable_extended_thinking:
            if (
                thinking_budget_tokens is None
                or thinking_budget_tokens <= 0
                or thinking_budget_tokens >= max_tokens
            ):
                raise ValueError(
                    "When enable_extended_thinking=True, thinking_budget_tokens must be set, >0, and < max_tokens."
                )
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_budget_tokens),
            }

        body = json.dumps(payload)

        last_exc: Optional[Exception] = None
        tried_without_xt = False
        for attempt in range(retries):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
                data = json.loads(resp["body"].read())

                # Expected Messages API shape:s
                # {"content":[{"type":"text","text":"..."}], ...}
                # --- NEW: robust parsing for thinking + text blocks ---
                parsed = self._parse_content_blocks(data)
                if parsed["answer"] or parsed["thinking"]:
                    return parsed
                logging.warning(
                    "Bedrock response contained no 'text' or 'thinking' blocks; returning empty strings."
                )
                return {"thinking": "", "answer": ""}
            except ClientError as e:
                last_exc = e
                msg = str(e)
                logging.warning(
                    f"[Bedrock ClientError try {attempt+1}/{retries}]: {msg}"
                )
                # If Extended Thinking is not supported in this region/model, auto-fallback once
                if (
                    enable_extended_thinking
                    and not tried_without_xt
                    and ("thinking" in msg or "validation" in msg.lower())
                ):
                    logging.info(
                        "Extended Thinking not accepted by model/region. Retrying once without it."
                    )
                    enable_extended_thinking = False
                    tried_without_xt = True
                    continue
            except Exception as e:
                last_exc = e
                logging.warning(f"[Bedrock Error try {attempt+1}/{retries}]: {e}")

            if attempt < retries - 1:
                wait = (backoff**attempt) * (1 + 0.1 * random.random())
                logging.info(f"Retrying in {wait:.2f}s…")
                time.sleep(wait)

        logging.error(f"All {retries} retries failed. Last exception: {last_exc}")
        return {"thinking": "", "answer": ""}
