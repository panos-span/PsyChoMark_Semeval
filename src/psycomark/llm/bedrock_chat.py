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
from typing import List, Optional
from botocore.exceptions import ClientError

# Configure logging once (respect callers who may reconfigure root logger)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            self.client = boto3.client(service_name="bedrock-runtime", region_name=region_name)
            logging.info(f"Bedrock client ready | model={self.model_id} region={region_name}")
        except Exception as e:
            logging.error("FATAL: Could not initialize Bedrock client (check AWS creds/region).")
            raise e

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

        body = json.dumps(payload)

        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
                data = json.loads(resp["body"].read())

                # Expected Messages API shape:
                # {"content":[{"type":"text","text":"..."}], ...}
                content = data.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    if content[0].get("type") == "text":
                        return content[0].get("text", "")

                logging.warning("Bedrock response missing expected content[0].text; returning empty string.")
                return ""

            except ClientError as e:
                last_exc = e
                logging.warning(f"[Bedrock ClientError try {attempt+1}/{retries}]: {e}")
            except Exception as e:
                last_exc = e
                logging.warning(f"[Bedrock Error try {attempt+1}/{retries}]: {e}")

            if attempt < retries - 1:
                wait = (backoff ** attempt) * (1 + 0.1 * random.random())
                logging.info(f"Retrying in {wait:.2f}s…")
                time.sleep(wait)

        logging.error(f"All {retries} retries failed. Last exception: {last_exc}")
        return ""
