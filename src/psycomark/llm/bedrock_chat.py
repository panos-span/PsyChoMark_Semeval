# src/psycomark/llm/bedrock_chat.py
import json
import time
import random
import logging
import base64
import os
from typing import Optional, List
import boto3


class Chat:
    """
    Chat interface for Amazon Bedrock with Claude Sonnet 4.5 support.

    The class manages conversation history and handles retries with exponential backoff.
    It supports both text and image inputs (vision), as well as streaming responses.

    Attributes:
        model_id (str): The Bedrock model identifier
        client: boto3 bedrock-runtime client
        use_stream (bool): Whether to use streaming responses
        payload (dict): The request payload containing system prompts, messages, and parameters
    """

    # Claude Sonnet 4.5 model IDs
    # NOTE: According to AWS Bedrock docs, use cross-region ID without region prefix
    # This works in all regions via intelligent routing
    SONNET_45_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"

    def __init__(
        self,
        model_id: str = None,
        region: str = None,
        client=None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        top_p: Optional[float] = None,  # <- default None
        top_k: Optional[int] = None,  # <- default None
        stop_sequences: Optional[List[str]] = None,
        use_stream: bool = False,
    ):
        """
        Initialize the Chat interface.

        Args:
            model_id: Bedrock model ID. Defaults to Claude Sonnet 4.5 cross-region ID
            region: AWS region. Defaults to AWS_DEFAULT_REGION env var or 'eu-central-1'
            client: Pre-configured boto3 client (optional)
            max_tokens: Maximum tokens in response (Sonnet 4.5 supports up to 8192)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences that stop generation
            use_stream: Enable streaming responses
        """
        # define first to avoid attribute errors later
        self.use_stream = bool(use_stream)

        # region/model
        region = region or os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
        self.model_id = model_id or os.getenv("MODEL_ID", self.SONNET_45_MODEL_ID)

        # boto3 client (let botocore pick up env/instance role automatically)
        self.client = client or boto3.client("bedrock-runtime", region_name=region)

        # ---- sampling: send EITHER top_p OR temperature (not both) ----
        sampling: dict = {}
        if top_p is not None:
            sampling["top_p"] = float(top_p)
        else:
            sampling["temperature"] = float(0.0 if temperature is None else temperature)
        if top_k is not None:
            sampling["top_k"] = int(top_k)

        # request payload
        self.payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": [],
            "messages": [],
            "max_tokens": int(max_tokens),
            **sampling,  # <- only one of {temperature, top_p} plus optional top_k
        }
        if stop_sequences:
            self.payload["stop_sequences"] = list(stop_sequences)

    def add_system(self, text: str):
        """
        Add a system prompt to guide the model's behavior.

        System prompts set the context and instructions for the conversation.
        They appear before any user/assistant messages.

        Args:
            text: System prompt text
        """
        self.payload["system"].append({"type": "text", "text": text})

    def add_user(
        self,
        text: str,
        image_base64: Optional[str] = None,
        media_type: str = "image/jpeg",
    ):
        """
        Add a user message to the conversation.

        Supports multimodal input with both text and images (vision capability).

        Args:
            text: User message text
            image_base64: Base64-encoded image data (optional)
            media_type: Image MIME type (e.g., 'image/jpeg', 'image/png', 'image/webp')
        """
        content = []

        # Add image first if provided (recommended order for vision tasks)
        if image_base64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64,
                    },
                }
            )

        # Add text content
        content.append({"type": "text", "text": text})

        self.payload["messages"].append({"role": "user", "content": content})

    def add_assistant(self, text: str):
        """
        Add an assistant (Claude) response to the conversation history.

        This is used to maintain conversation context or for few-shot prompting.

        Args:
            text: Assistant response text
        """
        self.payload["messages"].append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )

    def _invoke(self) -> dict:
        """
        Internal method to invoke the Bedrock model.

        Handles both streaming and non-streaming responses.

        Returns:
            dict: Response data from Bedrock
        """
        if self.use_stream:
            resp = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(self.payload),
            )
            chunks = []
            for event in resp.get("body"):
                if "chunk" in event:
                    chunks.append(event["chunk"]["bytes"].decode("utf-8"))
            return json.loads("".join(chunks))
        else:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(self.payload),
            )
            return json.loads(resp["body"].read())

    def generate(self, retries: int = 3, backoff: float = 1.5) -> str:
        """
        Generate a response from Claude with retry logic.

        Implements exponential backoff for handling transient failures like
        throttling or temporary service issues.

        The retry strategy uses exponential backoff with jitter:
        wait_time = (backoff^attempt) * (1 + random_jitter)

        For backoff=1.5:
        - Attempt 0: ~1.0s wait
        - Attempt 1: ~1.5s wait
        - Attempt 2: ~2.25s wait

        The jitter (±10% random factor) prevents thundering herd when multiple
        requests fail simultaneously.

        Args:
            retries: Maximum number of retry attempts
            backoff: Exponential backoff multiplier

        Returns:
            str: Generated response text

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(retries):
            try:
                data = self._invoke()

                # Extract text from response
                output = data["content"][0]["text"]

                # Add to conversation history
                self.add_assistant(output)

                return output

            except Exception as e:
                last_exception = e

                if attempt < retries - 1:
                    # Calculate wait time with jitter
                    # Formula: wait = backoff^i * (1 + uniform(0, 0.1))
                    wait_time = (backoff**attempt) * (1 + 0.1 * random.random())

                    logging.warning(
                        f"[bedrock] gen failed (try {attempt + 1}): {e} — "
                        f"retrying in {wait_time:.2f}s"
                    )

                    time.sleep(wait_time)

        # All retries exhausted
        raise last_exception

    def clear_history(self):
        """Clear the conversation history while keeping system prompts."""
        self.payload["messages"] = []

    def get_message_count(self) -> int:
        """Get the number of messages in conversation history."""
        return len(self.payload["messages"])


def load_image(path: str) -> str:
    """
    Load an image file and encode it as base64.

    Args:
        path: Path to the image file

    Returns:
        str: Base64-encoded image data

    Example:
        >>> img_b64 = load_image("photo.jpg")
        >>> chat.add_user("What's in this image?", image_base64=img_b64)
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
