import os
from pathlib import Path
from loguru import logger
from prompt_builder import (
    build_s1_discriminative_system,
    build_s1_critic_system,
    build_s1_refiner_system,
    build_s1_user_template,
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
)

PROMPT_DIR = Path("prompts/optimized")


def load_prompt_or_default(filename: str, default_func: callable) -> str:
    """
    Tries to load an optimized text file.
    If missing, falls back to the hardcoded Python builder function.
    """
    filepath = PROMPT_DIR / filename
    if filepath.exists():
        logger.info(f"Loading Optimized Prompt: {filename}")
        return filepath.read_text(encoding="utf-8").strip()

    # Fallback
    return default_func()


class S1Prompts:
    def __init__(self):
        # 1. Generator System
        self.gen_system = load_prompt_or_default(
            "s1_generator_optimized.txt", build_s1_discriminative_system
        )

        # 2. Generator User (Sandwich Trigger)
        self.gen_user_template = load_prompt_or_default(
            "s1_user_optimized.txt", build_s1_user_template
        )

        # 3. Critic
        self.critic_system = load_prompt_or_default(
            "s1_critic_optimized.txt", build_s1_critic_system
        )

        self.critic_user_template = load_prompt_or_default(
            "s1_critic_user_optimized.txt", build_s1_critic_user_template
        )

        # 4. Refiner
        self.refiner_system = load_prompt_or_default(
            "s1_refiner_optimized.txt", build_s1_refiner_system
        )

        self.refiner_user_template = load_prompt_or_default(
            "s1_refiner_user_optimized.txt", build_s1_refiner_user_template
        )


# Singleton instance
S1_PROMPTS = S1Prompts()
