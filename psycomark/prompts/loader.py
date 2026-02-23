"""
psycomark.prompts.loader — File-Based Prompt Loading with Builder Fallback.

Searches ``prompts/openai/``, ``prompts/optimized_s1/``, and
``prompts/optimized/`` for optimised ``.txt`` prompt files.  If no file
is found, falls back to the hardcoded builder functions.

Exports:
    - ``S1_PROMPTS``: Singleton with all S1 prompt strings
    - ``S2_PROMPTS``: Singleton with all S2 prompt strings
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from psycomark.prompts.builder import (
    build_s1_discriminative_system,
    build_s1_critic_system,
    build_s1_refiner_system,
    build_s1_user_template,
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
    build_s2_prosecutor_system,
    build_s2_defense_system,
    build_s2_literalist_system,
    build_s2_profiler_system,
    build_s2_prosecutor_user_template,
    build_s2_literalist_user_template,
    build_s2_profiler_user_template,
    build_s2_defense_user_template,
    build_s2_judge_system,
    build_s2_judge_user_template,
    build_s1_ddcot_system,
    build_s1_ddcot_user_template,
    build_s1_ddcot_critic_system,
    build_s1_ddcot_critic_user_template,
    build_s1_ddcot_refiner_system,
    build_s1_ddcot_refiner_user_template,
    build_s2_parallel_prosecutor_system,
    build_s2_parallel_defense_system,
    build_s2_parallel_literalist_system,
    build_s2_parallel_profiler_system,
    build_s2_parallel_user_template,
    build_s2_calibrated_judge_system,
    build_s2_calibrated_judge_user_template,
)


# ---------------------------------------------------------------------------
# Directory Configuration
# ---------------------------------------------------------------------------

S1_PROMPT_DIRS: List[Path] = [
    Path("prompts/openai"),
    Path("prompts/optimized_s1"),
    Path("prompts/optimized"),
]

S2_PROMPT_DIRS: List[Path] = [
    Path("prompts/openai"),
    Path("prompts/optimized_s2"),
    Path("prompts/optimized"),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def load_prompt_or_default(
    filename: str,
    default_func: Callable[[], str],
    search_dirs: Optional[List[Path]] = None,
) -> str:
    """Try loading an optimised prompt file; fall back to hardcoded builder."""
    if search_dirs is None:
        search_dirs = [Path("prompts/openai")]

    for prompt_dir in search_dirs:
        filepath = prompt_dir / filename
        if filepath.exists():
            logger.info(f"Loading Optimised Prompt: {prompt_dir.name}/{filename}")
            return filepath.read_text(encoding="utf-8").strip()

    logger.debug(f"Using default builder for: {filename}")
    return default_func()


# ---------------------------------------------------------------------------
# S1 Prompts Singleton
# ---------------------------------------------------------------------------


class S1Prompts:
    def __init__(self):
        # Legacy
        self.gen_system = load_prompt_or_default(
            "s1_generator_optimized.txt", build_s1_discriminative_system, S1_PROMPT_DIRS
        )
        self.gen_user_template = load_prompt_or_default(
            "s1_user_optimized.txt", build_s1_user_template, S1_PROMPT_DIRS
        )
        self.critic_system = load_prompt_or_default(
            "s1_critic_optimized.txt", build_s1_critic_system, S1_PROMPT_DIRS
        )
        self.critic_user_template = load_prompt_or_default(
            "s1_critic_user_optimized.txt",
            build_s1_critic_user_template,
            S1_PROMPT_DIRS,
        )
        self.refiner_system = load_prompt_or_default(
            "s1_refiner_optimized.txt", build_s1_refiner_system, S1_PROMPT_DIRS
        )
        self.refiner_user_template = load_prompt_or_default(
            "s1_refiner_user_optimized.txt",
            build_s1_refiner_user_template,
            S1_PROMPT_DIRS,
        )

        # DD-CoT
        self.ddcot_gen_system = load_prompt_or_default(
            "s1_ddcot_generator_optimized.txt", build_s1_ddcot_system, S1_PROMPT_DIRS
        )
        self.ddcot_gen_user_template = load_prompt_or_default(
            "s1_ddcot_user_optimized.txt", build_s1_ddcot_user_template, S1_PROMPT_DIRS
        )
        self.ddcot_critic_system = load_prompt_or_default(
            "s1_ddcot_critic_optimized.txt",
            build_s1_ddcot_critic_system,
            S1_PROMPT_DIRS,
        )
        self.ddcot_critic_user_template = load_prompt_or_default(
            "s1_ddcot_critic_user_optimized.txt",
            build_s1_ddcot_critic_user_template,
            S1_PROMPT_DIRS,
        )
        self.ddcot_refiner_system = load_prompt_or_default(
            "s1_ddcot_refiner_optimized.txt",
            build_s1_ddcot_refiner_system,
            S1_PROMPT_DIRS,
        )
        self.ddcot_refiner_user_template = load_prompt_or_default(
            "s1_ddcot_refiner_user_optimized.txt",
            build_s1_ddcot_refiner_user_template,
            S1_PROMPT_DIRS,
        )

        # Pattern (optional)
        self.pat_system = load_prompt_or_default(
            "s1_pattern_generator.txt", lambda: "", S1_PROMPT_DIRS
        )
        self.pat_user_template = load_prompt_or_default(
            "s1_pattern_user.txt", lambda: "", S1_PROMPT_DIRS
        )


S1_PROMPTS = S1Prompts()


# ---------------------------------------------------------------------------
# S2 Prompts Singleton
# ---------------------------------------------------------------------------


class S2Prompts:
    def __init__(self):
        # Legacy (Sequential Debate)
        self.pros_sys = load_prompt_or_default(
            "s2_prosecutor_optimized.txt", build_s2_prosecutor_system, S2_PROMPT_DIRS
        )
        self.def_sys = load_prompt_or_default(
            "s2_defense_optimized.txt", build_s2_defense_system, S2_PROMPT_DIRS
        )
        self.lit_sys = load_prompt_or_default(
            "s2_literalist_optimized.txt", build_s2_literalist_system, S2_PROMPT_DIRS
        )
        self.prof_sys = load_prompt_or_default(
            "s2_profiler_optimized.txt", build_s2_profiler_system, S2_PROMPT_DIRS
        )
        self.judge_sys = load_prompt_or_default(
            "s2_judge_optimized.txt", build_s2_judge_system, S2_PROMPT_DIRS
        )

        self.pros_user = load_prompt_or_default(
            "s2_prosecutor_user_optimized.txt",
            build_s2_prosecutor_user_template,
            S2_PROMPT_DIRS,
        )
        self.def_user = load_prompt_or_default(
            "s2_defense_user_optimized.txt",
            build_s2_defense_user_template,
            S2_PROMPT_DIRS,
        )
        self.lit_user = load_prompt_or_default(
            "s2_literalist_user_optimized.txt",
            build_s2_literalist_user_template,
            S2_PROMPT_DIRS,
        )
        self.prof_user = load_prompt_or_default(
            "s2_profiler_user_optimized.txt",
            build_s2_profiler_user_template,
            S2_PROMPT_DIRS,
        )
        self.judge_user = load_prompt_or_default(
            "s2_judge_user_optimized.txt", build_s2_judge_user_template, S2_PROMPT_DIRS
        )

        # Parallel (Anti-Echo Chamber)
        self.parallel_pros_sys = load_prompt_or_default(
            "s2_parallel_prosecutor.txt",
            build_s2_parallel_prosecutor_system,
            S2_PROMPT_DIRS,
        )
        self.parallel_def_sys = load_prompt_or_default(
            "s2_parallel_defense.txt", build_s2_parallel_defense_system, S2_PROMPT_DIRS
        )
        self.parallel_lit_sys = load_prompt_or_default(
            "s2_parallel_literalist.txt",
            build_s2_parallel_literalist_system,
            S2_PROMPT_DIRS,
        )
        self.parallel_prof_sys = load_prompt_or_default(
            "s2_parallel_profiler.txt",
            build_s2_parallel_profiler_system,
            S2_PROMPT_DIRS,
        )
        self.parallel_user = load_prompt_or_default(
            "s2_parallel_user_optimized.txt",
            build_s2_parallel_user_template,
            S2_PROMPT_DIRS,
        )

        # Calibrated Judge
        self.calibrated_judge_sys = load_prompt_or_default(
            "s2_calibrated_judge_eda.txt",
            build_s2_calibrated_judge_system,
            S2_PROMPT_DIRS,
        )
        self.calibrated_judge_user = load_prompt_or_default(
            "s2_calibrated_judge_user.txt",
            build_s2_calibrated_judge_user_template,
            S2_PROMPT_DIRS,
        )


S2_PROMPTS = S2Prompts()
