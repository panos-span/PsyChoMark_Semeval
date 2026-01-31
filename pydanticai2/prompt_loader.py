import sys
import pathlib
from pathlib import Path
from typing import Callable, Optional, List
from loguru import logger

# --- Make repo root importable FIRST ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pydanticai2.prompt_builder import (
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
)

# --- Directory Configuration ---
# S1 prompts: Check dedicated s1 dir FIRST (has newer/larger prompts)
S1_PROMPT_DIRS = [
    Path("prompts/openai"),  # Primary (dedicated OpenAI S1 dir)
    Path("prompts/optimized_s1"),  # Primary (dedicated S1 dir - NEWEST)
    Path("prompts/optimized"),  # Fallback (legacy unified)
]

# S2 prompts: Check dedicated s2 dir FIRST
S2_PROMPT_DIRS = [
    Path("prompts/openai"),  # Primary (dedicated OpenAI S1 dir)
    Path("prompts/optimized_s2"),  # Primary (dedicated S2 dir)
    Path("prompts/optimized"),  # Fallback (legacy unified)
]


def load_prompt_or_default(
    filename: str,
    default_func: Callable[[], str],
    search_dirs: Optional[List[Path]] = None,
) -> str:
    """
    Tries to load an optimized text file from multiple directories.
    If missing in all, falls back to the hardcoded Python builder function.
    """
    if search_dirs is None:
        search_dirs = [Path("prompts/openai")]

    for prompt_dir in search_dirs:
        filepath = prompt_dir / filename
        if filepath.exists():
            logger.info(f"Loading Optimized Prompt: {prompt_dir.name}/{filename}")
            return filepath.read_text(encoding="utf-8").strip()

    # Fallback to builder function
    logger.debug(f"Using default builder for: {filename}")
    return default_func()


class S1Prompts:
    def __init__(self):
        # ===========================================================
        # LEGACY PROMPTS (backward compatibility)
        # ===========================================================

        # 1. Generator System
        self.gen_system = load_prompt_or_default(
            "s1_generator_optimized.txt", build_s1_discriminative_system, S1_PROMPT_DIRS
        )

        # 2. Generator User (Sandwich Trigger)
        self.gen_user_template = load_prompt_or_default(
            "s1_user_optimized.txt", build_s1_user_template, S1_PROMPT_DIRS
        )

        # 3. Critic
        self.critic_system = load_prompt_or_default(
            "s1_critic_optimized.txt", build_s1_critic_system, S1_PROMPT_DIRS
        )

        self.critic_user_template = load_prompt_or_default(
            "s1_critic_user_optimized.txt",
            build_s1_critic_user_template,
            S1_PROMPT_DIRS,
        )

        # 4. Refiner
        self.refiner_system = load_prompt_or_default(
            "s1_refiner_optimized.txt", build_s1_refiner_system, S1_PROMPT_DIRS
        )

        self.refiner_user_template = load_prompt_or_default(
            "s1_refiner_user_optimized.txt",
            build_s1_refiner_user_template,
            S1_PROMPT_DIRS,
        )

        # ===========================================================
        # DD-CoT PROMPTS (Optimal Architecture)
        # ===========================================================
        from pydanticai2.prompt_builder import (
            build_s1_ddcot_system,
            build_s1_ddcot_user_template,
            build_s1_ddcot_critic_system,
            build_s1_ddcot_critic_user_template,
            build_s1_ddcot_refiner_system,
            build_s1_ddcot_refiner_user_template,
        )

        # DD-CoT Generator
        self.ddcot_gen_system = load_prompt_or_default(
            "s1_ddcot_generator_optimized.txt",
            build_s1_ddcot_system,
            S1_PROMPT_DIRS,
        )
        self.ddcot_gen_user_template = load_prompt_or_default(
            "s1_ddcot_user_optimized.txt",
            build_s1_ddcot_user_template,
            S1_PROMPT_DIRS,
        )

        # DD-CoT Critic (Enhanced)
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

        # DD-CoT Refiner
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
        self.pat_system = load_prompt_or_default(
            "s1_pattern_generator.txt",
            lambda: "You are a forensic pattern filter...",
            S1_PROMPT_DIRS,
        )
        self.pat_user_template = load_prompt_or_default(
            "s1_pattern_user.txt",
            lambda: "You are a forensic pattern filter user template...",
            S1_PROMPT_DIRS,
        )


# Singleton instance
S1_PROMPTS = S1Prompts()


class S2Prompts:
    def __init__(self):
        # ===========================================================
        # LEGACY PROMPTS (Sequential Debate - backward compatibility)
        # ===========================================================

        # --- System Prompts (Personas) ---
        self.pros_sys = load_prompt_or_default(
            "s2_prosecutor_optimized.txt",
            build_s2_prosecutor_system,
            S2_PROMPT_DIRS,
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

        # --- User Templates (Triggers) ---
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

        # ===========================================================
        # PARALLEL PROMPTS (Anti-Echo Chamber - Optimal Architecture)
        # ===========================================================
        from pydanticai2.prompt_builder import (
            build_s2_parallel_prosecutor_system,
            build_s2_parallel_defense_system,
            build_s2_parallel_literalist_system,
            build_s2_parallel_profiler_system,
            build_s2_parallel_user_template,
            build_s2_calibrated_judge_system,
            build_s2_calibrated_judge_user_template,
        )

        # Parallel Juror System Prompts
        self.parallel_pros_sys = load_prompt_or_default(
            "s2_parallel_prosecutor.txt",
            build_s2_parallel_prosecutor_system,
            S2_PROMPT_DIRS,
        )
        self.parallel_def_sys = load_prompt_or_default(
            "s2_parallel_defense.txt",
            build_s2_parallel_defense_system,
            S2_PROMPT_DIRS,
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

        # Shared Parallel User Template
        self.parallel_user = load_prompt_or_default(
            "s2_parallel_user_optimized.txt",
            build_s2_parallel_user_template,
            S2_PROMPT_DIRS,
        )

        # Calibrated Judge (Anti-Echo Chamber)
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


# Singleton
S2_PROMPTS = S2Prompts()
