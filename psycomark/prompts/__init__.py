"""
psycomark.prompts — Prompt Management.

    - ``builder``: Hardcoded prompt construction functions (fallback)
    - ``loader``: File-based prompt loading with builder fallback
"""

from psycomark.prompts.loader import S1_PROMPTS, S2_PROMPTS

__all__ = ["S1_PROMPTS", "S2_PROMPTS"]
