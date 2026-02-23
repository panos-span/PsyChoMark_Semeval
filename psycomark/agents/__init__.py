"""Agent definitions and runner functions for S1 and S2 pipelines."""

from psycomark.agents.span_utils import (
    deduplicate_overlapping_spans,
    find_best_span,
    find_span_with_context,
    merge_adjacent_spans,
    precompute_span_positions,
    verify_span_boundaries,
)
from psycomark.agents.s1_agents import (
    get_s1_ddcot_critic,
    get_s1_ddcot_generator,
    get_s1_ddcot_refiner,
    run_s1_ddcot,
)
from psycomark.agents.s2_agents import (
    run_s2_calibrated_judge,
    run_s2_parallel_council,
    synthesize_dossier,
)
