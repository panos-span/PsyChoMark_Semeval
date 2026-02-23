"""
psycomark.graphs — LangGraph Workflow Definitions.

Exports the two compiled graphs used at inference time:

    - ``s1_graph``: Generator → Critic → Refiner → Verifier (S1 span extraction)
    - ``s2_graph``: Profiler → Parallel Council → Calibrated Judge (S2 endorsement)
"""

from psycomark.graphs.s1_graph import s1_graph, S1DDCoTGraphState
from psycomark.graphs.s2_graph import s2_graph, S2ParallelGraphState

__all__ = [
    "s1_graph",
    "s2_graph",
    "S1DDCoTGraphState",
    "S2ParallelGraphState",
]
