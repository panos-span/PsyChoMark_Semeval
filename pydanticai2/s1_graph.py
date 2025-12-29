import asyncio
from collections import Counter, defaultdict
from typing import TypedDict, List, Dict, Any, Tuple
from loguru import logger
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

# Import your new Discriminative Agent
from psycomark_agents import run_s1_discriminative, S1Span, find_best_span


# --- 1. State Definition ---
class S1GraphState(TypedDict):
    doc_id: str
    text: str  # The raw document
    few_shots: List[dict]  # Context for the agent
    k: int  # <--- NEW: Ensemble Size
    raw_runs: List[List[S1Span]]  # Output from k=3 agents
    consensus_spans: List[S1Span]  # Spans that passed the vote
    final_spans: List[Dict]  # Final spans with start/end indices


# --- 2. Node A: The Ensemble (Parallel Execution) ---
async def s1_ensemble_node(state: S1GraphState):
    """
    Runs the discriminative agent k=3 times in parallel.
    """
    text = state["text"]
    k = state.get("k", 3)  # Use state param or default to 3

    logger.info(f"[{state['doc_id']}] Starting S1 Ensemble (k={k})...")

    # Create tasks for parallel execution
    tasks = [run_s1_discriminative(text, state.get("few_shots", [])) for _ in range(k)]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_runs = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"[{state['doc_id']}] Run {i} failed: {res}")
            valid_runs.append(
                []
            )  # Append empty list to maintain count if needed, or just skip
        else:
            valid_runs.append(res)

    return {"raw_runs": valid_runs}


def _normalize_key(text: str) -> str:
    """
    Normalizes text for voting purposes only.
    1. Lowercase
    2. Strips whitespace
    3. Removes leading 'the ', 'a ', 'an '
    """
    t = text.lower().strip()
    if t.startswith("the "):
        return t[4:]
    if t.startswith("a "):
        return t[2:]
    if t.startswith("an "):
        return t[3:]
    return t


# --- 3. Node B: The Consensus Engine (Voting) ---
def s1_consensus_node(state: S1GraphState):
    """
    Filters spans that didn't appear in at least 2 runs.
    Uses SMART NORMALIZATION to match "The CIA" with "CIA".
    """
    runs = state["raw_runs"]
    if not runs:
        return {"consensus_spans": []}

    # Flatten and count votes
    # Key = (Label, Normalized_Text)
    # Value = Count
    vote_counter = Counter()

    # We keep a map of {Normalized_Key -> Best_Original_Span}
    # We prefer the longest original string (e.g. keep "The CIA" over "CIA")
    best_span_map = {}

    for run in runs:
        seen_in_run = set()
        for span in run:
            # 1. Create robust key
            norm_text = _normalize_key(span.text)
            key = (span.label, norm_text)

            if key not in seen_in_run:
                vote_counter[key] += 1
                seen_in_run.add(key)

                # 2. Store the "Best" representation
                # If we haven't seen this key, OR this new span is longer/capitalized properly, keep it.
                if key not in best_span_map:
                    best_span_map[key] = span
                else:
                    # Heuristic: Prefer longer strings (e.g. "The Central Intelligence Agency" > "CIA")
                    # or strings that are essentially the same but retained casing
                    current_best = best_span_map[key]
                    if len(span.text) > len(current_best.text):
                        best_span_map[key] = span

    # Threshold: 2 out of 3 (Majority)
    # Note: If k=1 (debug mode), we accept everything (threshold=1)
    k_size = len(runs)
    threshold = 2 if k_size >= 3 else 1

    passed_spans = []

    for key, count in vote_counter.items():
        if count >= threshold:
            passed_spans.append(best_span_map[key])

    logger.info(
        f"[{state['doc_id']}] Consensus: {len(passed_spans)} spans passed out of {len(vote_counter)} normalized candidates."
    )
    return {"consensus_spans": passed_spans}


# --- 4. Node C: The Structure Verifier (Offset Mapper) ---
def s1_structure_verifier_node(state: S1GraphState):
    """
    Maps the consensus text strings back to (start, end) indices in the raw text.
    Adapts your old 's1_verifier_impl' logic but runs it deterministically.
    """
    raw_text = state["text"]
    candidates = state["consensus_spans"]
    final_output = []

    # Track assignments to handle duplicate phrases (e.g. "The CIA" appearing twice)
    # Key: (Label, Text) -> Next Nth occurrence to find
    assigned_count = defaultdict(int)

    for span in candidates:
        snippet = span.text
        label = span.label

        # 1. Normalize key for counting
        key = (label, snippet.strip())
        nth = assigned_count[key]

        # 2. Find specific Nth occurrence
        # (Using a helper function defined below)
        start, end = find_best_span(raw_text, snippet, nth=nth)

        if start == -1:
            # Fallback: Try finding ANY occurrence if specific Nth failed
            start, end = find_best_span(raw_text, snippet, nth=0)

        if start != -1:
            # Success: Map it
            # Snap to word boundaries if needed (optional optimization)
            final_output.append(
                {
                    "label": label,
                    "text": raw_text[start:end],  # Use actual text from doc
                    "start": start,
                    "end": end,
                    "why": None,  # Context if we had it
                }
            )
            # Increment counter for next time we see this same phrase
            assigned_count[key] += 1
        else:
            logger.warning(
                f"[{state['doc_id']}] Dropped phantom span: '{snippet}' (Not found in text)"
            )

    # Sort by start position for cleanliness
    final_output.sort(key=lambda x: x["start"])

    return {"final_spans": final_output}


# --- 6. Graph Compilation ---
workflow = StateGraph(S1GraphState)

workflow.add_node("ensemble", s1_ensemble_node)
workflow.add_node("consensus", s1_consensus_node)
workflow.add_node("verifier", s1_structure_verifier_node)

workflow.add_edge(START, "ensemble")
workflow.add_edge("ensemble", "consensus")
workflow.add_edge("consensus", "verifier")
workflow.add_edge("verifier", END)

s1_graph = workflow.compile()

# 2. Generate the PNG bytes
png_bytes = s1_graph.get_graph().draw_mermaid_png()

# 3. Save to a file
with open("s1_graph.png", "wb") as f:
    f.write(png_bytes)
