#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# --------- Artifact IO ----------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# --------- S1 builders ----------
def build_s1_system(priors: Dict[str, Any], conflicts: List[List[str]]) -> str:
    """
    Builds the streamlined and de-duplicated system prompt for S1 (Marker Extraction).
    This replaces your current complex system prompt.
    """
    priors_str = ""
    # Consolidate all priors into a single, clear list from your artifacts
    for label, data in priors.items():
        q90 = data.get("q90_len")
        mode_pos = data.get("mode_pos")
        if q90 is not None and mode_pos is not None:
            priors_str += f"- **{label}**: Typical span length ≤ {q90:.0f} chars; often starts near {mode_pos*100:.0f}% of the text.\n"

    conflict_pairs_str = (
        ", ".join([f"{p}-{p[1]}" for p in conflicts])
        if conflicts
        else "Action-Effect and Actor-Victim"
    )

    return f"""<role>
You are a precision-focused annotator for the SemEval-2026 PsyCoMark Task 10, Subtask 1. Your sole function is to extract psycholinguistic markers from a given text according to a strict schema and a set of rules.
</role>

<task_definition>
Extract all character spans for the five labels: Actor, Action, Effect, Victim, and Evidence.
</task_definition>

<marker_definitions>
- **Actor**: The agent (person/group) portrayed as initiating, planning, or controlling events.
- **Action**: The deliberate action expressed as a VERB PHRASE (e.g., "hiding information"). Exclude outcomes or goals.
- **Effect**: The consequence, intended goal, or purpose of the action (e.g., "to control the population").
- **Victim**: The entity harmed or targeted by the action.
- **Evidence**: Explicitly cited support, such as links, quotes, named sources, or attributions (e.g., "according to the report...").
</marker_definitions>

<rules>
<output_format>
You MUST output ONLY a valid JSON list of objects inside <answer> tags. Each object must contain three keys: "label", "start", and "end". Do not include any other text or explanations. If NO markers are present, output an empty JSON array.
</output_format>

<span_boundaries>
- Spans must be exact character offsets (0-indexed, end-exclusive).
- Spans must be tight. EXCLUDE leading/trailing whitespace and trailing punctuation.
- Minimum span length: 3 characters. Maximum span length: 90 characters (Evidence may be up to 120).
</span_boundaries>

<overlap_policy>
- Pay special attention to resolving overlaps between common ambiguous pairs like {conflict_pairs_str}.
- **Action vs. Effect**: If a verb phrase contains a purpose, split them. The core action is 'Action'; the purpose/goal is 'Effect'.
- **Actor vs. Victim**: An entity can be both, but spans should be minimal and role-specific for each mention.
- **Evidence**: Can overlap with any other marker type.
</overlap_policy>
</rules>

<statistical_priors>
Use these statistical priors as tie-breakers when a span is ambiguous:
{priors_str}
If Action and Effect overlap heavily (IoU ≥ 0.6), prefer the label whose start position is closer to its prior.
</statistical_priors>

<workflow>
1. First, you will perform a step-by-step analysis of the text inside `<thinking>` tags. In this block, identify potential markers, note any overlaps, and explain how you are applying the rules and priors to resolve them.
2. After your reasoning, you will generate the final, clean JSON output inside `<answer>` tags.
</workflow>
"""


def build_s1_user(
    text_input: str, s1_fewshots: List[Dict[str, Any]], include_cot: bool = True
) -> str:
    # format few-shots compactly
    ex_blocks = []
    for ex in (s1_fewshots or [])[:8]:
        spans = ex.get("spans", [])
        spans_json = json.dumps(spans, ensure_ascii=False)
        ex_blocks.append(
            f"<example>\n<text>\n{ex.get('text','')}\n</text>\n<answer>\n{spans_json}\n</answer>\n</example>"
        )
    examples = (
        "<examples>\n" + "\n".join(ex_blocks) + "\n</examples>" if ex_blocks else ""
    )
    cot_line = (
        "Provide reasoning in <thinking> then the final JSON in <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )
    return f"""
{examples}

<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
{cot_line}
</task>
""".strip()


# --------- S2 builders ----------
def build_s2_system() -> str:
    """Builds the system prompt for S2, setting up the evidence-based reasoning task."""
    return """<role>
You are an expert social scientist specializing in the analysis of online discourse. Your task is to classify a Reddit submission statement as "conspiracy", "non", or "cant_tell" based on the text and a pre-computed analysis of its psycholinguistic markers.
</role>

<label_definitions>
- **conspiracy**: The text alleges a secret plan by a powerful group that is harmful or illegal. The narrative is typically supported by claims of covert actions and specific actors.
- **non**: The text does not contain conspiratorial allegations. It may be a normal news report, opinion, question, or unrelated story.
- **cant_tell**: The text is too ambiguous, short, or lacks sufficient information to make a clear determination.
</label_definitions>

<workflow>
1. You will be given the original text and a JSON list of psycholinguistic markers that were extracted from it.
2. First, analyze the provided markers as evidence inside a `<thinking>` block. Consider their presence, density, and how they connect to form a narrative. A text with a clear Actor, Action, and Victim is a strong signal for a conspiracy. The absence of these markers is a strong signal for non-conspiracy.
3. Based on your analysis of the markers in the context of the original text, make a final classification.
4. Provide your final answer as a single JSON object inside an `<answer>` block: {"label": "...", "rationale": "..."}
</workflow>
"""


def build_s2_user(
    text_input: str,
    s1_output: List[Dict[str, Any]],
    s2_fewshots: List[Dict[str, Any]],
    include_cot: bool = True,
) -> str:
    ex_blocks = []
    for ex in (s2_fewshots or [])[:6]:
        markers = ex.get("markers", [])
        clf = {"label": ex.get("label", "non"), "rationale": ex.get("rationale", "")}
        ex_blocks.append(
            "<example>\n"
            f"<text>\n{ex.get('text','')}\n</text>\n"
            f"<extracted_markers>\n{json.dumps(markers, ensure_ascii=False)}\n</extracted_markers>\n"
            f"<answer>\n{json.dumps(clf, ensure_ascii=False)}\n</answer>\n"
            "</example>"
        )
    examples = (
        "<examples>\n" + "\n".join(ex_blocks) + "\n</examples>" if ex_blocks else ""
    )
    cot_line = (
        "Provide reasoning in <thinking> then the final JSON in <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )
    return f"""
{examples}

<task>
<text_to_analyze>
{text_input}
</text_to_analyze>

<extracted_markers>
{json.dumps(s1_output or [], ensure_ascii=False)}
</extracted_markers>
{cot_line}
</task>
""".strip()


# --------- Utilities ----------
_SMART = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
}


def _normalize_quotes(s: str) -> str:
    for k, v in _SMART.items():
        s = s.replace(k, v)
    return s


def extract_answer_json(text: str):
    """Extract JSON from inside <answer>...</answer>. Falls back to last JSON-like block."""
    if not isinstance(text, str):
        return None
    s = _normalize_quotes(text)
    m = re.search(r"<answer>\s*(\[.*?\]|\{.*?\})\s*</answer>", s, re.S)
    blob = m.group(1) if m else None
    if not blob:
        # fallback: last JSON-looking block
        cand = re.findall(r"(\[.*\]|\{.*\})", s, re.S)
        blob = cand[-1] if cand else None
    if not blob:
        return None
    # simple cleanup
    blob = blob.strip().strip("`")
    try:
        return json.loads(blob)
    except Exception:
        # crude repairs
        blob = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
        blob = blob.replace("'", '"')  # single -> double
        try:
            return json.loads(blob)
        except Exception:
            return None
