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
def build_s1_system(
    priors: Dict[str, Any],
    conflict_pairs: List[Tuple[str, str]],
    boundary_note: str | None = None,
    policy_text: str | None = None,
    include_cot: bool = True,
) -> str:
    priors_lines = []
    for label, d in (priors or {}).items():
        q90 = d.get("q90_len")
        mode_pos = d.get("mode_pos")
        if isinstance(q90, (int, float)) and isinstance(mode_pos, (int, float)):
            priors_lines.append(
                f"- {label}: q90≈{int(q90)} chars; often starts near {int(round(mode_pos*100))}% of the text."
            )
    priors_block = (
        "\n".join(priors_lines)
        if priors_lines
        else "- (insufficient priors; rely on rules)"
    )

    cps = [f"{a}-{b}" for (a, b) in conflict_pairs] if conflict_pairs else []
    conflicts_block = ", ".join(cps) if cps else "Action-Effect, Actor-Victim"

    boundary_block = (
        f"<boundary_note>\n{boundary_note.strip()}\n</boundary_note>\n"
        if boundary_note
        else ""
    )
    policy_block = (
        f"<policy>\n{policy_text.strip()}\n</policy>\n" if policy_text else ""
    )
    cot_workflow = (
        "<workflow>\n"
        "1) Do your full analysis inside <thinking>.\n"
        "2) Then output ONLY the final JSON inside <answer>.\n"
        "</workflow>\n"
        if include_cot
        else "<workflow>\n"
        "Output ONLY the final JSON inside <answer>. Do not include reasoning.\n"
        "</workflow>\n"
    )

    return f"""
<role>
You are a precision annotator for SemEval-2026 Task 10 (PsyCoMark), Subtask 1.
</role>

<task_definition>
Extract all character spans for: Actor, Action, Effect, Victim, Evidence.
Return STRICT JSON ONLY inside <answer>.
</task_definition>

<marker_definitions>
- Actor: agent initiating/controlling events.
- Action: deliberate verb phrase (exclude outcomes/goals).
- Effect: consequence/goal/purpose of the action.
- Victim: entity targeted/harmed.
- Evidence: explicit citation/link/quote/attribution.
</marker_definitions>

<rules>
<span_boundaries>
- 0-indexed, end-exclusive offsets.
- Tight spans: exclude leading/trailing whitespace and punctuation.
- Prefer minimal distinct spans; Evidence may overlap others.
</span_boundaries>

<overlap_policy>
- Ambiguous pairs: {conflicts_block}.
- Action vs Effect: split verb phrase (Action) from purpose (Effect).
- Actor vs Victim: same entity can appear in both roles in different mentions; keep spans minimal.
</overlap_policy>

<statistical_priors>
Use as tie-breakers when ambiguous:
{priors_block}
</statistical_priors>
{boundary_block}{policy_block}

<offset_scope>
- Compute start/end over EXACTLY the characters inside &lt;text_to_analyze&gt;, 0-indexed, end-exclusive.
- Do NOT rebase or normalize quotes or whitespace. Use the raw text offsets.
</offset_scope>

<span_length_limits>
- Minimum span length: 3 characters (after trimming).
- Maximum span length: 90 characters. Evidence may reach 120 if it is a single explicit citation/quote.
</span_length_limits>

<evidence_quality>
- Prefer explicit sources: URLs, quotations, “according to …”, named reports.
- Avoid purely hedged claims (“apparently”, “maybe”, “people say”) unless accompanied by an explicit source.
</evidence_quality>

<negative_case>
- If NO markers are present, output an empty JSON array [] in &lt;answer&gt;.
</negative_case>

<forbidden_output>
- Do NOT output anything outside &lt;answer&gt;.
- &lt;answer&gt; MUST be valid JSON: only keys "label","start","end". No trailing commas, comments, NaN/inf.
</forbidden_output>

<output_format>
ONLY output a JSON array: [{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}, ...]
The JSON MUST be valid and contain no extra keys. No prose outside <answer>.
</output_format>

{cot_workflow}
""".strip()


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
# --- before: def build_s2_system() -> str:
def build_s2_system(policy_text: str | None = None, include_cot: bool = True) -> str:
    policy_block = (
        f"<policy>\n{policy_text.strip()}\n</policy>\n" if policy_text else ""
    )
    cot_workflow = (
        "<workflow>\n"
        "1) You receive the original text and extracted S1 markers.\n"
        "2) Analyze markers as evidence within <thinking> (narrative coherence, Actor-Action-Victim, intent/effect).\n"
        '3) Output ONLY {"label":"...","rationale":"<=2 sentences"} inside <answer>.\n'
        "</workflow>\n"
        if include_cot
        else "<workflow>\n"
        'Output ONLY {"label":"...","rationale":"<=2 sentences"} inside <answer>. Do not include reasoning.\n'
        "</workflow>\n"
    )
    return (
        "<role>\n"
        'You are an expert social scientist. Classify a Reddit text as "conspiracy", "non", or "cant_tell".\n'
        "</role>\n\n"
        "<label_definitions>\n"
        "- conspiracy: alleges a harmful/illegal secret plan by powerful actors; narrative shows Actor+Action+Victim with intent/effect.\n"
        "- non: no conspiracy allegation.\n"
        "- cant_tell: insufficient or ambiguous evidence.\n"
        "</label_definitions>\n\n"
        f"{policy_block}"
        f"{cot_workflow}"
    ).strip()


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
