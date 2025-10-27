#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Any, Dict, List

from starter.prompt_sweep_joint import  (
    _render_boundary_block, _render_conflicts_block, _render_priors_block
)


# --------- Artifact IO ----------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# --------- S1 builders ----------
def build_s1_system(priors: Dict[str, Any], conflicts: List[List[str]], use_cot: bool) -> str:
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
    
    workflow_block = ""
    if use_cot:
        workflow_block = f"""<workflow>
1) In your <thinking> block, execute exactly two passes using the following XML sections.

<pass_a_candidate_scan>
- Actor: identify candidate agent nouns/phrases.
- Action: identify candidate core verb phrases.
- Effect: identify candidate goal or outcome phrases.
- Victim: identify candidate harmed/targeted entities.
- Evidence: identify candidate citations, quotes, links, or explicit attributions.
</pass_a_candidate_scan>

<pass_b_validation_and_refinement>
- Apply boundary rules: substrings must be tight (no leading/trailing whitespace or punctuation).
- Apply overlap_policy: resolve overlaps (especially Action vs Effect) by splitting or choosing the minimal appropriate span; Actor vs Victim should use the smallest role-specific mention.
- Apply statistical_priors: use the priors on length/position as tie-breakers for ambiguous cases.
- Filter for quality: discard substrings shorter than 3 characters or speculative-only content without explicit wording.
- Final check: if nothing remains, confirm the final answer will be [].
</pass_b_validation_and_refinement>

2) After your reasoning, output ONLY the final JSON array inside <answer>.
</workflow>"""

    return f"""<role>
You are a precision-focused annotator for SemEval-2026 PsyCoMark Task 10, Subtask 1. Extract markers exactly as specified.
</role>

<task_definition>
Extract all markers for: Actor, Action, Effect, Victim, Evidence. Return STRICT JSON ONLY inside <answer>.
</task_definition>

<marker_definitions>
- Actor: agent initiating/controlling events.
- Action: deliberate verb phrase (exclude outcomes/goals).
- Effect: consequence, goal, or purpose of the action.
- Victim: entity targeted or harmed.
- Evidence: explicit citation/link/quote/attribution.
</marker_definitions>

<rules>
<output_format>
Return a JSON array inside <answer> ONLY. Each element:
{{"label":"Actor|Action|Effect|Victim|Evidence","text":"<exact substring from TEXT>","start":<optional int>}}

Notes:
- "text" must be copied verbatim from <text_to_analyze>.
- "start" is optional (a hint). Do NOT include "end"; it will be computed downstream.
- No prose, no extra keys, no trailing commas.
</output_format>

<span_boundaries>
- Substrings are measured over raw characters of <text_to_analyze>.
- Do not output offsets other than the optional "start".
- Keep substrings tight; include particles/prepositions only if integral (e.g., "set up", "cover up", "in charge of").
</span_boundaries>

<overlap_policy>
- Ambiguous pairs: {conflict_pairs_str}.
- Action vs Effect: split verb phrase (Action) from purpose/result (Effect); allow minimal overlap only if unavoidable.
- Actor vs Victim: if the same surface form appears in different roles, pick the smallest role-specific mention for each.
- Evidence may overlap others when it is a quotation or citation.
</overlap_policy>

<offset_scope>
- Offsets are computed exactly over <text_to_analyze> by the evaluator; do not normalize quotes/whitespace.
</offset_scope>

<span_length_limits>
- Minimum substring length: 3 characters (after trimming).
- Maximum substring length: 90; Evidence may reach 120 if it is a single explicit citation/quote.
</span_length_limits>

<evidence_quality>
- Prefer explicit sources: URLs, quotations, “according to …”, named reports.
- Avoid purely hedged claims without sources.
</evidence_quality>

<statistical_priors>
Use as tie-breakers when ambiguous:
{priors_str}
If Action and Effect overlap heavily (IoU ≥ 0.6), prefer the label whose start position is closer to its prior.
</statistical_priors>

<negative_case>
- If no markers are present, output [] in <answer>.
</negative_case>

<forbidden_output>
- Do not output anything outside <answer>.
- Inside <answer>, only keys "label","text","start" are allowed. No comments, NaN/inf, or XML echoes.
</forbidden_output>
</rules>

{workflow_block}
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
        "Provide your reasoning in <thinking> (kept private), then output ONLY the JSON array described in <output_format> inside <answer>."
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


# prompt_builder.py

import json
import re

def build_s2_system(*, policy_text: str | None = None, include_cot: bool = False,
                    boundary_note: str | None = None, prompt_arts: dict | None = None) -> str:
    # Optional helper renders (reuse your existing ones if present)
    boundary_block = ""
    conflicts_block = ""
    priors_block = ""
    if prompt_arts:
        b = _render_boundary_block(prompt_arts)
        c = _render_conflicts_block(prompt_arts)
        p = _render_priors_block(prompt_arts)
        if b: boundary_block = f"\n<boundary_guidance>\n{b}\n</boundary_guidance>"
        if c: conflicts_block = f"\n<conflicts>\n{c}\n</conflicts>"
        if p: priors_block = f"\n<priors>\n{p}\n</priors>"

    policy_block = f"\n<policy>\n{policy_text}\n</policy>" if policy_text else ""
    workflow_block = ""
    if include_cot:
        workflow_block = """\n<workflow>
1. In your <thinking> block, briefly justify the label using the strongest cues (hidden plot, coordinated actor, agenda/cover-up).
2. Then output ONLY the final JSON in <answer>:
   {"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}
</workflow>"""

    return f"""<role>
You are an expert annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Classify a Reddit comment as "conspiracy" or "non".
</role>
<label_definitions>
- "conspiracy": asserts/endorses a hidden plot by actors coordinating actions toward a goal (agenda, cover-up, cabal, etc.).
- "non": neutral, debunking, jokes/irony, or content without hidden-plot framing.
</label_definitions>{policy_block}{boundary_block}{conflicts_block}{priors_block}
<output_format>
Return ONLY a single JSON object inside <answer>:
{{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}}
Ensure probabilities sum to 1.0 and label = argmax.
</output_format>{workflow_block}"""

def build_s2_user(*, text_input: str, s1_output: list | None,
                  s2_fewshots: list | None = None, include_cot: bool = False) -> str:
    ex_block = ""
    if s2_fewshots:
        # few-shot XML (compact)
        parts = []
        for ex in s2_fewshots:
            t = ex.get("text","")
            lbl = (ex.get("label") or (ex.get("gold") or {}).get("label") or "non").lower()
            lbl = "conspiracy" if lbl == "conspiracy" else "non"
            mks = ex.get("markers") or []
            mk_norm = []
            for m in mks:
                lab = (m.get("type") or m.get("label") or "").strip()
                s = m.get("startIndex", m.get("start"))
                e = m.get("endIndex", m.get("end"))
                try:
                    s, e = int(s), int(e)
                except: 
                    continue
                if lab and e > s:
                    mk_norm.append({"type": lab, "startIndex": s, "endIndex": e})
            gold = {"label": lbl, "p_conspiracy": 0.8 if lbl=="conspiracy" else 0.2,
                    "p_non": 0.2 if lbl=="conspiracy" else 0.8,
                    "rationale": "concise example rationale."}
            parts.append(
                "<example>\n"
                "<text>\n" + t + "\n</text>\n" +
                ("<extracted_markers>\n" + json.dumps(mk_norm, ensure_ascii=False) + "\n</extracted_markers>\n" if mk_norm else "") +
                "<answer>\n" + json.dumps(gold, ensure_ascii=False) + "\n</answer>\n"
                "</example>"
            )
        ex_block = "<examples>\n" + "\n\n".join(parts) + "\n</examples>\n\n"

    markers_json = json.dumps(s1_output or [], ensure_ascii=False)

    task_tail = ("Provide brief reasoning in <thinking> then the final JSON in <answer>."
                 if include_cot else
                 "Provide ONLY the final JSON in <answer>.")

    return f"""{ex_block}<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
<extracted_markers>
{markers_json}
</extracted_markers>
Instructions: Use the markers as evidence; ambiguity without hidden-plot framing should lean "non".
{task_tail}
</task>"""



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
