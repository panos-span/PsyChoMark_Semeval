#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import html
from pathlib import Path
from typing import List, Dict
from pathlib import Path as _Path


# --------- Artifact IO ----------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ---------- Few-shot & artifact loaders ----------
def load_artifacts(path: str | _Path) -> dict:
    """Loads priors & conflicts produced by make_prompt_artifacts.py"""
    p = _Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1_priors": {}, "s1_conflicts": []}


def load_fewshot_bank(path: str | _Path) -> dict:
    """Loads {"s1":[...], "s2":[...]}"""
    p = _Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1": [], "s2": []}


def playbook_block() -> str:
    return """
<psycomark_playbook version="1.0">
  <cues_actor>vague/collective agents alleging secret coordination: "they", "the elite", "globalists", "deep state", "big pharma".</cues_actor>
  <cues_action>intentional control/hostility/cover-up verbs: plot, scheme, infiltrate, engineer, manipulate, cover up, weaponize.</cues_action>
  <cues_effect>extreme stakes or grand outcomes: total control, enslavement, depopulation, tyranny.</cues_effect>
  <cues_epistemics>self-sealing logic: counter-evidence framed as disinformation; "do your own research"; "connect the dots".</cues_epistemics>
  <pitfalls>Do not rely on keywords alone; distinguish reporting/debunking from endorsement.</pitfalls>
</psycomark_playbook>
""".strip()


# --- add near the top of prompt_builder.py (next to playbook_block) ---
# 1) preamble keeps the role + theory
def psycho_theory_preamble() -> str:
    return """
<psycholinguistic_preamble version="1.0">
  <role>You are an expert computational psycholinguist. Align your reasoning with psycholinguistic and evolutionary accounts of conspiratorial rhetoric for SemEval-2026 PsyCoMark Subtask 1 (marker extraction).</role>
  <marker_definitions>
    <Actor>Agents alleged to secretly orchestrate events; the conspirators.</Actor>
    <Action>Deliberate acts attributed to the Actor (what they do). Verb phrase; exclude outcomes/goals.</Action>
    <Effect>Consequence/goal/purpose of the Action (why/result). Often purpose/result clause.</Effect>
    <Victim>Entity harmed/targeted by the Action.</Victim>
    <Evidence>Support claims: links; quoted+attributed material; numeric facts+units+named source.</Evidence>
  </marker_definitions>
</psycholinguistic_preamble>
""".strip()


def data_profile_block() -> str:
    return """
<data_profile>
  - Domain: Reddit submission statements (first-level comments that summarize a linked post).
  - Length: typically 160-1000 characters after preprocessing.
  - Preprocessing: markdown flattened; URLs replaced with [URL]; leading/trailing whitespace stripped; surrounding quote blocks removed.
  - Markers are defined over this preprocessed text; any indices/spans assume URLs -> [URL].
  - Content is topic-agnostic: politics, news, science, entertainment, etc.
</data_profile>
""".strip()


def s2_markers_guidance_block() -> str:
    return """
<using_s1_markers>
  - S1 markers highlight candidate roles (Actor, Action, Effect, Victim, Evidence).
  - They are often triggered by:
      * ordinary political actions,
      * local crimes or incidents,
      * or reported conspiracy claims.
  - NEVER treat the mere presence of markers as evidence that the document is "conspiracy".
  - A document is "conspiracy" ONLY if, in the full text:
      (a) a hidden, coordinated Actor is alleged,
      (b) an intentional, covert Action is described,
      (c) and Effects are framed as large-scale or grand stakes,
      (d) AND the authorial stance endorses or strongly leans toward this mechanism.
  - When markers appear in texts that merely report, mock, or question conspiracies,
    you must choose "non".
</using_s1_markers>
""".strip()


# --------- S1 builders ----------
def build_s1_system(
    priors: dict | None = None,
    conflicts: list[tuple[str, str]] | None = None,
    use_cot: bool = True,
) -> str:
    priors_str = json.dumps(priors or {}, ensure_ascii=False, separators=(",", ":"))
    conflict_pairs_str = json.dumps(
        conflicts or [], ensure_ascii=False, separators=(",", ":")
    )

    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

    # [IMPROVEMENT] Mandate the generation of the 'why' field
    rationale_instruction = """
<forensic_mandate>
  You are a forensic psycholinguist. Your goal is not just to extract spans, but to justify them.
  For EVERY extracted span, you must populate the "why" field in the output JSON.
  
  Criteria for a valid "why":
  1. CITE THE TRIGGER. Explicitly quote the specific word or phrase in the text that triggers the label (e.g., "The verb 'scheme' implies malicious intent").
  2. NO TAUTOLOGIES. Never say "It fits the definition." Explain *how* it fits.
  3. BE PRECISE. If labeling 'Actor', specify if they are 'Specific' or 'Vague/Collective'.
</forensic_mandate>
""".strip()

    rules = f"""
<rules>
  <evidence_gate>
    Evidence ONLY if at least ONE holds:
      (E1) Contains a URL or bare domain/host (e.g., http(s)://…, example.com, @handle).
      (E2) Quoted material WITH attribution verb AND named source.
      (E3) Numeric fact WITH unit/rates AND named source.
      (E4) Named-source attribution clause without a URL/quote (e.g., "the SEC said", "CNN reported", "Reuters: …") — span must include the named source + reporting verb.
      (E5) Inline citation markers that name a specific outlet or document (e.g., “[SEC 2021]”, “(WHO, 2020)”) — keep the entire marker as the Evidence span.

    Forbidden as Evidence:
      - Bare numbers without units/source.
      - Generic attributions without a named source (e.g., "reports say", "it is said").
      - Organization names alone without an attribution or factual content.

    When selecting Evidence, prefer the smallest span that still includes the qualifying feature(s).
  </evidence_gate>

  <span_rules>
    - Keep spans token-tight; include particles only if integral (e.g., "set up", "cover up").
    - Prefer minimal spans that still fully express the role.

    <actor_scope>
      - An Actor MUST be the agent of a conspiratorial Action.
      - CUE: Vague pronouns ("they") or abstract collectives ("the elite", "Big Pharma").
      - ALLOW: specific people/orgs (e.g., "the CDC", "Bill Gates") as Actors ONLY when they are subjects of intentional conspiratorial Actions.
      - REJECT: neutral mentions of people/orgs in purely descriptive or reporting contexts.
    </actor_scope>

    <victim_scope>
      - A Victim MUST be the entity harmed/targeted by an Action.
      - Reject substrings that are just part of an Action noun (e.g., "child" from "child trafficking").
      - Allow Victim inside an Action only if it's a clear NP object (e.g., "our children", "13-year-olds").
      - CUE: first-person plural and in-group collectives ("we", "us", "our", "the people", "our children", "patriots").
      - CHECK: Prefer spans functioning as the object/indirect object of an Action.
    </victim_scope>

    <action_scope>
      - Action MUST denote intentional agency linked to secrecy/control/hostility/harm.
      - Reject neutral/descriptive reporting verbs ("announced", "reported", "said", "posted", "went") UNLESS they are part of an alleged scheme/cover-up/intentional harm.
      - If a clause contains both an intentional verb and a reporting verb, choose the intentional act as Action and relegate attribution to Evidence (per the gate).
    </action_scope>
  </span_rules>

  <action_effect_split>
    - Action = what is done (verb phrase).
    - Effect = consequence/purpose/result (often NP or "to …"/"so that …").
    - CUE: Effects are frequently purpose clauses ("to …", "in order to …", "so that …") OR catastrophic/high-stakes noun phrases functioning as the object of an Action ("total control", "enslavement", "the great reset").
    - Do not merge Action and Effect; split off purpose/result clauses as Effect.
  </action_effect_split>

  <overlap_policy>
    - Forbid Actor <-> Victim overlaps; if uncertain, prefer Actor unless Victim clearly superior.
    - Allow short Victim NP inside Action; keep both if well-formed.
    - Evidence may overlap others only if part of a quote/citation per <evidence_gate>.
    - Ambiguous pairs hint: "Effect vs Victim", "Action vs Effect", "Action vs Victim", "Action vs Evidence", "Actor vs Evidence"
  </overlap_policy>

  <statistical_priors>
    {priors_str}
  </statistical_priors>
  
  <marker_noise_guidance>
   - Human marker annotations can be noisy: short or ambiguous spans exist.
   - In few-shots, trust the label but focus on prototypical uses (clear agents, actions, effects, and explicit evidence).
  </marker_noise_guidance>


  <notes>
    - Choose exact substrings first; offsets will be auto-filled from your text.
    - Don't over-mark generic function words.
    - If a label is absent, output none for that label.
  </notes>
</rules>
""".strip()

    workflow = ""
    if use_cot:
        workflow = """
<workflow>
  1) Scan roles: Actor, Action, Effect, Victim; then explicitly scan for Evidence.
  2) Apply <evidence_gate>.
  3) Enforce <action_effect_split>.
  4) Tighten boundaries; keep particles only if integral.
  5) Apply <overlap_policy>.
</workflow>""".strip()

    output_contract = """
<verbatim_rule>
  Every span's "text" MUST be a verbatim substring of <text_to_analyze>.
  DO NOT paraphrase, summarize, or invent. Copy exact characters from RAW.
</verbatim_rule>
<output_contract>
  Return a JSON object with a "spans" list. 
  Each span must have: {label, text, start, end, why, context}.
  "text" and "context" must be verbatim substrings from the RAW text.
</output_contract>
""".strip()

    return (
        header
        + "\n\n"
        + rationale_instruction
        + "\n"
        + rules
        + ("\n" + workflow if workflow else "")
        + "\n"
        + output_contract
    ).strip()


# ---------------------------------------------------------------------
# NEW: AoT System Prompt Builder
# ---------------------------------------------------------------------


def build_s1_system_aot(
    priors: dict | None = None,
    conflicts: list[tuple[str, str]] | None = None,
) -> str:
    """
    Algorithm of Thought (AoT) System Prompt.
    Forces a 4-step scan strategy to boost Macro-F1 on rare classes.
    Incorporates statistical priors to guide the model.
    """
    priors_str = json.dumps(priors or {}, ensure_ascii=False, separators=(",", ":"))
    conflict_pairs_str = json.dumps(
        conflicts or [], ensure_ascii=False, separators=(",", ":")
    )

    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

    # The Core Algorithm Instruction
    algorithm_block = """
<algorithm_of_thought>
  To ensure high Macro-F1 (detecting rare classes), you must follow this SEARCH ALGORITHM:

  1. **ACTOR SCAN**: Scan the text specifically for entities (people, groups, vague "they") that are alleged to be *conspiring* or acting in secret.
  2. **ACTION SCAN**: Look for verbs of manipulation, secrecy, or malevolent control attached to those Actors.
  3. **EFFECT/VICTIM SCAN**: Look for the *outcomes* (grand scale, negative) and the *targets* (the innocent, the public).
  4. **EVIDENCE SCAN**: Look for "epistemic" markers (links, "do your research", appeals to authority).

  For each step, list candidate spans in your thought trace, then verify them against definitions.
  Only after completing the scan, populate the `final_spans` list.
</algorithm_of_thought>
""".strip()

    rules = f"""
<rules>
  <evidence_gate>
    Evidence ONLY if at least ONE holds:
      (E1) Contains a URL or bare domain/host.
      (E2) Quoted material WITH attribution verb AND named source.
      (E3) Numeric fact WITH unit/rates AND named source.
      (E4) Named-source attribution clause.
      (E5) Inline citation markers.
  </evidence_gate>

  <span_rules>
    <actor_scope>CUE: Vague pronouns ("they") or abstract collectives.</actor_scope>
    <action_scope>CUE: Intentional agency linked to secrecy/control/harm.</action_scope>
    <effect_scope>CUE: Purpose clauses ("to enslave") or high-stakes outcomes.</effect_scope>
  </span_rules>

  <overlap_policy>
    - Ambiguous pairs hint: {conflict_pairs_str}
    - Prioritize Actor over Victim if ambiguous.
  </overlap_policy>

  <statistical_priors>
    Use these priors as soft guidance for likelihood of certain labels in ambiguous contexts:
    {priors_str}
  </statistical_priors>
</rules>
""".strip()

    output_contract = """
<output_contract>
  Your output must follow the `AoTResponse` schema:
  1. `strategy`: A list of analysis steps showing your work.
  2. `final_spans`: The final cleaned list of spans with `why` rationales.
</output_contract>
""".strip()

    return (
        header + "\n\n" + algorithm_block + "\n" + rules + "\n" + output_contract
    ).strip()


import html, json, re
from typing import List, Dict, Any

_LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _is_example_xml(s: str) -> bool:
    return isinstance(s, str) and "<example>" in s and "</example>" in s


def _clip_text(s: str, max_chars: int) -> str:
    return s[:max_chars] if s and len(s) > max_chars else (s or "")


def _dedup_by_text(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for it in items:
        t = it.get("text", "")
        if t not in seen and t:
            seen.add(t)
            out.append(it)
    return out


def _cap_spans_per_example(spans: List[dict], k: int) -> List[dict]:
    if not spans:
        return []
    kept, seen_txt = [], set()
    for m in spans:
        txt = (m.get("text") or "").strip()
        if not txt or txt in seen_txt:
            continue
        kept.append(m)
        seen_txt.add(txt)
        if len(kept) >= k:
            break
    return kept


def _has_ae_conflict(spans: List[dict]) -> bool:
    # simple heuristic: same sentence-ish or overlapping Action & Effect present
    has_a = any(m.get("label") == "Action" for m in spans)
    has_e = any(m.get("label") == "Effect" for m in spans)
    return has_a and has_e


def _norm_span(m: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Normalize incoming span.

    Minimal required keys:
      - "label" (or "type")
      - "text" (if present)

    Optional keys we *preserve* when available:
      - "start", "end" (or "startIndex", "endIndex")  [currently unused but allowed]
      - "why"        (LLM explanation from make_prompt_artifacts)
      - "context"    (local context window around the span)
    """
    if not isinstance(m, dict):
        return None

    label = m.get("label") or m.get("type")
    if not label or str(label) not in _LABELS:
        return None

    out: Dict[str, Any] = {"label": str(label)}

    # core text
    text = m.get("text")
    if text is not None:
        out["text"] = str(text)

    # optional offsets (kept if present, but we don't rely on them for matching)
    start = m.get("start", m.get("startIndex"))
    end = m.get("end", m.get("endIndex"))
    if start is not None and end is not None:
        try:
            out["start"] = int(start)
            out["end"] = int(end)
        except Exception:
            # if bad, just drop them; text is enough for span identity
            out.pop("start", None)
            out.pop("end", None)

    # preserve explanation/context if present (from make_prompt_artifacts)
    if "why" in m and m["why"] is not None:
        out["why"] = str(m["why"])
    if "context" in m and m["context"] is not None:
        out["context"] = str(m["context"])

    return out


def build_s1_user(
    *,
    text_input: str,
    s1_fewshots: list | None,
    include_cot: bool = True,
    want: int = 8,
    victim_min: int = 1,
    conflict_min: int = 1,
    neg_cap: int = 2,
    per_example_span_cap: int = 4,
    max_text_chars: int = 1200,
) -> str:
    """
    Robust few-shot packer (pydantic-AI ready):
    - Accepts dict few-shots or pre-rendered <example>...</example> strings.
    - Normalizes spans to {'label','text','start?','end?'}.
    - Caps spans/example; dedups by text; limits negatives.
    - Guarantees at least one Victim example and one Action–Effect example when available.
    - Emits <few_shots>…</few_shots> with JSON spans in current schema.
    """
    rendered_blocks: list[str] = []
    raw_structured: list[dict] = []

    # --- Normalize incoming few-shots ---
    for ex in s1_fewshots or []:
        if isinstance(ex, str) and _is_example_xml(ex):
            rendered_blocks.append(ex.strip())
            continue
        if not isinstance(ex, dict):
            continue

        text = _clip_text((ex.get("text") or "").strip(), max_chars=max_text_chars)
        spans_raw = ex.get("answer") or ex.get("spans") or ex.get("markers") or []
        spans = [m for m in (_norm_span(m) for m in spans_raw) if m]

        raw_structured.append(
            {
                "text": text,
                "spans": spans,
                "subreddit": ex.get("subreddit"),
                "_id": ex.get("_id") or ex.get("doc_id"),
            }
        )

    # Deduplicate by text
    raw_structured = _dedup_by_text(raw_structured)

    # Split positives / negatives and cap spans per example
    pos_prepped = []
    for e in raw_structured:
        ss = (
            _cap_spans_per_example(e["spans"], k=per_example_span_cap)
            if e["spans"]
            else []
        )
        pos_prepped.append({**e, "spans": ss})
    positives = [e for e in pos_prepped if e["spans"]]
    negatives = [e for e in pos_prepped if not e["spans"]]

    # Greedy label coverage first
    kept: list[dict] = []
    have_labels = set()
    for e in positives:
        labs = {m["label"] for m in e["spans"]}
        if not labs.issubset(have_labels):
            kept.append(e)
            have_labels |= labs
        if len(kept) >= want:
            break

    # Top-up with remaining positives
    if len(kept) < want:
        for e in positives:
            if e in kept:
                continue
            kept.append(e)
            if len(kept) >= want:
                break

    # Add up to neg_cap negatives if still under want
    if len(kept) < want and negatives:
        room = min(neg_cap, want - len(kept))
        kept.extend(negatives[:room])

    # Guarantees
    def _has_ae_conflict(spans: list[dict]) -> bool:
        labs = {m.get("label") for m in spans or []}
        return "Action" in labs and "Effect" in labs

    def ensure_victim(items: list[dict]) -> list[dict]:
        if any(
            any(m["label"] == "Victim" for m in it.get("spans", [])) for it in items
        ):
            return items
        for e in positives:
            if any(m["label"] == "Victim" for m in e["spans"]) and e not in items:
                return ([e] + items)[:want]
        return items

    def ensure_ae(items: list[dict]) -> list[dict]:
        if any(_has_ae_conflict(it.get("spans", [])) for it in items):
            return items
        for e in positives:
            if _has_ae_conflict(e["spans"]) and e not in items:
                return ([e] + items)[:want]
        return items

    kept = ensure_victim(kept) if victim_min > 0 else kept
    kept = ensure_ae(kept) if conflict_min > 0 else kept
    kept = kept[:want]

    # --- Render <few_shots> blocks in the CURRENT schema ---
    for ex in kept:
        spans = ex.get("spans", [])
        txt = ex.get("text", "")
        block = (
            "<example>\n"
            "<text>" + txt + "</text>\n"
            "<spans>\n"
            + json.dumps(spans, ensure_ascii=False, separators=(",", ":"))
            + "\n</spans>\n"
            "</example>"
        )
        rendered_blocks.append(block)

    fewshots_xml = (
        "<few_shots>\n" + "\n".join(rendered_blocks) + "\n</few_shots>\n\n"
        if rendered_blocks
        else ""
    )

    cot_hint = (
        """
<thinking>
    - Check roles in order: Actor → Action → Effect → Victim → Evidence.
    - Apply evidence_gate; reject bare names/places/dates as Evidence.
    - If Action looks like a neutral report (“reported/announced/went up”), reject unless it implies secrecy/control/harm.
    - Tighten to token boundaries only if the exact substring remains valid.
    (Keep this section under 3 bullets, <=40 tokens total. Do NOT copy text from RAW here.)
</thinking>
        """.strip()
        if include_cot
        else ""
    )
    target_hint = f"<target>Please extract up to {want} concise, token-tight spans if present.</target>"

    raw = text_input or ""
    return f"""{fewshots_xml}{cot_hint}
{target_hint}
<constraint>Return only substrings that already exist in RAW. No paraphrases.</constraint>
<verbatim_rule>Every span's text MUST be a verbatim substring of <text_to_analyze>.</verbatim_rule>
<text_to_analyze>
{raw}
</text_to_analyze>""".strip()


# --- S2 prompt adapter: builds prompts from tech flags and passes cant_tell policy ---
def build_s2_prompts_adapter(
    *,
    text: str,
    markers: list,
    fewshots: list | None,
    tech: str,
    allow_cant_tell: bool = False,
) -> tuple[str, str]:
    use_cot = "cot" in tech
    sys_prompt = build_s2_system(
        include_cot=use_cot,
        allow_cant_tell=allow_cant_tell,
        # the following are accepted but ignored (kept for backward-compat)
        policy_text=None,
        boundary_note=None,
        prompt_arts=None,
    )
    user_prompt = build_s2_user(
        text_input=text,
        s1_output=markers,
        s2_fewshots=fewshots or [],
        include_cot=use_cot,
    )
    return sys_prompt, user_prompt


def build_s2_system(
    *,
    include_cot: bool = True,
    allow_cant_tell: bool = False,  # Kept for compat
    policy_text: str = None,
    boundary_note: str = None,
    prompt_arts: dict = None,
) -> str:
    labels = ["conspiracy", "non"]

    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

    policy = f"""
<classification_policy>
  <labels>Choose exactly one: {", ".join(labels)}.</labels>

  <core_task>
    You will receive a <marker_summary> (the "Plot") and the RAW text.
    Your job is to determine the AUTHOR'S STANCE toward that Plot.
    
    The logic is simple:
    1. Does the text describe a secret plot by powerful actors? (Check Markers)
    2. If YES, does the author BELIEVE it? (Check Stance)
       - YES (Endorsement/Warning/Outrage) -> LABEL: conspiracy
       - NO (Debunking/Reporting/Mocking) -> LABEL: non
  </core_task>

  <positive_cues_for_conspiracy>
    - **Endorsement:** The author treats the plot as a hidden truth ("We must wake up", "They are hiding this").
    - **Cheater Detection:** Anger directed at rule-breakers ("The elite are cheating us").
    - **Epistemic Closure:** "Do your own research", "Mainstream media lies".
  </positive_cues_for_conspiracy>

  <negative_cues_non>
    - **Reporting:** "Users on X are claiming that..." (Attribution to others).
    - **Debunking:** "There is no evidence for..."
    - **Mockery:** "Look at this crazy theory."
    - **Just Asking Questions:** If the text asks questions *without* providing the conspiratorial answer, default to "non".
  </negative_cues_non>

  <using_s1_markers>
    - The <marker_summary> synthesizes the extracted roles into a narrative.
    - IF the summary says "No coherent narrative": The text is likely "non" unless it uses subtle dog-whistles.
    - IF the summary describes a Grand Plot: Check the text for ENDORSEMENT.
      * Example: Summary says "Gates poisoning water". Text says "Idiots think Gates poisoning water." -> Label: NON.
  </using_s1_markers>

  <rationale_guidance>
    - Your rationale must explicitly name the STANCE. 
    - Format: "Author [endorses/reports/debunks] the claim that [Mechanism]."
  </rationale_guidance>
</classification_policy>
""".strip()

    output_contract = """
<output_contract>
  Select the label that best fits the authorial intent.
</output_contract>
""".strip()

    return (header + "\n" + policy + "\n" + output_contract).strip()


# ... (keep existing imports and functions) ...

# ---------------------------------------------------------------------
# NEW: ReX-GoT (Reverse Exclusion) Prompting for S2
# ---------------------------------------------------------------------


def build_s2_system_rex() -> str:
    """
    Reverse Exclusion Graph-of-Thought (ReX-GoT) System Prompt.
    Forces the model to explicitly rule out "Non-Conspiracy" explanations
    (Reporting, Satire) before accepting "Conspiracy".
    """
    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

    rex_instructions = """
<rex_protocol>
  You are a forensic classifier using **Reverse Exclusion Logic**.
  Your goal is to determine the Author's Stance (Endorsement vs. Non-Endorsement) by iteratively trying to **EXCLUDE** potential classifications.
  
  <class_priors>
    **CRITICAL CONTEXT**: The 'Conspiracy' class is NOT a rare anomaly. 
    Historical data shows a balanced distribution:
    - Conspiracy (Endorsement): ~42%
    - Non-Conspiracy (Reporting): ~68%
    
    Do NOT default to 'Non' just because the text is ambiguous. If the evidence for endorsement exists, predict 'Conspiracy'.
  </class_priors>

  <classes>
    A. **Neutral Reporting/Analysis** (Non): The author attributes claims to others ("They say...") or discusses them neutrally.
    B. **Satire/Debunking/Mockery** (Non): The author mentions the plot only to ridicule or disprove it.
    C. **Genuine Endorsement** (Conspiracy): The author asserts the plot as true, urgent, or forbidden knowledge.
  </classes>

  <thought_process>
    For each class, you must attempt to construct an argument for why the text **IS NOT** that class.
    
    1. **Analyze Class A (Reporting)**: "Why is this text NOT just neutral reporting?"
       - *Successful Exclusion:* "It contains unattributed assertions of fact like 'The cabal controls us'."
       - *Failed Exclusion:* "It mostly says 'Users claim that...' so it might be reporting."
       
    2. **Analyze Class B (Satire/Debunking)**: "Why is this text NOT satire or debunking?"
       - *Successful Exclusion:* "The tone is deadly serious and urgent."
       
    3. **Analyze Class C (Endorsement)**: "Why is this text NOT genuine endorsement?"
       - *Successful Exclusion:* "The author calls the theory 'ridiculous'."
       - *Failed Exclusion:* "The text explicitly urges readers to 'wake up' to the truth."
  </thought_process>
</rex_protocol>

<output_contract>
  Select the single class you **COULD NOT** definitively exclude.
  If multiple remain plausible, prefer 'Non' (Reporting) for safety.
  Output JSON: {"label": "conspiracy" or "non", "rationale": "reasoning"}
</output_contract>
""".strip()

    return header + "\n\n" + rex_instructions


def build_s2_user_rex(
    *,
    text_input: str,
    s1_output: List[dict] | None,
    marker_summary: Dict[str, List[str]] | None = None,
    fewshots: List[dict] | None = None,  # <--- NEW ARG
) -> str:
    import json

    # Format markers for context
    markers_str = "[]"
    if s1_output:
        markers_str = json.dumps(s1_output, ensure_ascii=False)

    summary_str = ""
    if marker_summary:
        summary_str = json.dumps(marker_summary, ensure_ascii=False)

    # Format Few-Shots as "Legal Precedents"
    precedents_str = ""
    if fewshots:
        blocks = []
        for i, ex in enumerate(fewshots):
            label = ex.get("label", "unknown")
            txt = ex.get("text", "") or ex.get("doc_text", "")
            rationale = ex.get("rationale", "")
            blocks.append(
                f"""
<case_{i+1}>
  <text>{txt}</text>
  <verdict>{label}</verdict>
  <reasoning>{rationale}</reasoning>
</case_{i+1}>"""
            )
        precedents_str = (
            "<legal_precedents>\n" + "\n".join(blocks) + "\n</legal_precedents>"
        )

    return f"""
{precedents_str}

<case_file>
  <text_to_analyze>
  {text_input}
  </text_to_analyze>

  <extracted_markers>
  {markers_str}
  </extracted_markers>
  
  <narrative_summary>
  {summary_str}
  </narrative_summary>
</case_file>

<execution>
  Apply the Reverse Exclusion Protocol.
  1. Argument: Why is this NOT Reporting? (Check attribution vs assertion)
  2. Argument: Why is this NOT Satire/Debunking? (Check tone)
  3. Argument: Why is this NOT Endorsement? (Check distancing)
  
  Conclusion: Final Label.
</execution>
""".strip()


def build_s2_user(
    *,
    text_input: str,
    s1_output: List[dict] | None,
    s2_fewshots: List[dict] | None = None,
    include_cot: bool = False,
    marker_summary: Dict[str, List[str]] | None = None,
) -> str:
    """
    Updated to render 'marker_summary' in few-shot examples.
    """
    import json

    raw = text_input or ""

    # -------- Few-shot examples (with markers AND summary) --------
    examples_xml = ""
    if s2_fewshots:
        ex_parts: List[str] = []
        valid = {"conspiracy", "non"}
        for ex in s2_fewshots:
            lab = str(ex.get("label", "")).lower()
            if lab not in valid:
                continue

            rationale = str(ex.get("rationale", "") or "").strip()
            if not rationale:
                if lab == "conspiracy":
                    rationale = (
                        "The author endorses a hidden, coordinated conspiracy..."
                    )
                else:
                    rationale = "The author does not endorse a hidden conspiracy..."

            etext = (ex.get("text") or ex.get("doc_text") or "").strip()

            # 1. Markers
            markers = ex.get("markers") or []
            markers_block = ""
            if markers:
                try:
                    markers_json = json.dumps(
                        markers, ensure_ascii=False, separators=(",", ":")
                    )
                    markers_block = f"<markers>{markers_json}</markers>"
                except Exception:
                    markers_block = ""

            # 2. [NEW] Marker Summary (Narrative)
            summary_block = ""
            msum = ex.get("marker_summary")
            if msum:
                if isinstance(msum, (dict, list)):
                    msum = json.dumps(msum, ensure_ascii=False)
                summary_block = f"<marker_summary>{msum}</marker_summary>"

            ex_parts.append(
                "<example>\n"
                f"<label>{lab}</label>\n"
                f"{markers_block}\n"
                f"{summary_block}\n"
                f"<rationale>{rationale}</rationale>\n"
                f"<text>{etext}</text>\n"
                "</example>"
            )

        if ex_parts:
            examples_xml = "<few_shots>\n" + "\n".join(ex_parts) + "\n</few_shots>"

    # -------- Light CoT hint --------
    cot_hint = ""
    if include_cot:
        cot_hint = """
<thinking>
  - Read the full document and infer the AUTHOR'S stance.
  - Only label "conspiracy" if both mechanism and endorsement are present.
</thinking>
""".strip()

    # -------- Current document's S1 markers --------
    markers_xml = "<extracted_markers>[]</extracted_markers>"
    if s1_output:
        markers_xml = (
            "<extracted_markers>\n"
            + json.dumps(s1_output, ensure_ascii=False, separators=(",", ":"))
            + "\n</extracted_markers>"
        )

    # Current document's summary
    summary_xml = "<marker_summary>[]</marker_summary>"
    if marker_summary:
        try:
            val = (
                json.dumps(marker_summary, ensure_ascii=False)
                if isinstance(marker_summary, (dict, list))
                else str(marker_summary)
            )
            summary_xml = f"<marker_summary>\n{val}\n</marker_summary>"
        except Exception:
            summary_xml = "<marker_summary>[]</marker_summary>"

    return f"""
{examples_xml}
{cot_hint}
<text_to_analyze>
{raw}
</text_to_analyze>

{markers_xml}
{summary_xml}
""".strip()


def extract_answer_json(x):
    """
    Returns either:
      - list[dict] (for S1 spans), or
      - dict (for S2 {"label":..., "rationale":..., "confidence":...})
    Robust to dict/bytes/str; prefers JSON inside <answer>...</answer>; else last JSON blob.
    """

    def _as_text(v):
        if v is None:
            return ""
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="ignore")
            except Exception:
                return str(v)
        if isinstance(v, dict):
            cand = v.get("answer") or v.get("text") or v.get("content") or v
            return (
                json.dumps(cand, ensure_ascii=False)
                if not isinstance(cand, str)
                else cand
            )
        return str(v)

    s = _as_text(x)

    # Prefer content inside <answer>...</answer>
    m = re.search(r"<answer>\s*(\{.*?\}|\[.*?\])\s*</answer>", s, re.S)
    blob = m.group(1) if m else None

    # Fallback: take last JSON object/array in the text
    if not blob:
        parts = re.findall(r"(\{.*?\}|\[.*?\])", s, re.S)
        blob = parts[-1] if parts else None

    if not blob:
        return []  # default for S1 call sites

    try:
        js = json.loads(blob)
    except Exception:
        blob2 = re.sub(r",\s*([\}\]])", r"\1", blob)
        try:
            js = json.loads(blob2)
        except Exception:
            return []

    return js


# ---- Utilities (shared) ----
def to_s2_marker(m: dict, txt: str) -> dict:
    """
    Normalize an S1-style span into the S2 schema while PRESERVING metadata.

    Preserves: 'why', 'context', and any other keys in 'm'.
    Updates: 'text' (re-sliced), 'startIndex', 'endIndex', 'type'.
    """
    # 1. Calculate strictly bound offsets
    s = int(m.get("start", m.get("startIndex", 0)))
    e = int(m.get("end", m.get("endIndex", s)))
    s = max(0, min(s, len(txt)))
    e = max(s, min(e, len(txt)))

    # 2. Start with a COPY of the input to keep 'why', 'context', etc.
    out = m.copy()

    # 3. Update/Overwrite with normalized S2 fields
    out.update(
        {
            "type": (m.get("type") or m.get("label")),
            "startIndex": s,
            "endIndex": e,
            "text": txt[s:e],  # Enforce text matches offsets exactly
        }
    )

    # 4. (Optional) cleanup legacy S1 keys to avoid confusion,
    # but keep 'why' and 'context'
    out.pop("start", None)
    out.pop("end", None)
    out.pop("label", None)

    return out
