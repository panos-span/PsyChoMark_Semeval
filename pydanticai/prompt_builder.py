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
  - If <extracted_markers> are provided, treat them as noisy *hints* about Actor, Action, Effect, Victim, and Evidence.
  - They can be incomplete or partially wrong. NEVER decide solely from markers.
  - ALWAYS base the final label on:
      (a) the full RAW text, and
      (b) the authorial stance: endorsing vs. criticizing conspiratorial claims.
  - Use markers in your *reasoning and rationale*:
      - For "conspiracy": you may briefly note how Actor/Action/Effect markers support a coordinated, intentional mechanism.
      - For "non": you may note that markers pick up roles or claims, but the text distances itself (e.g., reports, debunks, or questions them).
  - In rationales:
      - Optionally mention markers when they make the reasoning clearer (e.g., "Markers highlight a vague Actor and a grand Effect, matching conspiratorial framing").
      - Keep rationales short and focused on stance + mechanism, not on listing every marker.
</using_s1_markers>
""".strip()


# --------- S1 builders ----------
def build_s1_system(
    priors: dict | None = None,
    conflicts: list[tuple[str, str]] | None = None,
    use_cot: bool = True,
) -> str:
    """
    Pydantic-AI mode:
      - No JSON schema / <answer> formatting rules (handled by output_type).
      - Single inclusion of theory + playbook.
      - Domain guidance only (Evidence gate, Action↔Effect split, overlap policy, priors).
    """
    priors_str = json.dumps(priors or {}, ensure_ascii=False, separators=(",", ":"))
    conflict_pairs_str = json.dumps(
        conflicts or [], ensure_ascii=False, separators=(",", ":")
    )

    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

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
      - CUE: vague pronouns or abstract collectives ("they", "the elite", "globalists", "big pharma").
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
    - Ambiguous pairs hint: {conflict_pairs_str}
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
<output_contract>Provide verbatim text</output_contract>
""".strip()

    return (
        header
        + "\n"
        + rules
        + ("\n" + workflow if workflow else "")
        + "\n"
        + output_contract
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
        allow_cant_tell=allow_cant_tell,
    )
    return sys_prompt, user_prompt


def build_s2_system(
    *,
    include_cot: bool = True,
) -> str:
    """
    Pydantic-AI mode for S2 (document-level classification):
      - No JSON schema here; output type is enforced by the agent.
      - Single inclusion of theory + playbook (re-uses S1 helpers).
      - Strong boundary rules to reduce false positives.
      - Explicit guidance on using S1 markers as *evidence*, not as the label itself.
    """
    labels = ["conspiracy", "non"]

    header = (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )

    policy = f"""
<classification_policy>
  <labels>Choose exactly one: {", ".join(labels)}.</labels>

  <positive_cues_for_conspiracy>
    - Coordinated, secretive, or omnipotent Actor(s) alleged to direct events (e.g., "deep state", "globalists", "big pharma", named cabals).
    - Intentional Action of control/cover-up/engineering/weaponization; not mere reporting.
    - Effects framed as extreme stakes or grand plans (enslavement, depopulation, total control).
    - Self-sealing epistemics: counter-evidence rebranded as disinformation/cover-up; "do your own research"/"connect the dots".
    - Narrative glue: multi-event linkage into a single hidden plot (e.g., tying unrelated crises to one cabal).
  </positive_cues_for_conspiracy>

  <negative_cues_non>
    - Straight reporting, quotations, or debate without endorsing conspiratorial mechanism.
    - Ordinary skepticism, policy critique, or corruption claims limited to documented facts without secret coordination claims.
    - Mere name-dropping of entities or URLs without conspiratorial frame.
    - Satire/irony where conspiracist content is lampooned or explicitly rejected.
  </negative_cues_non>

  <ambiguous_and_edge_cases>
    - Questions-as-accusations ("Is X running Y?") count *only if* the text supplies hidden coordination/intent as the explanation; otherwise treat as non.
    - Lists of allegations with sources: label depends on whether a hidden coordination mechanism is asserted/assumed.
    - Reporting-on-conspiracy: if the authorial voice is clearly descriptive/critical, prefer "non".
    - If evidence is fragmentary and the mechanism is implied but not stated, and you cannot infer intent/coordination from context: treat as "non" (gold labels are noisy—be conservative).
  </ambiguous_and_edge_cases>

  <using_s1_markers>
    - S1 spans (Actor/Action/Effect/Victim/Evidence) are noisy clues, not labels.
    - Always decide the final label from the full document and the authorial stance,
      even when S1 markers are present.
    - A doc is "conspiracy" when S1 spans jointly instantiate a conspiratorial *mechanism*:
        Actor (cabal/agent) + Action (intentional secrecy/control/hostility) -> Effect (grand outcome),
      optionally supported by Evidence spans.
    - Isolated Victim/Evidence spans without a coordinated Action+Actor do not suffice.
  </using_s1_markers>


  <tie_breakers>
    - If cues conflict: require both intentional Action and coordinated Actor for "conspiracy".
    - If only one is present or stance is unclear: choose "non".
  </tie_breakers>
  
  <annotation_uncertainty>
   - Gold labels come from multiple crowd annotators (Krippendorff's alpha around 0.58).
   - Treat borderline cases conservatively: avoid over-interpreting vague language as conspiratorial.
   - A single suggestive phrase does not suffice: look for a coherent mechanism and stance.
  </annotation_uncertainty>

  <rationale_guidance>
    - Provide 1-2 short sentences naming decisive cues (e.g., "alleges secret coordination by X; frames Y as intentional cover-up").
    - Do not summarize the whole document; cite the cues, not long quotes.
  </rationale_guidance>
</classification_policy>
""".strip()

    cot = (
        """
<workflow>
  1) Scan for S1-style roles in the doc (Actor, Action, Effect, Victim, Evidence).
  2) Check if Actor+Action imply hidden coordination/intentionality -> if yes, identify Effect scale.
  3) Apply boundary rules (reporting vs endorsing; satire; ordinary critique).
  4) Decide label using <tie_breakers>.
  5) Write a compact rationale naming the decisive cues (no summaries).
</workflow>
""".strip()
        if include_cot
        else ""
    )

    output_contract = """
<output_contract>
  - Output the single label only (the agent's output validator handles schema).
  - Keep rationale concise and focused on cues (when rationale is requested downstream).
</output_contract>
""".strip()

    return (
        header + "\n" + policy + ("\n" + cot if cot else "") + "\n" + output_contract
    ).strip()


def build_s2_user(
    *,
    text_input: str,
    s1_output: List[dict] | None,
    s2_fewshots: List[dict] | None = None,
    include_cot: bool = False,
) -> str:
    """
    Pydantic-AI mode:
      - Embed RAW text and normalized S1 markers as evidence.
      - Few-shots contain {label, rationale} only (compact).
    """
    raw = text_input or ""

    examples_xml = ""
    if s2_fewshots:
        ex_parts = []
        valid = {"conspiracy", "non"}
        for ex in s2_fewshots:
            lab = str(ex.get("label", "")).lower()
            if lab not in valid:
                continue

            rationale = ex.get("rationale", "")
            etext = ex.get("text", "")

            # NEW: optional markers per S2 few-shot (aligned S1 spans)
            markers = ex.get("markers") or []
            markers_block = ""
            if markers:
                try:
                    markers_json = json.dumps(
                        markers, ensure_ascii=False, separators=(",", ":")
                    )
                    markers_block = f"<markers>{markers_json}</markers>"
                except Exception:
                    # If something is off with markers serialization, just skip them
                    markers_block = ""

            ex_parts.append(
                "<example>"
                f"<label>{lab}</label>"
                f"<rationale>{rationale}</rationale>"
                f"<text>{etext}</text>"
                f"{markers_block}"
                "</example>"
            )

        if ex_parts:
            examples_xml = "<few_shots>\n" + "\n".join(ex_parts) + "\n</few_shots>"

    cot_hint = (
        """
<thinking>
    - Do S1-style scan; is there Actor+Action implying hidden coordination? Identify Effect scale.
    - If only one of Actor/Action is present or stance is reporting/satire -> prefer non.
    - State 1 cue that decides the label.
    (Max 2 sentences; do NOT quote the document.)
</thinking>
        """.strip()
        if include_cot
        else ""
    )

    markers_xml = "<extracted_markers>[]</extracted_markers>"
    if s1_output:
        markers_xml = (
            "<extracted_markers>\n"
            + json.dumps(s1_output, ensure_ascii=False, separators=(",", ":"))
            + "\n</extracted_markers>"
        )

    return f"""
{examples_xml}
{cot_hint}
<text_to_analyze>
{raw}
</text_to_analyze>

{markers_xml}
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
    Normalize an S1-style span (label/start/end[/text]) into the S2 schema:
      {"type","startIndex","endIndex","text"}
    - Clips to bounds
    - Recomputes 'text' slice from offsets (ignores any echoed text)
    """
    s = int(m.get("start", m.get("startIndex", 0)))
    e = int(m.get("end", m.get("endIndex", s)))
    s = max(0, min(s, len(txt)))
    e = max(s, min(e, len(txt)))
    return {
        "type": (m.get("type") or m.get("label")),
        "startIndex": s,
        "endIndex": e,
        "text": txt[s:e],
    }
