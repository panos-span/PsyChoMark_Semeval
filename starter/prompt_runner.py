#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
starter/prompt_runner.py

A clean runner that calls Bedrock via Chat(add_system/add_user/generate),
reusing the big prompt/plumbing from starter/prompt_sweep_joint.py.

Features:
- EDA artifacts (priors, conflicts, boundary, fewshots) with snapshotting
- S1 XML prompts + boundary/policy injections + fewshots (incl. negatives)
- S1 span post-processing: token snap, merge gaps, dedup, conflict NMS
- S2 XML prompts with self-consistency and probability coercion
- Auto threshold tuning on dev probs
- Prompt previews/snapshots + Codabench outputs + per-tech ZIPs
"""

import argparse
import json
import logging
import pathlib
import random
import re
import sys
import zipfile
from collections import defaultdict
from typing import Any, Dict, List

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# --- Bedrock Chat (Sonnet 4.5 wrapper) ---
from src.psycomark.llm.bedrock_chat import BedrockChat

# --- Reuse the largest runner's utilities / builders ---
# (We import only what we need to avoid code duplication.)
from starter.prompt_sweep_joint import (  # noqa: E402
    _load_prompt_artifacts,
    _snapshot_artifacts,
    postprocess_s1_spans,
    read_jsonl,
    write_jsonl,
    _save_prompt_bundle,
    _tokenize_eval,
    _valid_span,
    _snap_to_tokens,
    _prior_dist,
    _save_prompt_meta,
    _to_codabench_s1,
    _pick_balanced_s1_fewshots,
    _pick_balanced_s2_fewshots,
    tune_threshold_dev
)

from starter.prompt_builder import (  # noqa: E402
    build_s1_system,
    build_s1_user,
    extract_answer_json,
    build_s2_prompts_adapter,
    validate_and_repair_s1_spans
)



# ---------- tiny helpers you asked to include locally ----------
_LIST_RE = re.compile(r"\[.*?\]", re.S)

import pathlib as _pathlib
import os


# ---------- .env loader (no deps) ----------
def _load_dotenv_into_environ():
    root = _pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    # Map non-standard names to AWS_* so boto3 sees them
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()


def list_json_extract(s: str) -> list:
    """
    Extract the first top-level JSON array from a string; return [] if missing/bad.
    """
    if not isinstance(s, str):
        return []
    m = _LIST_RE.search(s)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, list) else []
    except Exception:
        return []


_CANON_LABEL = {
    k.lower(): k for k in ["Actor", "Action", "Effect", "Victim", "Evidence"]
}

def _all_occurrences(haystack: str, needle: str):
    if not needle:
        return
    i = 0
    while True:
        i = haystack.find(needle, i)
        if i == -1:
            break
        yield i
        i += 1

def align_text_span(item: dict, text: str, priors: dict | None):
    lab = (item.get("label") or "").strip()
    raw = (item.get("text") or "").strip()
    if not lab or not raw or not text:
        return None

    # exact matches for substring
    cand_starts = list(_all_occurrences(text, raw))
    if not cand_starts:
        # legacy fallback if model accidentally emitted start/end
        s = item.get("start")
        e = item.get("end")
        if isinstance(s, int) and isinstance(e, int) and e > s:
            toks = _tokenize_eval(text)
            snapped = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
            return snapped if snapped and _valid_span(text, snapped["start"], snapped["end"]) else None
        return None

    # pick candidate: prefer provided start hint if reasonable, else prior-closest
    hinted = item.get("start")
    if isinstance(hinted, int):
        chosen = min(cand_starts, key=lambda s: abs(s - hinted))
    else:
        L = len(text)
        def prior_cost(s0):
            return _prior_dist(priors or {}, lab, s0, s0 + len(raw), L)
        chosen = min(cand_starts, key=prior_cost)

    start, end = chosen, chosen + len(raw)
    toks = _tokenize_eval(text)
    snapped = _snap_to_tokens({"label": lab, "start": start, "end": end}, toks)
    if not snapped or not _valid_span(text, snapped["start"], snapped["end"]):
        return None
    return snapped  # {"label","start","end"}

def align_and_normalize_s1(model_items: list[dict], text: str, priors: dict | None):
    """
    Accepts items from <answer>:
      - new schema: {"label","text",["start"]}
      - legacy: {"label","start","end"} or {"type","startIndex","endIndex"}
    Returns [{"label","start","end"}], snapped to token boundaries.
    """
    if not isinstance(model_items, list):
        return []
    out = []

    # Preferred: text-based items
    for m in model_items:
        if m.get("text"):
            a = align_text_span(m, text, priors)
            if a:
                out.append(a)

    # Fallback: legacy spans if no text fields made it through
    if not out:
        for m in model_items:
            lab = (m.get("label") or m.get("type") or "").strip()
            s = m.get("start", m.get("startIndex"))
            e = m.get("end", m.get("endIndex"))
            try:
                s, e = int(s), int(e)
            except Exception:
                continue
            if e <= s:
                continue
            toks = _tokenize_eval(text)
            a = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
            if a and _valid_span(text, a["start"], a["end"]):
                out.append(a)

    # Canonicalize
    return [
        {"label": m["label"], "start": int(m["start"]), "end": int(m["end"])}
        for m in out
        if m.get("end", 0) > m.get("start", 0)
    ]


def clip_and_validate(spans: list, text: str) -> list:
    """
    Normalize S1 spans to [{"label", "start", "end"}], clip to text bounds, drop invalid.
    Accepts either {label,start,end} or {type,startIndex,endIndex}.
    """
    L = len(text or "")
    out = []
    for m in spans or []:
        lab_raw = (m.get("label") or m.get("type") or "").strip()
        lab = _CANON_LABEL.get(lab_raw.lower())
        if not lab:
            continue
        s = m.get("start", m.get("startIndex"))
        e = m.get("end", m.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out

def align_text_markers_to_spans(doc_text: str, items: List[dict], case_sensitive: bool = False) -> List[dict]:
    """
    Convert [{"label":..., "text":...}] to [{"label":..., "start":int, "end":int}]
    by exact substring matching (prefers non-overlapping, left-to-right).
    - If items already have start/end, they’re passed through.
    - If multiple matches exist, choose the first non-overlapping occurrence.
    """
    if not doc_text or not items:
        return []
    spans = []
    hay = doc_text if case_sensitive else doc_text.lower()

    # track used character positions to avoid overlaps
    used = [False] * len(doc_text)

    def _place_span(lab: str, s: int, e: int):
        if e <= s: 
            return False
        # avoid overlaps
        if any(used[i] for i in range(s, e)):
            return False
        spans.append({"label": lab, "start": s, "end": e})
        for i in range(s, e):
            used[i] = True
        return True

    for it in items:
        lab = (it.get("label") or it.get("type") or "").strip()
        if not lab:
            continue

        # already a span? keep it (we'll still snap/dedup downstream)
        if "start" in it and "end" in it:
            try:
                s, e = int(it["start"]), int(it["end"])
                _place_span(lab, s, e)
            except Exception:
                pass
            continue

        txt = (it.get("text") or "").strip()
        if not txt:
            continue

        needle = txt if case_sensitive else txt.lower()
        pos = 0
        placed = False
        while True:
            idx = hay.find(needle, pos)
            if idx < 0:
                break
            j = idx + len(needle)
            if _place_span(lab, idx, j):
                placed = True
                break
            pos = idx + 1

        # (optional) fallback: try case-sensitive if insensitive failed
        if not placed and not case_sensitive:
            pos = 0
            while True:
                idx = doc_text.find(txt, pos)
                if idx < 0:
                    break
                j = idx + len(txt)
                if _place_span(lab, idx, j):
                    break
                pos = idx + 1

    return spans


from collections import Counter
import random, json, re, logging

from collections import Counter
import json, random, re, logging

from collections import Counter
import json, re, random, logging

def _safe_label(x: str, allow_cant_tell: bool) -> str:
    x = (x or "").strip().lower()
    valid = {"conspiracy", "non"} | ({"cant_tell"} if allow_cant_tell else set())
    return x if x in valid else "non"

def run_s2_self_consistent(
    *,
    text: str,
    markers: list,
    fewshots_pool: list,
    tech: str,
    policy_text: str | None,      # kept in signature (ignored in new S2)
    prompt_arts: dict | None,     # kept in signature (ignored in new S2)
    model_id: str | None,
    region: str | None,
    bedrock,                      # BedrockChat stateless client
    max_tokens: int,
    base_temperature: float,
    sc_temperature: float,
    sc_runs: int,
    allow_cant_tell: bool = False  # NEW: expose cant_tell policy
):
    """
    S2 self-consistency with vote aggregation.
    - Rebuild prompts per run (few-shot order shuffled).
    - No model probabilities required; we derive pseudo-probabilities from vote share.
    Returns: (final_label, p_con, p_non, first_sys, first_user)
    """
    temp = sc_temperature if sc_runs > 1 else base_temperature
    preds, labels = [], []
    first_sys = first_user = None

    for r in range(max(1, sc_runs)):
        fs = fewshots_pool[:] if fewshots_pool else []
        if fs:
            random.shuffle(fs)

        sys_prompt, user_prompt = build_s2_prompts_adapter(
            text=text, markers=markers, fewshots=fs, tech=tech, allow_cant_tell=allow_cant_tell
        )
        if r == 0:
            first_sys, first_user = sys_prompt, user_prompt

        out = bedrock.chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temp
        )

        # Prefer extracting from <answer>...</answer>; fallback to last JSON object
        js = None
        m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", out or "", re.S)
        if m:
            try:
                js = json.loads(m.group(1))
            except Exception:
                js = None
        if js is None:
            m = re.findall(r"(\{.*?\})", out or "", re.S)
            if m:
                try:
                    js = json.loads(m[-1])
                except Exception:
                    js = None

        lbl = _safe_label((js or {}).get("label"), allow_cant_tell=allow_cant_tell)
        labels.append(lbl)
        preds.append({"label": lbl})

    # Majority vote
    maj_label, maj_count = Counter(labels).most_common(1)[0]
    vote_share = maj_count / max(1, len(labels))

    # Map vote share to pseudo-probs so your caller stays compatible
    if maj_label == "conspiracy":
        p_con, p_non = vote_share, 1.0 - vote_share
    elif maj_label == "non":
        p_con, p_non = 1.0 - vote_share, vote_share
    else:  # cant_tell
        p_con, p_non = 0.5, 0.5

    return maj_label, round(p_con, 6), round(p_non, 6), first_sys, first_user



# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Run PsyCoMark S1→S2 with Claude Sonnet on Bedrock (Chat API)."
    )
    ap.add_argument("--test-file-s1", default="data/raw/dev_rehydrated.jsonl")
    ap.add_argument("--test-file-s2", default="data/raw/dev_rehydrated.jsonl")
    ap.add_argument("--eda-root", default=None, help="Folder with EDA artifacts.")
    ap.add_argument(
        "--techniques",
        default="fs_policy_boundary_cot_sc5",
        help="Comma list, e.g., fs_policy_boundary_cot_sc5, fs_neg_cot_sc10, zs_cot_sc5",
    )
    ap.add_argument("--model-id", default=None, help="Override Bedrock model id.")
    ap.add_argument("--region", default=None, help="AWS region (e.g., eu-central-1).")
    ap.add_argument("--max-tokens-s1", type=int, default=1500)
    ap.add_argument("--max-tokens-s2", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--s2-self-consistency", type=int, default=None)
    ap.add_argument("--s1-iou", type=float, default=0.5)
    ap.add_argument("--out-root", default="runs/joint_llm_v4")
    ap.add_argument("--pp-merge-gap", type=int, default=1)
    ap.add_argument("--pp-dedup-iou", type=float, default=0.90)
    ap.add_argument("--pp-conflict-iou", type=float, default=0.50)
    ap.add_argument("--max-markers-per-label", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--s2-thresh", default="auto")
    ap.add_argument(
        "--save-prompts",
        choices=["none", "sample", "all"],
        default="sample",
        help="Save prompts to runs/<tech>/prompts (sample=first 3 docs/task).",
    )
    ap.add_argument("--print-prompts-preview", action="store_true", default=True)
    args = ap.parse_args()

    print("=== JOINT PROMPT (Chat API) START ===")
    random.seed(42)
    from os import getenv

    model_id = (
        args.model_id or getenv("MODEL_ID")
    )
    region = args.region or getenv("AWS_DEFAULT_REGION") or "us-east-1"
    
    # Initialize bedrock Chat
    bc = BedrockChat(
        model_id=(model_id or "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        region_name=(region or "eu-central-1"),
    )


    # ----- Load artifacts (priors/conflicts/boundary/fewshots) -----
    s1_policy = s2_policy = ""
    s1_shots: List[dict] = []
    s2_shots: List[dict] = []
    boundary = ""
    prompt_arts: Dict[str, Any] = {}

    if args.eda_root:
        eda = pathlib.Path(args.eda_root)
        print(f"Loading EDA artifacts from: {eda}")
        prompt_arts = _load_prompt_artifacts(eda)  # robust loader/snapshots
        # fallbacks for fewshots
        s1_shots = (prompt_arts.get("fewshot_bank") or {}).get("s1", [])[:8]
        s2_shots = (prompt_arts.get("fewshot_bank") or {}).get("s2", [])[:10]
        # lightweight policy/boundary strings (already formatted by your builder)
        s1_policy = (prompt_arts.get("fewshot_policy") or {}).get("s1_policy", "") or ""
        s2_policy = (prompt_arts.get("fewshot_policy") or {}).get("s2_policy", "") or ""
        boundary = (prompt_arts.get("boundary_prompts") or {}).get("note", "") or ""

    # ----- Load data -----
    rows_s1 = list(read_jsonl(args.test_file_s1))
    rows_s2 = list(read_jsonl(args.test_file_s2))
    if args.limit_docs is not None:
        rows_s1 = rows_s1[: args.limit_docs]
        rows_s2 = rows_s2[: args.limit_docs]
        logging.info(
            f"Limiting to first {args.limit_docs} docs for S1 ({len(rows_s1)}) and S2 ({len(rows_s2)})"
        )
    id2doc_s2 = {(r.get("_id") or r.get("doc_id")): r for r in rows_s2}

    # ----- Techniques -----
    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    for tech in techniques:
        print(f"\n=== JOINT S1→S2 :: {tech} ===")

        has_fs = "fs" in tech
        has_neg = "neg" in tech
        use_cot = "cot" in tech
        use_boundary = "boundary" in tech
        use_policy = "policy" in tech
        sc_match = re.search(r"sc(\d+)", tech)
        sc_n = int(sc_match.group(1)) if sc_match else 1
        if isinstance(args.s2_self_consistency, int) and args.s2_self_consistency >= 1:
            sc_n = args.s2_self_consistency

        # few-shot pools (optionally adding a negative S1/S2 example)
        fs_s1 = s1_shots[:] if has_fs else []
        fs_s2 = s2_shots[:] if has_fs else []
        if has_fs and has_neg and prompt_arts:
            s1_neg = [
                ex
                for ex in (prompt_arts.get("fewshot_bank") or {}).get("s1", [])
                if isinstance(ex.get("spans"), list) and len(ex["spans"]) == 0
            ][:1]
            s2_neg = [
                ex
                for ex in (prompt_arts.get("fewshot_bank") or {}).get("s2", [])
                if (ex.get("gold") or {}).get("label") == "non"
            ][:1]
            fs_s1 += s1_neg
            fs_s2 += s2_neg

        tech_dir = pathlib.Path(args.out_root) / tech
        (tech_dir / "s1").mkdir(parents=True, exist_ok=True)
        (tech_dir / "s2").mkdir(parents=True, exist_ok=True)
        (tech_dir / "prompts").mkdir(parents=True, exist_ok=True)
        if prompt_arts:
            _snapshot_artifacts(prompt_arts, tech_dir)

        s1_sub = tech_dir / "s1" / "submission.jsonl"
        s1_pruned_sub = tech_dir / "s1" / "submission_pruned.jsonl"
        s2_sub = tech_dir / "s2" / "submission.jsonl"

        # save meta once
        _save_prompt_meta(
            tech_dir,
            {
                "tech": tech,
                "model_id": model_id,
                "region": args.region,
                "max_tokens_s1": args.max_tokens_s1,
                "max_tokens_s2": args.max_tokens_s2,
                "temperature": args.temperature,
                "sc_temperature": args.sc_temperature,
                "s1_policy_used": bool(s1_policy) and use_policy,
                "s2_policy_used": bool(s2_policy) and use_policy,
                "boundary_note_used": bool(boundary) and use_boundary,
            },
        )

        printed_s1_preview = False
        printed_s2_preview = False

        # ===== S1 =====
        s1_out_rows = []
        s1_pruned_rows = []
        id2markers = {}
        total_raw = total_valid = total_pruned = 0
        saved_s1 = 0

        for rec in rows_s1:
            _id = rec.get("_id") or rec.get("doc_id")
            txt = (rec.get("text") or "").strip()
            if not txt:
                id2markers[_id] = []
                s1_out_rows.append({"_id": _id, "markers": []})
                s1_pruned_rows.append({"_id": _id, "markers": []})
                continue

            fewshots_s1 = (
                _pick_balanced_s1_fewshots(fs_s1, k=min(8, len(fs_s1))) if fs_s1 else []
            )

            # Build prompts (XML + boundary/policy/priors loaded in s1_prompt via prompt_arts)
            s1_system = build_s1_system(
                priors=(prompt_arts.get("priors_prompt") or {}),
                conflict_pairs=[tuple(p) for p in (prompt_arts.get("conflicts") or {}).get("pairs", [])],
                boundary_note=(boundary if use_boundary else None),
                policy_text=(s1_policy if use_policy else None),
                include_cot=use_cot,
            )
            s1_user = build_s1_user(
                text_input=txt,
                s1_fewshots=(fewshots_s1 if has_fs else []),
                include_cot=use_cot,
            )
            sys_blocks, user_block = [s1_system], s1_user
            n_samples, temp = 1, args.temperature

            # Preview/save
            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s1 < 3
            ):
                _save_prompt_bundle(tech_dir, "s1", str(_id), sys_blocks, user_block)
                saved_s1 += 1
            if args.print_prompts_preview and not printed_s1_preview:
                print(
                    "[S1 prompt preview]\nSYSTEM:\n"
                    + "\n\n---\n\n".join(sys_blocks)
                    + "\n\nUSER:\n"
                    + user_block
                )
                printed_s1_preview = True

            system_str = "\n\n".join(sys_blocks)  # MUST be a single string
            # --- Bedrock: stateless call with a single system string ---
            try:
                s1_raw = bc.chat(
                    system_prompt=system_str,
                    user_prompt=user_block,
                    max_tokens=args.max_tokens_s1,
                    temperature=args.temperature,
                    # stop_sequences=["</answer>"],  # optional: helps stop right after JSON
                )

                # prefer extracting JSON inside <answer>…</answer>
                arr = extract_answer_json(s1_raw) or []
                # align our new schema {"label","text",["start"]} -> {"label","start","end"}
                #spans = clip_and_validate(arr, txt)  # your existing bounds/token snapper still fine
                
                # NEW: validate/repair hybrid
                spans = validate_and_repair_s1_spans(
                    arr, txt, win=16, use_tokens=True
                )
                
                #spans = align_and_normalize_s1(
                #    arr,
                #    txt,
                #    priors=(prompt_arts.get("priors_prompt") or {}),
                #)
            except Exception as e:
                logging.warning(f"[S1] generate() failed -> empty spans: {e}")
                spans = []


            total_raw += len(spans)

            # Post-process to evaluator-aligned spans
            markers = postprocess_s1_spans(
                text=txt,
                spans=spans,
                priors=(prompt_arts.get("priors_prompt") or {}),
                merge_gap=args.pp_merge_gap,
                dedup_iou=args.pp_dedup_iou,
                conflict_iou=args.pp_conflict_iou,
            )
            total_valid += len(markers)

            # prune per label for S2 prompt size
            by_lab = defaultdict(list)
            for m in markers:
                by_lab[m["type"]].append(m)

            pruned = []
            k = args.max_markers_per_label

            def _start_end(m):
                s = m.get("start", m.get("startIndex", 0))
                e = m.get("end", m.get("endIndex", s))
                return int(s), int(e)

            for lab, arr in by_lab.items():
                arr_sorted = sorted(
                    arr, key=lambda m: (-(m["endIndex"] - m["startIndex"]))
                )
                pruned.extend(arr_sorted[:k])

            # normalize schema for S2 prompt
            def _to_s2_marker(m):
                s, e = _start_end(m)
                return {
                    "type": m.get("type") or m.get("label"),
                    "startIndex": s,
                    "endIndex": e,
                    "text": txt[s:e],
                }

            id2markers[_id] = [_to_s2_marker(m) for m in pruned]
            total_pruned += len(id2markers[_id])

            s1_out_rows.append({"_id": _id, "markers": markers})
            s1_pruned_rows.append({"_id": _id, "markers": id2markers[_id]})

        write_jsonl(s1_sub, s1_out_rows)
        write_jsonl(s1_pruned_sub, s1_pruned_rows)
        print(f"S1 done -> {s1_sub}")
        print(f"S1 pruned-for-S2 -> {s1_pruned_sub}")
        print(
            f"S1 debug: spans raw/valid/pruned = {total_raw}/{total_valid}/{total_pruned}"
        )

        # ===== S2 =====
        s2_out_rows, s2_prob_rows = [], []
        saved_s2 = 0
        printed_s2_preview = False

        for _id, doc2 in id2doc_s2.items():
            txt = (doc2.get("text") or "").strip()
            mks = id2markers.get(_id, [])

            # --- Build few-shots for this doc once (pool) ---
            fewshots_s2 = _pick_balanced_s2_fewshots(fs_s2, k=min(8, len(fs_s2))) if fs_s2 else []

            # --- Self-consistency (uses BedrockChat + XML builders) ---
            sc_match = re.search(r"sc(\d+)", tech)
            sc_n = int(sc_match.group(1)) if sc_match else 1
            if isinstance(args.s2_self_consistency, int) and args.s2_self_consistency >= 1:
                sc_n = args.s2_self_consistency

            final_label, avg_con, avg_non, dbg_sys, dbg_user = run_s2_self_consistent(
                text=txt,
                markers=mks,
                fewshots_pool=fewshots_s2,
                tech=tech,
                policy_text=(s2_policy if use_policy else None),
                prompt_arts=prompt_arts,
                bedrock=bc,                                # pass the shared stateless client
                max_tokens=args.max_tokens_s2,
                base_temperature=args.temperature,
                sc_temperature=args.sc_temperature,
                sc_runs=max(1, sc_n),
                allow_cant_tell=False 
            )

            # (Optional) save a sample of prompts
            if args.save_prompts == "all" or (args.save_prompts == "sample" and saved_s2 < 3):
                _save_prompt_bundle(tech_dir, "s2", str(_id), [dbg_sys], dbg_user)
                saved_s2 += 1
            if args.print_prompts_preview and not printed_s2_preview:
                print("[S2 prompt preview]\nSYSTEM:\n" + dbg_sys + "\n\nUSER:\n" + dbg_user)
                printed_s2_preview = True

            # collect outputs
            s2_out_rows.append({
                "_id": _id,
                "conspiracy": ("Yes" if final_label == "conspiracy" else "No"),
            })
            s2_prob_rows.append({
                "_id": _id,
                "label": final_label,
                "p_conspiracy": round(avg_con, 6),
                "p_non": round(avg_non, 6),
            })

            # self-consistency
            #runs = max(1, int(sc_n))
            #temp2 = args.sc_temperature if runs > 1 else args.temperature
            #preds, label_votes = [], []

            #for _ in range(runs):
            #    chat_s2 = Chat(
            #        model_id=(model_id or Chat.SONNET_45_MODEL_ID),
            #        region=region,
            #        max_tokens=args.max_tokens_s2,
            #        temperature=temp2,
            #    )
            #    for s in sys_blocks:
            #        chat_s2.add_system(s)
            #    chat_s2.add_user(user_block)
            #    try:
            #        out = chat_s2.generate()
            #        # permissive JSON object extraction
            #        m = re.search(r"\{.*\}", out, re.S)
            #        js = (
            #            json.loads(m.group(0))
            #            if m
            #            else {"label": "non", "confidence": 0.55}
            #        )
            #    except Exception as e:
            #        logging.warning(f"[S2] generate() failed -> default non: {e}")
            #        js = {"label": "non", "confidence": 0.55}
            #
            #    # coerce probabilities + label
            #    label = (js.get("label") or "non").strip().lower()
            #    p_con = js.get("p_conspiracy")
            #    p_non = js.get("p_non")
            #
            #    def _flt(x):
            #        try:
            #            return float(x)
            #        except Exception:
            #            return None
            #
            #    p_con = _flt(p_con)
            #    p_non = _flt(p_non)
            #    if p_con is None or p_non is None:
            #        conf = _flt(js.get("confidence")) or (
            #            0.9 if label == "conspiracy" else 0.1
            #        )
            #        if label == "conspiracy":
            #            p_con, p_non = conf, 1.0 - conf
            #        else:
            #            p_con, p_non = 1.0 - conf, conf
            #    s = p_con + p_non
            #    if s <= 0:
            #        p_con = p_non = 0.5
            #    else:
            #        p_con, p_non = p_con / s, p_non / s
            #    if label not in ("conspiracy", "non"):
            #        label = "conspiracy" if p_con >= p_non else "non"
            #
            #    preds.append({"label": label, "p_conspiracy": p_con, "p_non": p_non})

            # average probs; final label = argmax
            #avg_con = sum(p["p_conspiracy"] for p in preds) / len(preds)
            #avg_non = sum(p["p_non"] for p in preds) / len(preds)
            #tot = max(1e-9, avg_con + avg_non)
            #avg_con, avg_non = avg_con / tot, avg_non / tot
            #final_label = "conspiracy" if avg_con >= avg_non else "non"

            #s2_out_rows.append(
            #    {
            #        "_id": _id,
            #        "conspiracy": ("Yes" if final_label == "conspiracy" else "No"),
            #    }
            #)
            #s2_prob_rows.append(
            #    {
            #        "_id": _id,
            #        "label": final_label,
            #        "p_conspiracy": round(avg_con, 6),
            #        "p_non": round(avg_non, 6),
            #    }
            #)

        # write raw probs, then choose threshold
        probs_path = tech_dir / "s2" / "probs.jsonl"
        write_jsonl(probs_path, s2_prob_rows)

        thr = 0.50
        if isinstance(args.s2_thresh, str) and args.s2_thresh.lower() == "auto":
            best_t, best_f1, stats = tune_threshold_dev(rows_s2, s2_prob_rows)
            if stats.get("n", 0) > 0:
                thr = best_t
                print(
                    f"[S2] tuned threshold on dev: t={thr:.2f} (dev f1={best_f1:.3f}, mean_p={stats['mean_p']:.3f})"
                )
            else:
                print("[S2] no gold labels; keep default thr=0.50")
        else:
            try:
                thr = float(args.s2_thresh)
            except Exception:
                logging.warning(
                    f"[S2] invalid --s2-thresh={args.s2_thresh}; using 0.50"
                )
                thr = 0.50

        # final submission using chosen threshold
        pred2 = [
            {
                "_id": r["_id"],
                "conspiracy": ("Yes" if r["p_conspiracy"] >= thr else "No"),
            }
            for r in s2_prob_rows
        ]
        write_jsonl(s2_sub, pred2)
        print(f"S2 done -> {s2_sub}")

        # top-level Codabench files
        codabench_s1 = [
            {"_id": r["_id"], "markers": _to_codabench_s1(r.get("markers", []))}
            for r in s1_out_rows
        ]
        write_jsonl("submission_s1.jsonl", codabench_s1)
        write_jsonl("submission_s2.jsonl", pred2)
        print("Wrote top-level submissions: submission_s1.jsonl, submission_s2.jsonl")

        # per-tech ZIP
        try:
            submissions_dir = pathlib.Path("submissions")
            submissions_dir.mkdir(parents=True, exist_ok=True)
            zip_path = submissions_dir / f"submission_{tech}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if pathlib.Path("submission_s1.jsonl").exists():
                    zf.write("submission_s1.jsonl", arcname="submission_s1.jsonl")
                if pathlib.Path("submission_s2.jsonl").exists():
                    zf.write("submission_s2.jsonl", arcname="submission_s2.jsonl")
                if s1_sub.exists():
                    zf.write(s1_sub, arcname=f"{tech}/s1/submission.jsonl")
                if s1_pruned_sub.exists():
                    zf.write(
                        s1_pruned_sub, arcname=f"{tech}/s1/submission_pruned.jsonl"
                    )
                if s2_sub.exists():
                    zf.write(s2_sub, arcname=f"{tech}/s2/submission.jsonl")
                if probs_path.exists():
                    zf.write(probs_path, arcname=f"{tech}/s2/probs.jsonl")
            print(f"Packaged ZIP -> {zip_path}")
        except Exception as e:
            logging.warning(f"[ZIP] Failed to create technique ZIP: {e}")


if __name__ == "__main__":
    main()
