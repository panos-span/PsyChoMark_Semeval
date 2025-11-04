#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_runner.py  — clean, defensive runner for PsyCoMark S1/S2

- Loads train/dev, fewshot_bank.json
- Picks per-doc few-shots (balanced, deterministic)
- Builds prompts via prompt_builder.py
- Calls LLM (BedrockChat or Chat) and validates outputs
- Never crashes on bad few-shot entries or bad generations

Fixes the common crash:
  [S1] generate() failed -> empty spans: 'str' object has no attribute 'get'
by strictly normalizing few-shot spans and by validating generations.

Author: you
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---- Paths / imports ---------------------------------------------------------
# _THIS = pathlib.Path(__file__).resolve()
# ROOT = _THIS.parents[3]
# SRC = ROOT / "src"
# for p in (str(ROOT), str(SRC)):
#    if p not in sys.path:
#        sys.path.insert(0, p)


# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Optional Bedrock wrapper (ok if missing for dry runs / unit tests)
from src.psycomark.llm.bedrock_chat import BedrockChat as Chat


# Your prompt builder (we rely on these)
from starter.prompt_builder import (
    build_s1_system,
    build_s1_user,
    build_s1_verify_system,
    load_artifacts,
    build_s1_verify_user,
    build_s2_system,
    build_s1_prompts_adapter,
    build_s2_prompts_adapter,  # <-- FIX: Import the correct S2 adapter
    build_s2_user,
)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("prompt_runner")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


# ---------- .env loader (no deps) ----------
def _load_dotenv_into_environ():
    root = pathlib.Path(__file__).resolve().parents[1]
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

# ------------------------------------------------------------------------------
# Small utils
# ------------------------------------------------------------------------------
ALLOWED_S1 = {"Actor", "Action", "Effect", "Evidence", "Victim"}
_WS = re.compile(r"\s+")
JSON_GUESS = re.compile(r"\{|\[")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: pathlib.Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _tokset(s: str) -> set:
    return set(t for t in re.split(_WS, s.strip()) if t)


def _dedup_by_text(items: List[dict], key="text", min_jaccard_keep=0.80) -> List[dict]:
    """Keep items whose token set Jaccard < threshold against already kept."""
    seen, out = [], []
    for ex in items:
        txt = (ex.get(key) or "").strip()
        if not txt:
            continue
        toks = _tokset(txt)
        ok = True
        for t2 in seen:
            inter = len(toks & t2)
            union = len(toks | t2) or 1
            j = inter / union
            if j >= min_jaccard_keep:
                ok = False
                break
        if ok:
            seen.append(toks)
            out.append(ex)
    return out


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Few-shot bank normalization
# ------------------------------------------------------------------------------
def _norm_s1_span(m: Any) -> Optional[dict]:
    """
    Normalize ONE S1 span to {type,startIndex,endIndex,text} OR return None.
    Drop anything malformed (strings, missing fields, bad bounds).
    """
    if not isinstance(m, dict):
        return None
    t = (m.get("type") or m.get("label") or "").strip()
    if t not in ALLOWED_S1:
        return None
    s = _safe_int(m.get("startIndex", m.get("start")), -1)
    e = _safe_int(m.get("endIndex", m.get("end")), -1)
    txt = (m.get("text") or "").strip()
    if s < 0 or e <= s or not txt:
        return None
    return {"type": t, "startIndex": s, "endIndex": e, "text": txt}


def _norm_s1_entry(ex: Any) -> Optional[dict]:
    """
    Normalize ONE S1 few-shot example:
      {"text":str, "answer":[spans...], "_id":..., "subreddit":...}
    Drop example if text empty OR all spans invalid.
    """
    if not isinstance(ex, dict):
        return None
    txt = (ex.get("text") or "").strip()
    if not txt:
        return None
    ans = ex.get("answer") or ex.get("spans") or []
    if not isinstance(ans, list):
        # support single span dict
        ans = [ans] if isinstance(ans, dict) else []
    spans = []
    for m in ans:
        nm = _norm_s1_span(m)
        if nm:
            spans.append(nm)
    # S1 few-shot can include negatives (zero spans) — allow them.
    return {
        "text": txt,
        "answer": spans,
        "_id": ex.get("_id") or ex.get("doc_id"),
        "subreddit": ex.get("subreddit"),
    }


def _norm_s2_entry(ex: Any) -> Optional[dict]:
    """
    Keep S2 schema as {"text":..., "answer":{"label": "conspiracy|non", "rationale": str}}
    Drop if text empty or label invalid.
    """
    if not isinstance(ex, dict):
        return None
    txt = (ex.get("text") or "").strip()
    if not txt:
        return None
    a = ex.get("answer") or {}
    if isinstance(a, str):
        a = {"label": a, "rationale": ""}
    if not isinstance(a, dict):
        return None
    lab = (a.get("label") or ex.get("label") or "").strip().lower()
    if lab not in {"conspiracy", "non"}:
        return None
    rat = (a.get("rationale") or "").strip()
    return {
        "text": txt,
        "answer": {"label": lab, "rationale": rat},
        "_id": ex.get("_id") or ex.get("doc_id"),
        "subreddit": ex.get("subreddit"),
    }


def _load_fewshot_bank(path: pathlib.Path) -> dict:
    """
    Load fewshot_bank.json with strict normalization so we never crash on malformed data.
    """
    raw = _read_json(path)
    raw_s1 = list(raw.get("s1") or [])
    raw_s2 = list(raw.get("s2") or [])
    s1 = []
    s2 = []
    for ex in raw_s1:
        ne = _norm_s1_entry(ex)
        if ne:
            s1.append(ne)
    for ex in raw_s2:
        ne = _norm_s2_entry(ex)
        if ne:
            s2.append(ne)
    if not s1:
        LOG.warning("[fewshots] S1 bank empty after normalization.")
    if not s2:
        LOG.warning("[fewshots] S2 bank empty after normalization.")
    # light dedup
    s1 = _dedup_by_text(s1, key="text", min_jaccard_keep=0.90)
    s2 = _dedup_by_text(s2, key="text", min_jaccard_keep=0.90)
    return {"s1": s1, "s2": s2}


# ------------------------------------------------------------------------------
# Per-doc few-shot selection (balanced, deterministic)
# ------------------------------------------------------------------------------
def _is_positive(ex: dict) -> bool:
    return bool(ex.get("answer"))


def _has_victim(ex: dict) -> bool:
    return any(m.get("type") == "Victim" for m in ex.get("answer") or [])


def _has_ae(ex: dict) -> bool:
    labs = {m.get("type") for m in (ex.get("answer") or [])}
    return "Action" in labs and "Effect" in labs


def _order_s1_examples(exs: List[dict]) -> List[dict]:
    # AE first, then Victim, then other positives, then negatives
    def key(ex):
        score = 0
        if _has_ae(ex):
            score -= 100
        if _has_victim(ex):
            score -= 10
        if _is_positive(ex):
            score -= 1
        return (score, (ex.get("_id") or ""))

    return sorted(exs, key=key)


def _select_s1_fewshots_for_doc(pool: List[dict], k: int = 8) -> List[dict]:
    if not pool:
        return []
    # deterministic shuffle to avoid global-order bias
    rng = random.Random(42)
    pool = list(pool)
    rng.shuffle(pool)
    base = pool[:k]
    # Ensure ≥2 positives
    pos = [ex for ex in base if _is_positive(ex)]
    if len(pos) < 2:
        extras = [ex for ex in pool if _is_positive(ex) and ex not in base]
        for ex in extras:
            if len(pos) >= 2:
                break
            # replace a negative if any, else append if space
            replaced = False
            for i, cur in enumerate(base):
                if not _is_positive(cur):
                    base[i] = ex
                    replaced = True
                    pos.append(ex)
                    break
            if not replaced and len(base) < k:
                base.append(ex)
                pos.append(ex)
    # Ensure ≥1 Victim
    if not any(_has_victim(ex) for ex in base):
        cand = next((ex for ex in pool if _has_victim(ex) and ex not in base), None)
        if cand:
            if base:
                base[-1] = cand
            else:
                base.append(cand)
    # Ensure ≥1 AE
    if not any(_has_ae(ex) for ex in base):
        cand = next((ex for ex in pool if _has_ae(ex) and ex not in base), None)
        if cand:
            if base:
                base[0] = cand
            else:
                base.append(cand)
    return _order_s1_examples(base[:k])


def _select_s2_fewshots_for_doc(pool: List[dict], k: int = 8) -> List[dict]:
    if not pool:
        return []
    rng = random.Random(42)
    pool = list(pool)
    rng.shuffle(pool)
    # try to keep label balance 50/50 if possible
    pos = [e for e in pool if (e.get("answer") or {}).get("label") == "conspiracy"]
    neg = [e for e in pool if (e.get("answer") or {}).get("label") == "non"]
    half = k // 2
    out = pos[:half] + neg[: (k - half)]
    if len(out) < k:
        rest = [e for e in pool if e not in out]
        out += rest[: (k - len(out))]
    return _dedup_by_text(out[:k], key="text", min_jaccard_keep=0.95)


# ------------------------------------------------------------------------------
# Model I/O
# ------------------------------------------------------------------------------
def _get_chat(model_id: Optional[str], region: Optional[str]):
    if Chat is None:
        LOG.warning("BedrockChat not available; returning None (dry run).")
        return None
    # env overrides
    mdl = (
        os.getenv("MODEL_ID")
        or model_id
        or "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    reg = os.getenv("AWS_DEFAULT_REGION") or region or "eu-central-1"
    LOG.info("Bedrock client ready | model=%s region=%s", mdl, reg)
    return Chat(model_id=mdl, region_name=reg)


def _extract_json_from_answer(raw: str) -> str:
    """
    Given a raw LLM answer, try to isolate the JSON array/object.
    Prefer <answer>...</answer> if present.
    """
    if not raw:
        return ""
    m = ANSWER_TAG_RE.search(raw)
    if m:
        return m.group(1).strip()
    # crude fallback: find the first JSON-looking section
    if JSON_GUESS.search(raw):
        # find first { or [
        i = min([i for i in [raw.find("{"), raw.find("[")] if i >= 0] or [0])
        return raw[i:].strip()
    return raw.strip()


def _safe_parse_s1_generation(raw: str) -> List[dict]:
    """
    Parse S1 model output into a list of valid spans.
    Drop malformed entries; never raise.
    Expected final objects: [{"label"/"type", "start"/"startIndex", "end"/"endIndex", "text"}...]
    """
    js = _extract_json_from_answer(raw)
    if not js:
        return []
    try:
        obj = json.loads(js)
    except Exception:
        return []
    if isinstance(obj, dict):
        # sometimes returned inside {"answer":[...]}
        if "answer" in obj and isinstance(obj["answer"], list):
            obj = obj["answer"]
        else:
            obj = [obj]
    if not isinstance(obj, list):
        return []
    out = []
    for m in obj:
        nm = _norm_s1_span(m)
        if nm:
            # convert back to {"label", "start", "end", "text"} if your downstream expects that,
            # but we'll keep normalized for consistency.
            out.append(nm)
    return out


def _print_prompt_preview(
    task_name: str, system_prompt: str, user_prompt: str, max_chars: int = 2000
):
    sep = "=" * 80
    print(
        f"\n{sep}\n[{task_name}] SYSTEM PROMPT (preview)\n{sep}\n{system_prompt[:max_chars]}"
    )
    print(
        f"\n{sep}\n[{task_name}] USER PROMPT (preview)\n{sep}\n{user_prompt[:max_chars]}\n"
    )


def _print_answer_preview(
    task_name: str, doc_id: str, raw_answer: str, ordinal: int, max_chars: int = 1200
):
    print(f"[{task_name}] #{ordinal:02d} doc={doc_id}  raw_answer↓")
    print((raw_answer or "").strip()[:max_chars] + ("\n" if raw_answer else ""))


def _normalize_span_key(m: dict) -> tuple:
    # key for voting/merging
    t = (m.get("type") or "").strip()
    s = int(m.get("startIndex", m.get("start", -1)))
    e = int(m.get("endIndex", m.get("end", -1)))
    txt = (m.get("text") or "").strip()
    return (t, s, e, txt)


_LABEL_ORDER = {"Actor": 0, "Action": 1, "Effect": 2, "Victim": 3, "Evidence": 4}


def _normalize_span(m: dict):
    lab = (m.get("label") or m.get("type") or "").strip()
    s = m.get("start", m.get("startIndex"))
    e = m.get("end", m.get("endIndex"))
    try:
        s, e = int(s), int(e)
    except Exception:
        return None
    if not lab or e <= s or s < 0:
        return None
    txt = m.get("text")
    # Optional: trim text if it mismatches bounds (defensive)
    if isinstance(txt, str) and len(txt) > 0:
        txt = txt
    else:
        txt = None
    return lab, s, e, txt


def _merge_sc_spans(samples: list[list[dict]], top_k: int = 4) -> list[dict]:
    """
    Majority vote over identical spans by (type, startIndex, endIndex).
    Returns up to top_k spans ranked by:
      1) vote desc, 2) span length desc, 3) start asc, 4) label order asc.
    (No per-role cap here; apply any caps later in the pipeline.)
    """
    from collections import Counter, defaultdict

    votes = Counter()
    texts = defaultdict(Counter)  # key -> Counter of candidate texts

    for spans in samples or []:
        for m in spans or []:
            norm = _normalize_span(m)
            if not norm:
                continue
            lab, s, e, txt = norm
            key = (lab, s, e)
            votes[key] += 1
            if isinstance(txt, str) and txt:
                texts[key][txt] += 1

    if not votes:
        return []

    def rank_key(item):
        (lab, s, e), v = item
        length = e - s
        return (
            v,
            length,
            -s,
            -(10 - _LABEL_ORDER.get(lab, 9)),
        )  # v, len desc, start asc, label order asc

    # We can’t invert mixed signs easily; use tuple and reverse at sort.

    ranked = sorted(
        votes.items(),
        key=lambda kv: (
            kv[1],  # votes (desc via reverse=True)
            (kv[0][2] - kv[0][1]),  # length
            -kv[0][1],  # start asc -> invert for reverse
            -(10 - _LABEL_ORDER.get(kv[0][0], 9)),  # label order asc -> invert
        ),
        reverse=True,
    )

    # Determine how many to return
    k = len(ranked) if (top_k is None or top_k <= 0) else min(top_k, len(ranked))

    out = []
    for (lab, s, e), _ in ranked[:k]:
        # choose the most common text for this key (or None)
        txt = None
        if texts[(lab, s, e)]:
            txt = texts[(lab, s, e)].most_common(1)[0][0]
        out.append({"type": lab, "startIndex": s, "endIndex": e, "text": txt})
    return out


def _verify_s1_spans(
    chat, text: str, spans: list[dict], keep_max: int = 4
) -> list[dict]:
    if not spans:
        return []
    try:
        sys_p = build_s1_verify_system()
        usr_p = build_s1_verify_user(text=text, candidates=spans)
        resp = chat.chat(
            system_prompt=sys_p,
            user_prompt=usr_p,
            max_tokens=8196,
            temperature=0.0,
            stop_sequences=["</answer>"],
        )
        raw = resp.get("answer") if isinstance(resp, dict) else (resp or "")
        js = _extract_json_from_answer(raw)
        obj = json.loads(js) if js else {}
        kept = obj.get("kept") if isinstance(obj, dict) else []
        # sanitize
        clean = []
        for m in kept or []:
            try:
                t = (m.get("type") or "").strip()
                s = int(m.get("startIndex"))
                e = int(m.get("endIndex"))
                txt = (m.get("text") or "").strip()
                if t and txt and e > s and txt == text[s:e]:
                    clean.append(
                        {"type": t, "startIndex": s, "endIndex": e, "text": txt}
                    )
            except Exception:
                continue
        # role-cap and max
        final, role_seen = [], {}
        for m in clean:
            r = m["type"]
            if role_seen.get(r, 0) >= 1:
                continue
            role_seen[r] = role_seen.get(r, 0) + 1
            final.append(m)
            if len(final) >= keep_max:
                break
        return final
    except Exception:
        return spans[:keep_max]  # fail-open but capped


_VALID_S2 = {"conspiracy", "non"}


def _parse_s2_answer(raw: str, allow_cant_tell: bool = False) -> tuple[str | None, str]:
    """
    Return (label, rationale). Label in {"conspiracy","non"} (or "cant_tell" if allowed), else None.
    Robust to free-form answers.
    """
    js = _extract_json_from_answer(raw)
    label, rationale = None, ""
    try:
        obj = json.loads(js) if js else {}
        if isinstance(obj, dict):
            label = (obj.get("label") or "").strip().lower() or None
            rationale = (obj.get("rationale") or "").strip()
    except Exception:
        pass

    # fallback regex if JSON missing/invalid
    if label not in _VALID_S2:
        if re.search(r"\bconspiracy\b", raw, re.I):
            label = "conspiracy"
        elif re.search(r"\bnon\b|\bnot\b", raw, re.I):
            label = "non"

    if label == "cant_tell" and not allow_cant_tell:
        label = None

    if label not in _VALID_S2:
        label = None

    return label, rationale


def _vote_s2(
    samples: list[tuple[str | None, str]], allow_cant_tell: bool = False
) -> tuple[str | None, str]:
    """
    Majority vote over labels; tie-break: prefer label with longest rationale,
    then deterministic order ("conspiracy" > "non" > "cant_tell" if allowed).
    Returns (label, fused_rationale).
    """
    from collections import Counter

    labels_only = [l for (l, _) in samples if l in _VALID_S2]
    if not labels_only:
        return None, ""

    # exclude cant_tell from majority unless it's the only label present
    if not allow_cant_tell:
        labels_eff = [l for l in labels_only if l != "cant_tell"] or labels_only
    else:
        labels_eff = labels_only

    cnt = Counter(labels_eff)
    if not cnt:
        return None, ""

    top = cnt.most_common()
    best_label, best_ct = top[0]

    # tie-break if needed
    competitors = [l for l, c in top if c == best_ct]
    if len(competitors) > 1:
        # prefer longer rationale among tied labels
        long_by_label: dict[str, tuple[int, str]] = {}
        for l, r in samples:
            if l in competitors:
                ln = len(r or "")
                if l not in long_by_label or ln > long_by_label[l][0]:
                    long_by_label[l] = (ln, r)
        # deterministic label order preference if still tied
        ordered = sorted(
            competitors,
            key=lambda x: {"conspiracy": 0, "non": 1, "cant_tell": 2}.get(x, 3),
        )
        best_label = max(ordered, key=lambda l: long_by_label.get(l, (0, ""))[0])

    # fused rationale: take the longest rationale among samples with best_label (capped to ~2 sentences)
    rats = [r for (l, r) in samples if l == best_label and r]
    fused = max(rats, key=len) if rats else ""
    # heuristic cap to 2 sentences
    if fused.count(".") > 2:
        parts = re.split(r"(?<=\.)\s+", fused)
        fused = " ".join(parts[:2]).strip()

    return best_label, fused


def _write_jsonl_zip(
    records: list[dict],
    out_path: pathlib.Path,
    *,
    force_jsonl: bool = False,
    name_hint: str = "predictions",
):
    """
    Write records either as plain .jsonl (if force_jsonl or path doesn't end with .zip)
    or zip with a {name_hint}.jsonl inside when path ends with .zip.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if force_jsonl or not str(out_path).lower().endswith(".zip"):
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    import zipfile, io

    buf = io.StringIO()
    for r in records:
        buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    data = buf.getvalue().encode("utf-8")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        inner = f"{name_hint}.jsonl"
        zf.writestr(inner, data)


# ---------------- Write submission files (exact Codabench schemas) ----------------
def _to_coda_s1(markers: list[dict], text: str) -> list[dict]:
    out = []
    for m in markers or []:
        lab = (m.get("label") or m.get("type") or "").strip()
        s = m.get("start", m.get("startIndex"))
        e = m.get("end", m.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        if e <= s or s < 0 or e > len(text):
            continue
        out.append(
            {
                "type": lab,
                "startIndex": s,
                "endIndex": e,
                "text": text[s:e],
            }
        )
    return out


def _to_coda_s2(label: str) -> str:
    lab = (label or "").strip().lower()
    if lab == "conspiracy":
        return "Yes"
    if lab == "non":
        return "No"
    return ""  # allowed backstop (keeps line order)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run PsyCoMark prompts with robust few-shot handling."
    )
    ap.add_argument(
        "--input-file",
        type=pathlib.Path,
        required=True,
        help="jsonl with rows to annotate/classify",
    )
    ap.add_argument(
        "--fewshot-bank",
        type=pathlib.Path,
        required=True,
        help="fewshot_bank.json produced earlier",
    )
    ap.add_argument(
        "--out-file",
        type=pathlib.Path,
        required=False,
        help="Where to save model outputs (jsonl) — used if --out-s1/--out-s2 are not given.",
    )
    ap.add_argument("--model-id", type=str, default=None)
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--task", type=str, default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--limit", type=int, default=0, help="Limit docs (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s1-fs-k", type=int, default=10)
    ap.add_argument("--s2-fs-k", type=int, default=10)
    ap.add_argument(
        "--sub-s1",
        type=pathlib.Path,
        help="Where to write S1 *submission* file (minimal schema). If ends with .zip, a JSONL is zipped inside.",
    )
    ap.add_argument(
        "--sub-s2",
        type=pathlib.Path,
        help="Where to write S2 *submission* file (minimal schema). If ends with .zip, a JSONL is zipped inside.",
    )
    ap.add_argument(
        "--no-zip",
        action="store_true",
        help="Write plain .jsonl even if submission/output path ends with .zip.",
    )

    # token limits
    ap.add_argument(
        "--s1-max-tokens",
        type=int,
        default=8196,
        help="Max tokens for S1 generations (per sample if self-consistency).",
    )
    ap.add_argument(
        "--s2-max-tokens",
        type=int,
        default=8196,
        help="Max tokens for S2 generations (per sample if self-consistency).",
    )

    # S1 self-consistency + verifier
    ap.add_argument(
        "--s1-sc",
        action="store_true",
        default=False,
        help="Enable S1 self-consistency (multi-sample + merge).",
    )
    ap.add_argument(
        "--s1-samples",
        type=int,
        default=4,
        help="Number of S1 generations to sample when --s1-sc.",
    )
    ap.add_argument(
        "--s1-sc-temp", type=float, default=0.7, help="Sampling temperature for S1 SC."
    )
    ap.add_argument(
        "--s1-verify",
        action="store_true",
        default=True,
        help="Validate S1 spans with a verifier prompt.",
    )
    ap.add_argument(
        "--s1-verify-max",
        type=int,
        default=4,
        help="Max spans to keep per example after verification.",
    )

    # S2 self-consistency
    ap.add_argument(
        "--s2-sc",
        action="store_true",
        default=True,
        help="Enable S2 self-consistency (multi-sample + vote).",
    )
    ap.add_argument(
        "--s2-samples", type=int, default=5, help="Number of S2 generations for SC."
    )
    ap.add_argument(
        "--s2-sc-temp", type=float, default=0.7, help="Sampling temperature for S2 SC."
    )
    ap.add_argument(
        "--s2-allow-cant-tell",
        action="store_true",
        default=False,
        help='Let the model output {"label":"cant_tell"}; excluded from majority vote unless it wins outright.',
    )
    ap.add_argument(
        "--artifacts-file",
        type=pathlib.Path,
        required=True,
        help="prompt_artifacts.json produced by make_prompt_artifacts.py",
    )
    # Optional flags if you want to flip CoT per task
    ap.add_argument("--s1-cot", action="store_true", default=True)
    ap.add_argument("--s2-cot", action="store_true", default=True)

    # split outputs
    ap.add_argument(
        "--out-s1",
        type=pathlib.Path,
        help="Where to write S1 results (if .zip, a JSONL will be zipped inside).",
    )
    ap.add_argument(
        "--out-s2",
        type=pathlib.Path,
        help="Where to write S2 results (if .zip, a JSONL will be zipped inside).",
    )
    # optional preview of prompts at start
    ap.add_argument(
        "--print-prompts",
        action="store_true",
        default=True,
        help="Print the FIRST S1/S2 system+user prompts at startup (S2 uses empty S1 markers for preview).",
    )
    ap.add_argument(
        "--s1-ext-thinking",
        action="store_true",
        help="Enable Anthropic Extended Thinking for S1.",
    )
    ap.add_argument(
        "--s1-thinking-budget",
        type=int,
        default=8196,
        help="Token budget for S1 Extended Thinking (must be < max tokens).",
    )
    ap.add_argument(
        "--s2-ext-thinking",
        action="store_true",
        help="Enable Anthropic Extended Thinking for S2.",
    )
    ap.add_argument(
        "--s2-thinking-budget",
        type=int,
        default=512,
        help="Token budget for S2 Extended Thinking.",
    )

    args = ap.parse_args()
    random.seed(args.seed)

    s1_budget = max(0, min(args.s1_thinking_budget, args.s1_max_tokens - 1))
    s2_budget = max(0, min(args.s2_thinking_budget, args.s2_max_tokens - 1))

    # -------- Load inputs & few-shots --------
    rows = _read_jsonl(args.input_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    LOG.info("Loaded %d rows.", len(rows))

    arts = load_artifacts(args.artifacts_file)
    bank = _load_fewshot_bank(args.fewshot_bank)
    fs_s1_pool = bank.get("s1") or []
    fs_s2_pool = bank.get("s2") or []

    chat = _get_chat(args.model_id, args.region)

    # -------- Print FIRST prompts (preview) --------
    first_text = next(
        ((r.get("text") or "").strip() for r in rows if (r.get("text") or "").strip()),
        "",
    )
    if args.print_prompts and first_text:
        # S1 preview
        if args.task in ("s1", "both"):
            fewshots_s1_ = _select_s1_fewshots_for_doc(
                fs_s1_pool, k=min(args.s1_fs_k, len(fs_s1_pool))
            )
            sys_prompt_s1, user_prompt_s1 = build_s1_prompts_adapter(
                text=first_text,
                prompt_arts=arts,
                s1_fewshot_list=fewshots_s1_,
                tech=("fs_cot" if args.s1_cot else "fs"),
            )

            print("\n======= S1 SYSTEM PROMPT (preview) =======\n")
            print(sys_prompt_s1)
            print("\n======= S1 USER PROMPT (preview) =======\n")
            print(user_prompt_s1)

        # S2 preview (uses empty S1 markers just to show structure)
        if args.task in ("s2", "both"):
            fewshots_s2_ = _select_s2_fewshots_for_doc(
                fs_s2_pool, k=min(args.s2_fs_k, len(fs_s2_pool))
            )
            sys_prompt_s2, user_prompt_s2 = (
                build_s2_prompts_adapter(  # <-- FIX: Call correct adapter
                    text=first_text,
                    markers=[],
                    fewshots=fewshots_s2_,  # <-- FIX: Pass the balanced list
                    tech=("fs_cot" if args.s2_cot else "fs"),
                    allow_cant_tell=args.s2_allow_cant_tell,
                )
            )

            print("\n======= S2 SYSTEM PROMPT (preview) =======\n")
            print(sys_prompt_s2)
            print("\n======= S2 USER PROMPT (preview) =======\n")
            print(user_prompt_s2)

    # -------- Run --------
    n = len(rows)
    t0 = time.time()
    s1_records: list[dict] = []
    s2_records: list[dict] = []
    first10_raw: list[dict] = []

    first10_printed = 0
    for i, r in enumerate(tqdm(rows, total=n, desc="Docs", unit="doc"), 1):
        first10_printed += 1
        doc_id = r.get("_id") or r.get("doc_id") or f"row{i}"
        text = (r.get("text") or "").strip()
        subreddit = r.get("subreddit")
        if not text:
            continue

        # ---------- S1 ----------
        s1_spans_for_s2: list[dict] = []
        s1_raw_dump = ""
        if args.task in ("s1", "both"):
            fewshots_s1 = _select_s1_fewshots_for_doc(
                fs_s1_pool, k=min(args.s1_fs_k, len(fs_s1_pool))
            )
            # sys_prompt_s1 = build_s1_system()
            # user_prompt_s1 = build_s1_user(text=text, fewshots=fewshots_s1)
            sys_prompt_s1, user_prompt_s1 = build_s1_prompts_adapter(
                text=text,
                prompt_arts=arts,
                s1_fewshot_list=fewshots_s1,
                tech=("fs_cot" if args.s1_cot else "fs"),
                shots=args.s1_fs_k,
            )
            s1_thinking_dump = ""  # <-- NEW: variable to store CoT
            try:
                if chat is not None:
                    if args.s1_sc:
                        samples = []
                        raws = []  # <-- NEW: store raw answers for debug
                        thinks = []  # <-- NEW: store thinking for debug

                        for _ in range(max(1, int(args.s1_samples))):
                            resp = chat.chat(
                                system_prompt=sys_prompt_s1,
                                user_prompt=user_prompt_s1,
                                max_tokens=args.s1_max_tokens,
                                temperature=max(0.1, float(args.s1_sc_temp)),
                                stop_sequences=["</answer>"],
                                # --- ADD THESE ---
                                enable_extended_thinking=args.s1_ext_thinking,
                                thinking_budget_tokens=s1_budget,
                            )
                            # --- UPDATED RESPONSE HANDLING ---
                            resp_dict = (
                                resp
                                if isinstance(resp, dict)
                                else {"thinking": "", "answer": str(resp or "")}
                            )
                            raw_i = resp_dict.get("answer", "")
                            think_i = resp_dict.get("thinking", "")
                            raws.append(raw_i)
                            thinks.append(think_i)

                            spans_i = _safe_parse_s1_generation(raw_i)
                            if spans_i:
                                samples.append(spans_i)

                        merged = _merge_sc_spans(samples, top_k=args.s1_verify_max)
                        if args.s1_verify and merged:
                            merged = _verify_s1_spans(
                                chat, text, merged, keep_max=args.s1_verify_max
                            )
                        s1_spans_for_s2 = merged
                        s1_raw_dump = json.dumps(
                            {
                                "samples": len(samples),
                                "merged": merged,
                                "raw_answers": raws,
                            },
                            ensure_ascii=False,
                        )
                        s1_thinking_dump = (
                            thinks[0] if thinks else ""
                        )  # Store first CoT
                    else:
                        resp = chat.chat(
                            system_prompt=sys_prompt_s1,
                            user_prompt=user_prompt_s1,
                            max_tokens=args.s1_max_tokens,
                            temperature=0.0,
                            stop_sequences=["</answer>"],
                            # --- ADD THESE ---
                            enable_extended_thinking=args.s1_ext_thinking,
                            thinking_budget_tokens=s1_budget,
                        )
                        # --- UPDATED RESPONSE HANDLING ---
                        resp_dict = (
                            resp
                            if isinstance(resp, dict)
                            else {"thinking": "", "answer": str(resp or "")}
                        )
                        s1_raw_dump = resp_dict.get("answer", "")
                        s1_thinking_dump = resp_dict.get("thinking", "")

                        s1_spans_for_s2 = _safe_parse_s1_generation(s1_raw_dump)

                        # --- ADD THIS BLOCK ---
                        if args.s1_verify and s1_spans_for_s2:
                            s1_spans_for_s2 = _verify_s1_spans(
                                chat, text, s1_spans_for_s2, keep_max=args.s1_verify_max
                            )
            except Exception as e:
                LOG.warning("[S1] generate() failed -> %s", e)
                s1_spans_for_s2 = []

            s1_records.append(
                {
                    "_id": doc_id,
                    "subreddit": subreddit,
                    "fewshots_used": len(fewshots_s1),
                    "raw": s1_raw_dump,
                    "thinking": s1_thinking_dump,  # <-- NEW: store thinking
                    "spans": s1_spans_for_s2,
                }
            )

        if first10_printed < 10:
            lines = [f"[{first10_printed+1}] id={doc_id}"]
            if args.task in ("s1", "both"):
                lines.append("  S1 raw: " + (s1_raw_dump or "")[:800])
                # --- ADD THIS ---
                lines.append(f"  S1 parsed: {len(s1_spans_for_s2)} spans")
            tqdm.write("\n".join(lines))

        # ---------- S2 ----------
        s2_raw_dump = ""
        s2_thinking_dump = ""  # <-- NEW: variable to store CoT
        s2_label, s2_rationale = None, ""
        if args.task in ("s2", "both"):
            fewshots_s2 = _select_s2_fewshots_for_doc(
                fs_s2_pool, k=min(args.s2_fs_k, len(fs_s2_pool))
            )
            sys_prompt_s2, user_prompt_s2 = build_s2_prompts_adapter(
                text=text,
                markers=(s1_spans_for_s2 if args.task == "both" else []),
                fewshots=fewshots_s2,
                tech=("fs_cot" if args.s2_cot else "fs"),
                allow_cant_tell=args.s2_allow_cant_tell,
            )
            try:
                if chat is not None:
                    if args.s2_sc:
                        samples = []
                        raws = []
                        thinks = []  # <-- NEW
                        for _ in range(max(1, int(args.s2_samples))):
                            resp = chat.chat(
                                system_prompt=sys_prompt_s2,
                                user_prompt=user_prompt_s2,
                                max_tokens=args.s2_max_tokens,
                                temperature=max(0.0, float(args.s2_sc_temp)),
                                stop_sequences=["</answer>"],
                                # --- ADD THESE ---
                                enable_extended_thinking=args.s2_ext_thinking,
                                thinking_budget_tokens=s2_budget,
                            )
                            # --- UPDATED RESPONSE HANDLING ---
                            resp_dict = (
                                resp
                                if isinstance(resp, dict)
                                else {"thinking": "", "answer": str(resp or "")}
                            )
                            raw_i = resp_dict.get("answer", "")
                            think_i = resp_dict.get("thinking", "")
                            raws.append(raw_i)
                            thinks.append(think_i)

                            label_i, rat_i = _parse_s2_answer(
                                raw_i, allow_cant_tell=args.s2_allow_cant_tell
                            )
                            samples.append((label_i, rat_i))

                        s2_label, s2_rationale = _vote_s2(
                            samples, allow_cant_tell=args.s2_allow_cant_tell
                        )
                        s2_raw_dump = "[SC] " + " ||| ".join(raws[:5])
                        s2_thinking_dump = (
                            thinks[0] if thinks else ""
                        )  # Store first CoT
                    else:
                        resp = chat.chat(
                            system_prompt=sys_prompt_s2,
                            user_prompt=user_prompt_s2,
                            max_tokens=args.s2_max_tokens,
                            temperature=0.0,
                            stop_sequences=["</answer>"],
                            # --- ADD THESE ---
                            enable_extended_thinking=args.s2_ext_thinking,
                            thinking_budget_tokens=s2_budget,
                        )
                        # --- UPDATED RESPONSE HANDLING ---
                        resp_dict = (
                            resp
                            if isinstance(resp, dict)
                            else {"thinking": "", "answer": str(resp or "")}
                        )
                        s2_raw_dump = resp_dict.get("answer", "")
                        s2_thinking_dump = resp_dict.get("thinking", "")

                        s2_label, s2_rationale = _parse_s2_answer(
                            s2_raw_dump, allow_cant_tell=args.s2_allow_cant_tell
                        )
            except Exception as e:
                LOG.warning("[S2] generate() failed -> %s", e)
                s2_label, s2_rationale = None, ""

            if s2_label not in {"conspiracy", "non", "cant_tell"}:
                if re.search(r"\bconspiracy\b", s2_raw_dump, re.I):
                    s2_label = "conspiracy"
                elif re.search(r"\bnon\b|\bnot\b", s2_raw_dump, re.I):
                    s2_label = "non"

            s2_records.append(
                {
                    "_id": doc_id,
                    "subreddit": subreddit,
                    "fewshots_used": len(fewshots_s2),
                    "raw": s2_raw_dump,
                    "thinking": s2_thinking_dump,  # <-- NEW: store thinking
                    "label": s2_label,
                    "rationale": s2_rationale,
                }
            )

            if first10_printed < 10:
                lines = [f"[{first10_printed}] id={doc_id}"]
                if args.task in ("s2", "both"):
                    lines.append("  S2 raw: " + (s2_raw_dump or "")[:800])
                    # --- ADD THIS ---
                    lines.append(f"  S2 parsed: {s2_label}")
                tqdm.write("\n".join(lines))

        # save first 10 raw dumps for inspection
        if len(first10_raw) < 10:
            first10_raw.append(
                {
                    "_id": doc_id,
                    "s1_raw": s1_raw_dump if args.task in ("s1", "both") else None,
                    "s2_raw": s2_raw_dump if args.task in ("s2", "both") else None,
                }
            )

    # Build a map from id -> original text (needed to emit S1 "text" snippets)
    _id2text = {(r.get("_id") or r.get("doc_id")): (r.get("text") or "") for r in rows}

    # S1: {"_id": "...", "markers":[{"type","startIndex","endIndex","text"}, ...]}
    if args.sub_s1 and args.task in ("s1", "both"):
        s1_submit = []
        for r in s1_records:
            did = r["_id"]
            txt = _id2text.get(did, "")
            s1_submit.append(
                {"_id": did, "markers": _to_coda_s1(r.get("spans") or [], txt)}
            )
        _write_jsonl_zip(
            s1_submit, args.sub_s1, force_jsonl=args.no_zip, name_hint="submission"
        )
        LOG.info("S1 submission -> %s (%d lines)", args.sub_s1, len(s1_submit))

    # S2: {"_id": "...", "conspiracy": "Yes|No"}
    if args.sub_s2 and args.task in ("s2", "both"):
        s2_submit = [
            {"_id": r["_id"], "conspiracy": _to_coda_s2(r.get("label"))}
            for r in s2_records
        ]
        _write_jsonl_zip(
            s2_submit, args.sub_s2, force_jsonl=args.no_zip, name_hint="submission"
        )
        LOG.info("S2 submission -> %s (%d lines)", args.sub_s2, len(s2_submit))

    # -------- Print first 10 raw answers --------
    if first10_raw:
        print("\n======= FIRST 10 RAW ANSWERS =======\n")
        for j, rec in enumerate(first10_raw, 1):
            print(f"[{j}] id={rec['_id']}")
            if args.task in ("s1", "both"):
                print("  S1 raw:", (rec.get("s1_raw") or "")[:800], "\n")
            if args.task in ("s2", "both"):
                print("  S2 raw:", (rec.get("s2_raw") or "")[:800], "\n")

    elapsed = time.time() - t0
    LOG.info("All done. Elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
