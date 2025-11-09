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
import json, re, sys, os, pathlib, logging, argparse, random, time
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter

# ---- Paths / imports ---------------------------------------------------------
_THIS = pathlib.Path(__file__).resolve()
ROOT = _THIS.parents[3]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional Bedrock wrapper (ok if missing for dry runs / unit tests)
try:
    from src.psycomark.llm.bedrock_chat import BedrockChat as Chat
except Exception:
    Chat = None

# Your prompt builder (we rely on these)
from src.psycomark.eda.prompt_builder import (
    build_s1_system,
    build_s1_user,
    build_s2_system,
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
        required=True,
        help="Where to save model outputs (jsonl)",
    )
    ap.add_argument("--model-id", type=str, default=None)
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--task", type=str, default="both", choices=["s1", "s2", "both"])
    ap.add_argument("--limit", type=int, default=0, help="Limit docs (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s1-fs-k", type=int, default=8)
    ap.add_argument("--s2-fs-k", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)

    rows = _read_jsonl(args.input_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    LOG.info("Loaded %d rows.", len(rows))

    bank = _load_fewshot_bank(args.fewshot_bank)
    fs_s1_pool = bank.get("s1") or []
    fs_s2_pool = bank.get("s2") or []

    chat = _get_chat(args.model_id, args.region)

    # write outputs as JSONL
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out_file.open("w", encoding="utf-8")

    n = len(rows)
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        doc_id = r.get("_id") or r.get("doc_id") or f"row{i}"
        text = (r.get("text") or "").strip()
        subreddit = r.get("subreddit")
        if not text:
            continue

        out_rec = {"_id": doc_id, "subreddit": subreddit}

        # ---------- S1 ----------
        if args.task in ("s1", "both"):
            # per-doc fewshots (robust)
            fewshots_s1 = _select_s1_fewshots_for_doc(
                fs_s1_pool, k=min(args.s1_fs_k, len(fs_s1_pool))
            )
            sys_prompt = build_s1_system()
            user_prompt = build_s1_user(text=text, fewshots=fewshots_s1)

            s1_answer_raw = ""
            s1_spans = []
            if chat is not None:
                try:
                    resp = chat.chat(
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    s1_answer_raw = (
                        resp.get("answer") if isinstance(resp, dict) else (resp or "")
                    )
                except Exception as e:
                    LOG.warning("[S1] generate() failed -> %s", e)

            # Parse & validate (never crash on malformed)
            s1_spans = _safe_parse_s1_generation(s1_answer_raw)
            out_rec["s1"] = {
                "fewshots_used": len(fewshots_s1),
                "raw": s1_answer_raw,
                "spans": s1_spans,
            }

        # ---------- S2 ----------
        if args.task in ("s2", "both"):
            fewshots_s2 = _select_s2_fewshots_for_doc(
                fs_s2_pool, k=min(args.s2_fs_k, len(fs_s2_pool))
            )
            sys_prompt = build_s2_system()
            user_prompt = build_s2_user(text=text, fewshots=fewshots_s2)

            s2_answer_raw = ""
            s2_label = None
            s2_rationale = ""
            if chat is not None:
                try:
                    resp = chat.chat(
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        max_tokens=384,
                        temperature=0.2,
                    )
                    s2_answer_raw = (
                        resp.get("answer") if isinstance(resp, dict) else (resp or "")
                    )
                except Exception as e:
                    LOG.warning("[S2] generate() failed -> %s", e)

            # Parse: accept JSON or simple tagged text
            js = _extract_json_from_answer(s2_answer_raw)
            try:
                obj = json.loads(js) if js else {}
            except Exception:
                obj = {}
            if isinstance(obj, dict):
                s2_label = (obj.get("label") or "").strip().lower() or None
                s2_rationale = (obj.get("rationale") or "").strip()
            if s2_label not in {"conspiracy", "non"}:
                # loose regex fallback
                if re.search(r"\bconspiracy\b", s2_answer_raw, re.I):
                    s2_label = "conspiracy"
                elif re.search(r"\bnon\b|\bnot\b", s2_answer_raw, re.I):
                    s2_label = "non"
            out_rec["s2"] = {
                "fewshots_used": len(fewshots_s2),
                "raw": s2_answer_raw,
                "label": s2_label,
                "rationale": s2_rationale,
            }

        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        # progress (human)
        if i % max(1, n // 10) == 0 or i == n:
            per = 100.0 * i / max(1, n)
            eta = (time.time() - t0) * (n - i) / max(1, i)
            LOG.info("Progress: %.0f%% | %d/%d | ETA ~ %.0fs", per, i, n, eta)

    fout.close()
    LOG.info("Done. Wrote -> %s", args.out_file)


if __name__ == "__main__":
    main()
