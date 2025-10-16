# starter/llm_infer_spans.py
import argparse
import json
import logging
import os
import pathlib
import time
import random
import re
from typing import Any, Dict, List

import orjson

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- sys.path for repo root ----------
import pathlib as _pathlib
import sys

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))


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
# ------------------------------------------

from src.psycomark.llm.bedrock_chat import Chat
from src.psycomark.llm.eda_support import build_s1_policy, load_fewshots

LABELS = ["Actor", "Action", "Effect", "Victim", "Evidence"]

S1_BASE_RULES = """\
You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 1).
Extract character spans for labels: Actor, Action, Effect, Victim, Evidence.
STRICT OUTPUT: return ONLY JSON list with objects:
[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}]
Guidelines:
- Character offsets are 0-indexed; end is exclusive.
- Keep spans tight (exclude trailing punctuation/stopwords).
- If two labels overlap heavily, keep both only if clearly distinct.
- Do not invent spans. Omit a label if the text doesn’t support it.
"""

S1_USER_TEMPLATE = """\
TASK: Extract spans for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY strict JSON:
[{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}]

TEXT:
{doc_text}
"""

# Fallback few-shot (used only if EDA few-shots are not available)
S1_FALLBACK_FEWSHOTS = [
    {
        "text": "They paid off the officials to hide the scandal.",
        "json": [
            {"label": "Actor", "start": 0, "end": 4},  # "They"
            {"label": "Action", "start": 5, "end": 12},  # "paid off"
            {"label": "Victim", "start": 20, "end": 29},  # "officials"
            {"label": "Effect", "start": 33, "end": 47},  # "hide the scandal"
        ],
    },
    {
        "text": "The agency coordinated a covert operation against whistleblowers.",
        "json": [
            {"label": "Actor", "start": 0, "end": 10},  # "The agency"
            {"label": "Action", "start": 11, "end": 22},  # "coordinated"
            {"label": "Effect", "start": 25, "end": 41},  # "a covert operation"
            {"label": "Victim", "start": 50, "end": 64},  # "whistleblowers"
        ],
    },
]


# -------- helpers --------
def _shorten(txt: str, max_chars: int = 1600) -> str:
    txt = txt or ""
    return txt if len(txt) <= max_chars else txt[:max_chars] + "..."


def json_list_from_text(s: str) -> List[Dict[str, Any]]:
    # Greedy bracket match; robust to extra text around JSON
    m = re.search(r"\[.*\]", s, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def clip_and_validate(spans: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    L = len(text)
    out: List[Dict[str, Any]] = []
    for m in spans:
        lab = m.get("label")
        if lab not in LABELS:
            continue
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out


def nms_per_label(spans: List[Dict[str, Any]], iou=0.9) -> List[Dict[str, Any]]:
    # Simple char-level IoU NMS per label
    by_lab = {lab: [] for lab in LABELS}
    for m in spans:
        by_lab[m["label"]].append(m)
    keep: List[Dict[str, Any]] = []
    for lab, arr in by_lab.items():
        arr = sorted(arr, key=lambda x: (x["start"], x["end"]))
        sel: List[Dict[str, Any]] = []
        for s in arr:
            ok = True
            for t in sel:
                inter = max(0, min(s["end"], t["end"]) - max(s["start"], t["start"]))
                union = (s["end"] - s["start"]) + (t["end"] - t["start"]) - inter
                i = (inter / union) if union > 0 else 0.0
                if i >= iou:
                    ok = False
                    break
            if ok:
                sel.append(s)
        keep.extend(sel)
    return keep


def to_codabench(_id: str, text: str, spans: List[Dict[str, int]]) -> Dict[str, Any]:
    markers = []
    for m in spans:
        s, e, lab = m["start"], m["end"], m["label"]
        markers.append({"type": lab, "startIndex": s, "endIndex": e, "text": text[s:e]})
    return {"_id": _id, "markers": markers}


def render_s1_fewshots(shots: List[Dict[str, Any]]) -> str:
    if not shots:
        return ""
    blocks = []
    for ex in shots:
        txt = ex.get("text") or ex.get("doc_text") or ""
        gold = ex.get("gold") or ex.get("json") or []
        blocks.append(
            "EXAMPLE:\nTEXT:\n"
            + _shorten(txt, 800)
            + "\nJSON:\n"
            + json.dumps(gold, ensure_ascii=False)
        )
    return "\n\n".join(blocks)


# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--submission-file", required=True)
    ap.add_argument("--model-id", default=None)  # allow env fallback
    ap.add_argument("--region", default=None)  # allow env fallback
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eda-root", required=False, default=None)
    args = ap.parse_args()
    random.seed(args.seed)

    # Defaults aligned to your Bedrock setup
    model_id = args.model_id or os.environ.get(
        "MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    logging.info(
        f"[S1] model_id={model_id} region={region} "
        f"max_tokens={args.max_tokens} temperature={args.temperature}"
    )
    logging.info(f"[S1] test_file={args.test_file} out={args.submission_file}")

    subp = pathlib.Path(args.submission_file)
    subp.parent.mkdir(parents=True, exist_ok=True)

    # EDA policy & few-shots
    policy = ""
    fewshots: List[Dict[str, Any]] = []
    if args.eda_root:
        eda_root = pathlib.Path(args.eda_root)
        logging.info(f"[S1] Loading EDA from: {eda_root}")
        policy = build_s1_policy(eda_root) or ""
        fewshots = load_fewshots(eda_root, "s1", max_n=6) or []
        logging.info(f"[S1] fewshots loaded={len(fewshots)} (fallback used if 0)")

    # Build few-shot block once
    shots = fewshots if fewshots else S1_FALLBACK_FEWSHOTS
    fewshot_block = render_s1_fewshots(shots)
    logging.info(f"[S1] fewshot_block_len={len(fewshot_block)}")

    n_docs = 0
    n_parse_ok = 0
    total_spans_raw = total_spans_valid = total_spans_nms = 0
    t0_all = time.time()

    with open(args.test_file, "r", encoding="utf-8") as fi, open(
        args.submission_file, "w", encoding="utf-8"
    ) as fo:

        logging.info(f"[S1] reading test file: {args.test_file}")
        for line in fi:
            if not line.strip():
                continue
            rec = orjson.loads(line)
            _id = rec.get("_id") or rec.get("doc_id")
            text = rec.get("text", "")
            n_docs += 1
            t0 = time.time()

            convo = Chat(
                model_id=model_id,
                region=region,
                max_tokens=args.max_tokens,
                temperature=args.temperature,  # Chat class enforces sampling constraints
            )

            # policy first, then base rules
            if policy:
                convo.add_system(policy)
            convo.add_system(S1_BASE_RULES)

            user = S1_USER_TEMPLATE.format(doc_text=_shorten(text))
            convo.add_user((fewshot_block + "\n\n" if fewshot_block else "") + user)

            # Call Bedrock with retry inside Chat.generate
            try:
                out = convo.generate()
            except Exception as e:
                logging.exception(f"[S1] generation failed _id={_id}")
                # Fail-safe: emit empty markers for this doc and continue
                fo.write(orjson.dumps({"_id": _id, "markers": []}).decode() + "\n")
                continue

            # Parse model output to list of spans
            raw_list = json_list_from_text(out)
            total_spans_raw += len(raw_list)
            if isinstance(raw_list, list):
                n_parse_ok += 1
            else:
                logging.warning(f"[S1] JSON repair for _id={_id}")

            valid = clip_and_validate(raw_list, text)
            total_spans_valid += len(valid)

            final_spans = nms_per_label(valid, iou=0.9)
            total_spans_nms += len(final_spans)

            fo.write(orjson.dumps(to_codabench(_id, text, final_spans)).decode() + "\n")

            if n_docs % 50 == 0:
                logging.info(
                    f"[S1] progress {n_docs} docs | last_doc_ms={(time.time()-t0)*1000:.0f} "
                    f"| spans raw/valid/nms={total_spans_raw}/{total_spans_valid}/{total_spans_nms}"
                )

    logging.info(
        f"[S1] done. docs={n_docs} parse_ok={n_parse_ok} wall_sec={time.time()-t0_all:.2f} "
        f"spans raw/valid/nms={total_spans_raw}/{total_spans_valid}/{total_spans_nms} "
        f"out={args.submission_file}"
    )


if __name__ == "__main__":
    main()
