#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_runner.py — Pydantic-AI native S1 / S2 runner (drop-in replacement)

This script preserves the core functionality of the old prompt runner while
moving generation, validation, and retries into pydantic-ai agents defined in
`psycomark_agents.py`. It also includes optional local-align repair (±window)
for S1 spans before saving and passing them to S2.

Usage (examples):
  # Single text file -> run S1 then S2, save outputs
  python prompt_runner.py input.txt --s1-out s1.jsonl --s2-out s2.jsonl

  # JSONL with {"id","text"} rows
  python prompt_runner.py data/dev.jsonl --format jsonl --s1-out s1.jsonl --s2-out s2.jsonl

  # Run only S1 with custom model and region, disable CoT, disable repair
  python prompt_runner.py input.txt --task s1 --model-id eu.anthropic.claude-sonnet-4-5-20250929-v1:0 \
      --region eu-central-1 --no-cot --no-repair

Notes:
- Few-shots, priors, conflicts are optional. If provided, they are passed
  through to the pydantic-ai agents while reusing your existing prompt text.
- Local-align repair searches for span.text within ±window chars around
  the predicted (start,end); if found, we adjust the offsets.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pathlib
from tqdm.auto import tqdm

# -----------------------
# CLI & Environment setup
# -----------------------

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pydantic-AI S1/S2 runner")
    p.add_argument(
        "input", help="Path to a .txt file (single doc) or .jsonl file (multiple docs)."
    )
    p.add_argument(
        "--format",
        choices=["text", "jsonl"],
        default="jsonl",
        help="Input format; auto-detected by extension if omitted.",
    )
    p.add_argument(
        "--task",
        choices=["s1", "s2", "both"],
        default="both",
        help="Which parts of the pipeline to run.",
    )
    p.add_argument(
        "--s1-out", default="s1_predictions.jsonl", help="Output JSONL for S1 spans."
    )
    p.add_argument(
        "--s2-out", default="s2_predictions.jsonl", help="Output JSONL for S2 labels."
    )
    p.add_argument(
        "--fewshot-bank",
        default=None,
        help="Optional fewshot bank JSON file (schema-agnostic best-effort).",
    )
    p.add_argument(
        "--priors-json", default=None, help="Optional priors JSON file for S1."
    )
    p.add_argument(
        "--conflicts-json", default=None, help="Optional conflicts JSON file for S1."
    )
    p.add_argument(
        "--allow-cant-tell",
        action="store_true",
        default=False,
        help="Allow 'cant_tell' in S2.",
    )
    p.add_argument(
        "--no-cot",
        action="store_true",
        default=False,
        help="Disable chain-of-thought instructions in prompts.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for few-shot sampling."
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Override MODEL_ID env (e.g., Bedrock model ID).",
    )
    p.add_argument("--region", default=None, help="Override AWS_DEFAULT_REGION env.")
    p.add_argument(
        "--repair-window",
        type=int,
        default=16,
        help="Local-align search window (±N) for S1 span text/offset mismatches.",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        default=True,
        help="Disable local-align repair (validator still enforces equality).",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p


def configure_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_env_overrides(model_id: Optional[str], region: Optional[str]):
    if model_id:
        os.environ["MODEL_ID"] = model_id
    if region:
        os.environ["AWS_DEFAULT_REGION"] = region


# -----------------------
# I/O and utilities
# -----------------------


def is_jsonl_path(path: str) -> bool:
    return path.lower().endswith(".jsonl")


def load_input(path: str, fmt: str | None) -> list[dict]:
    """
    Simple loader:
      - text file  -> [{"id": <filename>, "text": <file contents>}]
      - jsonl file -> one row per line, id priority: _id -> doc_id -> id -> row_i
                      text from 'text' (else empty string)
    """
    if fmt is None:
        fmt = "jsonl" if is_jsonl_path(path) else "text"

    rows: list[dict] = []

    if fmt == "text":
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        rows.append({"id": os.path.basename(path), "text": txt})
        return rows

    # jsonl (keep it simple, like before)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue  # ignore bad lines (old behavior)
            if not isinstance(obj, dict):
                continue
            tid = str(
                obj.get("_id") or obj.get("doc_id") or obj.get("id") or f"row_{i}"
            )
            txt = obj.get("text", "")
            rows.append({"id": tid, "text": txt})

    return rows


def save_jsonl(file_name: str, zip_name: str, items: Iterable[Dict[str, Any]]) -> None:
    """Persist items to JSONL inside subs_pydanticai/ and zip that file.

    Robust path handling:
      - Always creates subs_pydanticai directory.
      - If caller passes a path with directories, we take only the basename.
      - If caller accidentally passes a trailing slash (directory), we append .jsonl.
    """
    base_dir = "subs_pydanticai"
    os.makedirs(base_dir, exist_ok=True)
    # Normalize file_name to ensure it's a file, not a directory path.
    if file_name.endswith(("/", "\\")):
        # Strip trailing slashes and add extension if missing
        file_name = file_name.rstrip("/\\")
        if not file_name.lower().endswith(".jsonl"):
            file_name = file_name + ".jsonl"
    # Only keep the basename so we don't create nested arbitrary directories
    file_name = os.path.basename(file_name)
    if not file_name:
        file_name = "submission.jsonl"
    jsonl_path = os.path.join(base_dir, file_name)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Zip into the same directory for easier discovery
    import zipfile

    zip_path = os.path.join(base_dir, os.path.basename(zip_name))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(jsonl_path, arcname=os.path.basename(jsonl_path))

    # Always write jsonl as submission.jsonl first

    # os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # with open(path, "w", encoding="utf-8") as f:
    #    for obj in items:
    #        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    ## Zip the jsonl file
    # zip_path = path + ".zip"
    # import zipfile
    # with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #    zipf.write(path, arcname=os.path.basename(path))


# -----------------------
# Few-shot / priors / conflicts (best-effort)
# -----------------------


def safe_load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.warning("Failed to load %s: %s", path, e)
        return None


def pick_s1_fewshots(bank: Any, k_total: int = 10, seed: int = 42) -> List[dict]:
    """
    Best-effort selection from a flexible fewshot bank.
    Expected S1 schema example:
      {"task":"s1","text":"...","spans":[{"label":"Action","start":..,"end":..,"text":"..."}]}
    """
    if not bank:
        return []
    rnd = random.Random(seed)
    pool = [
        x
        for x in (bank if isinstance(bank, list) else bank.get("s1", []))
        if isinstance(x, dict)
    ]
    if not pool:
        return []
    rnd.shuffle(pool)
    return pool[:k_total]


def pick_s2_fewshots(
    bank: Any, k_total: int = 10, seed: int = 42, allow_cant_tell: bool = False
) -> List[dict]:
    """
    Best-effort selection for balanced S2 few-shots.
    Expected S2 example:
      {"task":"s2","text":"...","label":"conspiracy|non|cant_tell","rationale":"..."}
    """
    if not bank:
        return []
    rnd = random.Random(seed)
    pool = [
        x
        for x in (bank if isinstance(bank, list) else bank.get("s2", []))
        if isinstance(x, dict)
    ]
    if not pool:
        return []
    groups = {"conspiracy": [], "non": [], "cant_tell": []}
    for x in pool:
        lab = str(x.get("label", "")).lower()
        if lab in groups:
            groups[lab].append(x)
    k_each = max(1, k_total // (3 if allow_cant_tell else 2))
    out: List[dict] = []
    for lab in ("conspiracy", "non"):
        rnd.shuffle(groups[lab])
        out.extend(groups[lab][:k_each])
    if allow_cant_tell:
        rnd.shuffle(groups["cant_tell"])
        out.extend(groups["cant_tell"][:k_each])
    rnd.shuffle(out)
    return out[:k_total]


# -----------------------
# Local-align repair for S1 spans
# -----------------------


def local_align_search(
    raw: str, expected: str, approx_start: int, approx_end: int, window: int
) -> Optional[Tuple[int, int]]:
    """
    Search ±window around approx_start..approx_end for the first exact occurrence of `expected`.
    Returns (start,end) if found else None.
    """
    if not expected:
        return None
    L = len(raw)
    a = max(0, approx_start - window)
    b = min(L, approx_end + window)
    hay = raw[a:b]
    i = hay.find(expected)
    if i < 0:
        return None
    s = a + i
    e = s + len(expected)
    return (s, e)


def repair_s1_spans_with_local_align(
    raw_text: str,
    spans: List[Dict[str, Any]],
    window: int,
) -> List[Dict[str, Any]]:
    """
    For any span whose text != raw[start:end], try relocating within ±window.
    """
    fixed: List[Dict[str, Any]] = []
    for m in spans:
        s = int(m.get("start", 0))
        e = int(m.get("end", 0))
        t = m.get("text", "")
        label = m.get("label", m.get("type"))
        s = max(0, min(s, len(raw_text)))
        e = max(s, min(e, len(raw_text)))
        echo = raw_text[s:e]
        if t and echo != t:
            hit = local_align_search(raw_text, t, s, e, window)
            if hit:
                s, e = hit
        fixed.append({"label": label, "start": s, "end": e, "text": raw_text[s:e]})
    return fixed


# -----------------------
# Main pipeline
# -----------------------


async def run_pipeline_for_doc(
    *,
    doc_id: str,
    text: str,
    allow_cant_tell: bool,
    include_cot: bool,
    do_repair: bool,
    repair_window: int,
    priors: Optional[dict],
    conflicts: Optional[list],
    fewshot_bank: Optional[Any],
    task: str,
    agents_mod,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (s1_record, s2_record) as JSON-serializable dicts (or None on failure).
    """
    src_text = text
    if not src_text.strip():
        logging.warning("[%s] Empty text after trim; skipping.", doc_id)
        return None, None

    # Few-shots (best-effort)
    s1_fs = pick_s1_fewshots(fewshot_bank, k_total=8)
    s2_fs = pick_s2_fewshots(fewshot_bank, k_total=10, allow_cant_tell=allow_cant_tell)

    # ---------------- S1 ----------------
    s1_out = None
    s1_spans_for_s2: List[Dict[str, Any]] = []

    if task in ("s1", "both"):
        try:
            s1_struct = await agents_mod.run_s1(
                doc_id=doc_id,
                text=src_text,
                priors=priors or {},
                conflicts=conflicts or [],
                fewshots=s1_fs,
                include_cot=include_cot,
                temperature=0.0,
            )
            # Convert to plain dict for saving
            spans = [
                {
                    "type": (
                        s.label.value if hasattr(s.label, "value") else str(s.label)
                    ),
                    "startIndex": int(s.start),
                    "endIndex": int(s.end),
                    "text": s.text,
                }
                for s in (s1_struct.spans or [])
            ]

            # Optional local-align repair
            if do_repair and spans:
                spans = repair_s1_spans_with_local_align(
                    src_text, spans, window=repair_window
                )

            s1_out = {"_id": doc_id, "markers": spans}  # <-- exact submission shape
            s1_spans_for_s2 = spans

        except Exception as e:
            logging.error("[%s] S1 failed: %s", doc_id, e, exc_info=True)
            s1_out = {"id": doc_id, "error": f"S1 failed: {e}"}

    # ---------------- S2 ----------------
    s2_out = None
    if task in ("s2", "both"):
        try:
            # If we didn't just run S1, attempt to keep S2 going with empty spans
            s1_spans_for_s2 = s1_spans_for_s2 or []

            s2_struct = await agents_mod.run_s2(
                doc_id=doc_id,
                text=src_text,
                s1_output_spans=s1_spans_for_s2,
                fewshots=s2_fs,
                include_cot=include_cot,
                allow_cant_tell=allow_cant_tell,
                temperature=0.0,
            )
            s2_out = {
                "_id": doc_id,
                "conspiracy": (
                    "Yes" if str(s2_struct.label).lower() == "conspiracy" else "No"
                ),
            }
        except Exception as e:
            logging.error("[%s] S2 failed: %s", doc_id, e, exc_info=True)
            s2_out = {"id": doc_id, "error": f"S2 failed: {e}"}

    return s1_out, s2_out


async def main_async(args: argparse.Namespace):
    # Set env overrides before importing agents (so they read correct MODEL_ID/REGION)
    set_env_overrides(args.model_id, args.region)

    # Import here (lazy) so env is set first
    import psycomark_agents as agents_mod

    # Seed
    random.seed(args.seed)

    # Load inputs
    rows = load_input(args.input, args.format)

    # Load optional config artifacts
    priors = safe_load_json(args.priors_json) or {}
    conflicts = safe_load_json(args.conflicts_json) or []
    fewshot_bank = safe_load_json(args.fewshot_bank)

    # Process docs
    s1_records: List[Dict[str, Any]] = []
    s2_records: List[Dict[str, Any]] = []

    import asyncio

    sem = asyncio.Semaphore(2)  # mild concurrency to keep Bedrock happy

    async def _worker(row):
        async with sem:
            return await run_pipeline_for_doc(
                doc_id=row["id"],
                text=row["text"],
                allow_cant_tell=args.allow_cant_tell,
                include_cot=not args.no_cot,
                do_repair=not args.no_repair,
                repair_window=args.repair_window,
                priors=priors,
                conflicts=conflicts,
                fewshot_bank=fewshot_bank,
                task=args.task,
                agents_mod=agents_mod,
            )

    tasks = [_worker(r) for r in rows]
    # Progress bar over completed documents
    with tqdm(total=len(tasks), desc="Processing docs", unit="doc") as pbar:
        for fut in asyncio.as_completed(tasks):
            s1_out, s2_out = await fut
            if s1_out is not None:
                s1_records.append(s1_out)
            if s2_out is not None:
                s2_records.append(s2_out)
            pbar.update(1)

    # Save
    if s1_records and args.task in ("s1", "both"):
        from datetime import datetime

        save_jsonl(
            file_name="submission.jsonl",
            zip_name=f"submission_s1_{datetime.now().strftime('_%Y%m%d_%H%M%S')}.zip",
            items=s1_records,
        )
        logging.info("Wrote S1 outputs: %s (%d rows)", args.s1_out, len(s1_records))
    if s2_records and args.task in ("s2", "both"):
        save_jsonl(
            file_name="submission.jsonl",
            zip_name=f"submission_s2_{datetime.now().strftime('_%Y%m%d_%H%M%S')}.zip",
            items=s2_records,
        )
        logging.info("Wrote S2 outputs: %s (%d rows)", args.s2_out, len(s2_records))

    # Summary
    logging.info(
        "Done. Docs processed: %d | S1: %d | S2: %d",
        len(rows),
        len(s1_records),
        len(s2_records),
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    try:
        import asyncio

        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
