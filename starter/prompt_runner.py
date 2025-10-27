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
from collections import defaultdict, Counter
from typing import Any, Dict, List

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# --- Bedrock Chat (Sonnet 4.5 wrapper) ---
from src.psycomark.llm.bedrock_chat import Chat

# --- Reuse the largest runner's utilities / builders ---
# (We import only what we need to avoid code duplication.)
from starter.prompt_sweep_joint import (  # noqa: E402
    _load_prompt_artifacts,
    _snapshot_artifacts,
    s1_prompt,
    s2_prompt_with_markers,
    postprocess_s1_spans,
    read_jsonl,
    write_jsonl,
    _save_prompt_bundle,
    _save_prompt_meta,
    _to_codabench_s1,
    _pick_balanced_s1_fewshots,
    _pick_balanced_s2_fewshots,
    tune_threshold_dev,
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
    print(
        f"Model ID: {args.model_id or Chat.SONNET_45_MODEL_ID}, Region: {args.region}"
    )

    random.seed(42)
    from os import getenv

    model_id = (
        args.model_id or getenv("MODEL_ID") or getattr(Chat, "SONNET_45_MODEL_ID", None)
    )
    region = args.region or getenv("AWS_DEFAULT_REGION") or "us-east-1"

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
                "model_id": args.model_id or Chat.SONNET_45_MODEL_ID,
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
            sys_blocks, user_block, n_samples, temp = s1_prompt(
                txt,
                policy=(s1_policy if use_policy else ""),
                fewshots=fewshots_s1 if has_fs else [],
                boundary_note=(boundary if use_boundary else ""),
                tech=tech,
                prompt_arts=prompt_arts or None,
            )

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

            # === Bedrock: Chat(add_system/add_user/generate) ===
            chat_s1 = Chat(
                model_id=(model_id or Chat.SONNET_45_MODEL_ID),
                region=region,
                max_tokens=args.max_tokens_s1,
                temperature=args.temperature,
            )
            for s in sys_blocks:
                chat_s1.add_system(s)
            chat_s1.add_user(user_block)

            try:
                s1_raw = chat_s1.generate()
                spans_raw = list_json_extract(s1_raw)  # robust array extraction
                spans = clip_and_validate(spans_raw, txt)  # schema+bounds
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
        s2_out_rows = []
        s2_prob_rows = []
        saved_s2 = 0

        for _id, doc2 in id2doc_s2.items():
            txt = (doc2.get("text") or "").strip()
            mks = id2markers.get(_id, [])

            fewshots_s2 = (
                _pick_balanced_s2_fewshots(fs_s2, k=min(8, len(fs_s2))) if fs_s2 else []
            )

            sys_blocks, user_block, n_samples, temp = s2_prompt_with_markers(
                doc_text=txt,
                policy=(s2_policy if use_policy else ""),
                fewshots=fewshots_s2,
                tech=tech,
                markers_json=json.dumps(mks, ensure_ascii=False),
                prompt_arts=prompt_arts or None,
            )

            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s2 < 3
            ):
                _save_prompt_bundle(tech_dir, "s2", str(_id), sys_blocks, user_block)
                saved_s2 += 1
            if args.print_prompts_preview and not printed_s2_preview:
                print(
                    "[S2 prompt preview]\nSYSTEM:\n"
                    + "\n\n---\n\n".join(sys_blocks)
                    + "\n\nUSER:\n"
                    + user_block
                )
                printed_s2_preview = True

            # self-consistency
            runs = max(1, int(sc_n))
            temp2 = args.sc_temperature if runs > 1 else args.temperature
            preds = []

            for _ in range(runs):
                chat_s2 = Chat(
                    model_id=(args.model_id or Chat.SONNET_45_MODEL_ID),
                    region=args.region,
                    max_tokens=args.max_tokens_s2,
                    temperature=temp2,
                )
                for s in sys_blocks:
                    chat_s2.add_system(s)
                chat_s2.add_user(user_block)
                try:
                    out = chat_s2.generate()
                    # permissive JSON object extraction
                    m = re.search(r"\{.*\}", out, re.S)
                    js = (
                        json.loads(m.group(0))
                        if m
                        else {"label": "non", "confidence": 0.55}
                    )
                except Exception as e:
                    logging.warning(f"[S2] generate() failed -> default non: {e}")
                    js = {"label": "non", "confidence": 0.55}

                # coerce probabilities + label
                label = (js.get("label") or "non").strip().lower()
                p_con = js.get("p_conspiracy")
                p_non = js.get("p_non")

                def _flt(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                p_con = _flt(p_con)
                p_non = _flt(p_non)
                if p_con is None or p_non is None:
                    conf = _flt(js.get("confidence")) or (
                        0.9 if label == "conspiracy" else 0.1
                    )
                    if label == "conspiracy":
                        p_con, p_non = conf, 1.0 - conf
                    else:
                        p_con, p_non = 1.0 - conf, conf
                s = p_con + p_non
                if s <= 0:
                    p_con = p_non = 0.5
                else:
                    p_con, p_non = p_con / s, p_non / s
                if label not in ("conspiracy", "non"):
                    label = "conspiracy" if p_con >= p_non else "non"

                preds.append({"label": label, "p_conspiracy": p_con, "p_non": p_non})

            # average probs; final label = argmax
            avg_con = sum(p["p_conspiracy"] for p in preds) / len(preds)
            avg_non = sum(p["p_non"] for p in preds) / len(preds)
            tot = max(1e-9, avg_con + avg_non)
            avg_con, avg_non = avg_con / tot, avg_non / tot
            final_label = "conspiracy" if avg_con >= avg_non else "non"

            s2_out_rows.append(
                {
                    "_id": _id,
                    "conspiracy": ("Yes" if final_label == "conspiracy" else "No"),
                }
            )
            s2_prob_rows.append(
                {
                    "_id": _id,
                    "label": final_label,
                    "p_conspiracy": round(avg_con, 6),
                    "p_non": round(avg_non, 6),
                }
            )

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
