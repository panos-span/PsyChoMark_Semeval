#!/usr/bin/env python3
# tools/make_prompt_artifacts.py
import json

from pathlib import Path

ALLOWED = ["Actor", "Action", "Effect", "Victim", "Evidence"]

ABSOLUTIST = [
    "always",
    "never",
    "everyone",
    "no one",
    "impossible",
    "undeniable",
    "without a doubt",
    "completely",
    "totally",
    "entirely",
    "absolutely",
    "certainly",
    "no doubt",
    "no doubts",
]
HEDGES = [
    "maybe",
    "perhaps",
    "possibly",
    "likely",
    "unlikely",
    "appears",
    "seems",
    "suggests",
    "might",
    "could",
    "may",
    "arguably",
]


def _latest_dir(ptr: Path) -> Path:
    d = Path(ptr.read_text().strip())
    if not d.exists():
        raise FileNotFoundError(d)
    return d


def _safe_load(path: Path, default):
    try:
        return (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        )
    except Exception:
        return default


def _top_pairs(overlap_stats: dict, k=2):
    items = []
    for pair, d in (overlap_stats or {}).items():
        items.append((pair, float(d.get("iou@0.5", 0.0))))
    items.sort(key=lambda x: x[1], reverse=True)
    res = []
    for p, _ in items[:k]:
        a, b = p.split("/")
        if a in ALLOWED and b in ALLOWED and a != b:
            res.append([a, b])
    return res


def _boundary_prompts(boundary_ctx: dict, top=2):
    out = {}
    for lab in ALLOWED:
        d = boundary_ctx.get(lab, {}) if boundary_ctx else {}
        out[lab] = {
            "before_1w": (d.get("before_1w", [])[:top]),
            "after_1w": (d.get("after_1w", [])[:top]),
            "before_2w": (d.get("before_2w", [])[:top]),
            "after_2w": (d.get("after_2w", [])[:top]),
        }
    return out


def _priors_prompt(length_position_priors: dict):
    # Expect shape: {Label: {"length_lognorm":{"mu":...,"sigma":...}, "start_beta":{"alpha":...,"beta":...}, "q50_len":..., "q90_len":...}}
    out = {}
    for lab in ALLOWED:
        p = (length_position_priors or {}).get(lab, {})
        q50 = p.get("q50_len") or p.get("q50")
        q90 = p.get("q90_len") or p.get("q90")
        mode_pos = None
        sb = p.get("start_beta") or {}
        a, b = sb.get("alpha"), sb.get("beta")
        if (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and a > 1
            and b > 1
        ):
            mode_pos = (a - 1) / (a + b - 2)
        out[lab] = {
            "q50_len": float(q50) if q50 else None,
            "q90_len": float(q90) if q90 else None,
            "mode_pos": float(mode_pos) if mode_pos is not None else None,
        }
    return out


def _fewshot_bank(latest: Path):
    bank = {"s1": [], "s2": []}
    # Prefer your curated best fewshots if exist
    best = _safe_load(latest / "best_fewshot_examples.json", {"s1": [], "s2": []})
    for ex in best.get("s1", []):
        bank["s1"].append(
            {
                "id": ex.get("doc_id"),
                "task": "s1",
                "difficulty": "mixed",
                "text": ex.get("text", ""),
                "json": ex.get("spans") or ex.get("json") or [],
            }
        )
    for ex in best.get("s2", []):
        bank["s2"].append(
            {
                "id": ex.get("doc_id"),
                "task": "s2",
                "difficulty": "mixed",
                "text": ex.get("text", ""),
                "json": ex.get("json")
                or {
                    "label": ex.get("label", "non"),
                    "rationale": ex.get("rationale", ""),
                },
            }
        )
    # Ensure at least one negative S1 exemplar (empty)
    if not any(
        isinstance(x.get("json"), list) and len(x["json"]) == 0 for x in bank["s1"]
    ):
        # pick a short dev row with no markers
        dev = _safe_load(latest / "dev.jsonl", [])
        fallback = None
        if dev and isinstance(dev, list):
            for r in dev:
                if not r.get("markers"):
                    t = (r.get("text") or "")[:320]
                    if t:
                        fallback = {
                            "id": r.get("_id") or r.get("doc_id"),
                            "task": "s1",
                            "difficulty": "easy",
                            "text": t,
                            "json": [],
                        }
                        break
        if fallback:
            bank["s1"].insert(0, fallback)
    return bank


def main():
    root = Path("data/derived/psycomark_latest.txt")
    latest = _latest_dir(root)
    # 1) lexicons
    (latest / "lexicons.json").write_text(
        json.dumps({"ABSOLUTIST": ABSOLUTIST, "HEDGES": HEDGES}, indent=2),
        encoding="utf-8",
    )
    # 2) conflicts
    stats = _safe_load(latest / "overlap_pair_stats_ci.json", {})
    if not stats:
        stats = _safe_load(latest / "overlap_pair_stats.json", {})
    pairs = _top_pairs(stats, k=2)
    (latest / "conflicts.json").write_text(
        json.dumps({"pairs": pairs}, indent=2), encoding="utf-8"
    )
    # 3) boundary prompts
    boundary_ctx = _safe_load(latest / "boundary_context.json", {})
    bp = _boundary_prompts(boundary_ctx, top=2)
    (latest / "boundary_prompts.json").write_text(
        json.dumps(bp, indent=2), encoding="utf-8"
    )
    # 4) priors prompt
    pri = _safe_load(latest / "length_position_priors.json", {})
    pp = _priors_prompt(pri)
    (latest / "priors_prompt.json").write_text(
        json.dumps(pp, indent=2), encoding="utf-8"
    )
    # 5) fewshot bank
    fb = _fewshot_bank(latest)
    (latest / "fewshot_bank.json").write_text(
        json.dumps(fb, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[make_prompt_artifacts] wrote: lexicons.json, conflicts.json, boundary_prompts.json, priors_prompt.json, fewshot_bank.json in {latest}"
    )


if __name__ == "__main__":
    main()
