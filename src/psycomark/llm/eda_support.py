# src/llm/eda_support.py
import csv
import json
import pathlib
import random
from typing import Any, Dict, List, Tuple


def load_json(p: pathlib.Path, default):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def load_csv_top_terms(p: pathlib.Path, k=12) -> List[Tuple[str, float]]:
    # Expect lexical_effect_sizes.csv with columns: term,effect_size
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append((row["term"], float(row["effect_size"])))
            except Exception:
                continue
    # sort by |effect| descending, keep top-k
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:k]


def summarize_beta(alpha: float, beta: float) -> str:
    # Convert Beta(alpha,beta) to a plain-English position prior
    if alpha <= 0 or beta <= 0:
        return "no clear start-position prior"
    mean = alpha / (alpha + beta)
    if mean < 0.33:
        zone = "EARLY"
    elif mean < 0.66:
        zone = "MIDDLE"
    else:
        zone = "LATE"
    return f"start tends to {zone.lower()} (mean≈{mean:.2f})"


def build_s1_policy(eda_root: pathlib.Path) -> str:
    priors = load_json(eda_root / "length_position_priors.json", {})
    pairs_ci = load_json(eda_root / "overlap_pair_stats_ci.json", {})
    lines = []

    # Length + position priors
    for lab, d in (priors or {}).items():
        if not isinstance(d, dict):
            continue
        q90 = d.get("q90_len", 0)
        startb = d.get("start_beta", {})
        alpha, beta = startb.get("alpha", 0), startb.get("beta", 0)
        pos_text = summarize_beta(alpha, beta)
        if q90:
            lines.append(
                f"- {lab}: typical span length ≤ q90 ≈ {q90} chars; {pos_text}."
            )
        else:
            lines.append(f"- {lab}: {pos_text}.")

    # Known-confusing overlaps (Action↔Effect, Actor↔Victim)
    rules = [
        "- If Action and Effect overlap heavily (IoU≥0.6), prefer the label whose start aligns better with its prior; if uncertain, keep both but avoid duplicate substrings.",
        "- For nested Actor/Victim, prefer the **smaller** contained span; if ambiguous starts, Actor earlier, Victim later.",
        "- Evidence can cover a clause/sentence; avoid relabeling it as Action/Effect unless it is the **minimal** trigger.",
    ]
    # Add note if we have pair stats
    if pairs_ci:
        rules.append(
            "- Corpus overlap stats loaded; apply tie-breaks using priors above."
        )

    header = (
        "S1 PRIORS & TIE-BREAKS:\n" + "\n".join(lines[:10]) + "\n" + "\n".join(rules)
    )
    return header.strip()


def build_s2_policy(eda_root: pathlib.Path) -> str:
    # Absolutist/hedge summaries + lexical signals
    terms = load_csv_top_terms(eda_root / "lexical_effect_sizes.csv", k=10)
    tips = [
        "- Label **conspiracy** if the author asserts/endorses a conspiratorial claim (hidden cabal/cover-up/coordinated deception).",
        "- Label **non** for neutral reporting, criticism of conspiracies, jokes, or unrelated topics.",
        "- Do NOT use subreddit as a proxy; base decision on TEXT only.",
        "- Hedging and uncertainty words **reduce** likelihood of conspiracy; absolutist phrases often **increase** it, but content dominates.",
    ]
    if terms:
        phr = "; ".join([f"{t} ({'+' if s>0 else ''}{s:.2f})" for t, s in terms])
        tips.append(
            f"- Top lexical signals (effect size): {phr} (use as weak hints only)."
        )
    return "S2 DECISION TIPS:\n" + "\n".join(tips)


def load_fewshots(
    eda_root: pathlib.Path, for_task: str, max_n: int = 8
) -> List[Dict[str, Any]]:
    # best_fewshot_examples.json format assumed: {"s1":[...], "s2":[...]}
    print(f"Loading few-shot examples for task '{for_task}' from EDA root: {eda_root}")
    p = eda_root / "best_fewshot_examples.json"
    data = load_json(p, {})
    arr = (data or {}).get(for_task, [])
    random.shuffle(arr)
    return arr[:max_n]
