# starter/llm_infer_binary.py
import argparse, json, orjson, pathlib, re, random, os
from typing import Dict, Any, Optional

# ensure repo root is importable when running from starter/
import sys, pathlib as _pathlib

sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

# NEW: logging setup (put near the imports)
import logging, time

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---- simple .env loader (no deps) ----
def _load_dotenv_into_environ():
    root = _pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            os.environ.setdefault(k, v)
    # map non-standard keys to what boto3 expects
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()
# --------------------------------------

from src.psycomark.llm.bedrock_chat import Chat
from src.psycomark.llm.eda_support import (
    build_s2_policy,
    load_fewshots,
)

YES = "Yes"
NO = "No"

S2_SYS_RULES = """\
You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Goal: decide whether the REDDIT COMMENT promotes a conspiracy narrative.
Use brief private reasoning, but return ONLY the final JSON below.

Label decision rubric:
- "conspiracy": The author asserts or clearly endorses a conspiratorial claim.
- "non": Neutral reporting, critique/debunking, jokes/irony, or unrelated content.
Avoid using subreddit as a proxy; rely on text content and framing.

Probability rubric (p_conspiracy, p_non should sum to 1.0):
- 0.90–1.00: Explicit assertion/endorsement of a conspiracy.
- 0.60–0.80: Strong implication or supportive framing without explicit claim.
- ~0.50: Ambiguous/uncertain.
- 0.00–0.20: Clearly non-conspiratorial (neutral/debunking/irrelevant).

Return strict JSON ONLY:
{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}
Constraints:
- Ensure p_conspiracy + p_non = 1.0 (within rounding). Set label to the higher of the two.
- Keep rationale concise and non-revealing of chain-of-thought.
"""

S2_FEWSHOT = [
    {
        "text": "James Alefantis was identified as central to child trafficking... #pizzagate remains the most critical conspiracy...",
        "gold": {
            "label": "conspiracy",
            "rationale": "Affirms the Pizzagate conspiracy as true.",
            "p_conspiracy": 0.95,
            "p_non": 0.05,
        },
    },
    {
        "text": "Very interesting read on budget deficits. I'm brainstorming for a macroeconomics class.",
        "gold": {
            "label": "non",
            "rationale": "No conspiratorial claim; general discussion.",
            "p_conspiracy": 0.05,
            "p_non": 0.95,
        },
    },
]

S2_USER_TEMPLATE = """\
TASK: Document-level classification (conspiracy vs non). Return JSON only, no extra text.

TEXT:
{doc_text}

REQUIREMENTS:
- Decide based on linguistic content, not subreddit.
- If uncertain, choose "non".
- Keep rationale <= 2 sentences.
- Provide p_conspiracy and p_non such that they sum to 1.0; set label to the higher probability.

RETURN JSON:
{{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"..."}}
"""


def strict_json_extract(s: str) -> Optional[Dict[str, Any]]:
    # grab the first JSON object
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _shorten(txt: str, max_chars: int = 1200) -> str:
    txt = txt or ""
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--submission-file", required=True)
    ap.add_argument("--probs-file", default=None)
    ap.add_argument("--model-id", default=None)  # allow env fallback
    ap.add_argument("--region", default=None)  # allow env fallback
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eda-root", required=False, default=None)
    args = ap.parse_args()
    random.seed(args.seed)

    logging.info(
        f"[S2] model_id={args.model_id} region={args.region} "
        f"max_tokens={args.max_tokens} temperature={args.temperature}"
    )

    # Resolve model/region from flags or env
    model_id = args.model_id or os.environ.get(
        "MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    region = args.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # Load EDA policy + few-shots (optional)
    eda_text = ""
    fewshots = []
    if args.eda_root:
        eda_root = pathlib.Path(args.eda_root)
        logging.info(f"[S2] Loading EDA from: {eda_root}")
        eda_text = build_s2_policy(eda_root) or ""
        fewshots = load_fewshots(eda_root, "s2", max_n=8) or []
        logging.info(f"[S2] fewshots loaded={len(fewshots)} (fallback used if 0)")

    # Build base few-shot block ONCE (prefer EDA-curated, else fallback)
    examples = fewshots if fewshots else S2_FEWSHOT
    base_fewshot_block = ""
    for ex in examples:
        txt = ex.get("text") or ex.get("doc_text") or ""
        gold = (
            ex.get("gold")
            or ex.get("json")
            or {"label": "non", "rationale": "baseline", "confidence": 0.7}
        )
        base_fewshot_block += (
            f"\n\nEXAMPLE:\nTEXT:\n{_shorten(txt)}\n"
            f"JSON:\n{json.dumps(gold, ensure_ascii=False)}\n"
        )

    logging.info(f"[S2] fewshot_block_len_chars={len(base_fewshot_block)}")

    sub_path = pathlib.Path(args.submission_file)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    probs_path = pathlib.Path(args.probs_file) if args.probs_file else None
    if probs_path:
        probs_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"[S2] will write probs to: {probs_path}")

    n_docs = 0
    n_parse_ok = 0
    t0_all = time.time()

    with open(args.test_file, "r", encoding="utf-8") as fi, open(
        args.submission_file, "w", encoding="utf-8"
    ) as fo, (
        open(args.probs_file, "w", encoding="utf-8")
        if probs_path
        else open(os.devnull, "w")
    ) as fp:

        logging.info(f"[S2] reading test file: {args.test_file}")

        for line in fi:
            if not line.strip():
                continue
            rec = orjson.loads(line)
            _id = rec.get("_id") or rec.get("doc_id")
            txt = rec.get("text", "")
            n_docs += 1
            t0 = time.time()

            # fresh convo per doc → avoids cross-doc bleed
            convo = Chat(
                model_id=model_id,
                region=region,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            if eda_text:
                convo.add_system(eda_text)  # policy card from EDA
            convo.add_system(S2_SYS_RULES)  # fixed rules for S2

            user = S2_USER_TEMPLATE.format(doc_text=_shorten(rec.get("text", "")))
            convo.add_user(base_fewshot_block + "\n\n" + user)

            try:
                out = convo.generate()
            except Exception:
                logging.exception(f"[S2] generation failed _id={_id}")
                # write a safe default and continue
                fo.write(orjson.dumps({"_id": _id, "conspiracy": "No"}).decode() + "\n")
                if probs_path:
                    fp.write(
                        orjson.dumps(
                            {"_id": _id, "p_non": 0.5, "p_conspiracy": 0.5}
                        ).decode()
                        + "\n"
                    )
                continue

            parsed = strict_json_extract(out) or {
                "label": "non",
                "rationale": "repair",
                "confidence": 0.51,
            }

            if parsed is None:
                logging.warning(f"[S2] JSON repair for _id={_id}")
                parsed = {"label": "non", "rationale": "repair", "confidence": 0.51}
            else:
                n_parse_ok += 1

            lab = (parsed.get("label") or "non").strip().lower()
            pred_is_consp = lab == "conspiracy"
            pred = YES if pred_is_consp else NO

            fo.write(
                orjson.dumps({"_id": _id, "conspiracy": pred}).decode("utf-8") + "\n"
            )

            if probs_path:
                # Prefer explicit dual probabilities; else derive from confidence conditioned on predicted label
                p_consp = parsed.get("p_conspiracy")
                p_non = parsed.get("p_non")
                try:
                    if p_consp is not None and p_non is not None:
                        p_consp = float(p_consp)
                        p_non = float(p_non)
                        total = p_consp + p_non
                        if total <= 1e-9:
                            raise ValueError("invalid prob sum")
                        p_consp = max(0.0, min(1.0, p_consp / total))
                        p_non = 1.0 - p_consp
                    else:
                        raise ValueError("missing dual probs")
                except Exception:
                    raw_conf = float(parsed.get("confidence", 0.5))
                    raw_conf = max(0.0, min(1.0, raw_conf))
                    p_consp = raw_conf if pred_is_consp else 1.0 - raw_conf
                    p_non = 1.0 - p_consp
                fp.write(
                    orjson.dumps(
                        {"_id": _id, "p_non": p_non, "p_conspiracy": p_consp}
                    ).decode("utf-8")
                    + "\n"
                )

            if n_docs % 50 == 0:
                logging.info(
                    f"[S2] progress: {n_docs} docs, last_doc_ms={(time.time()-t0)*1000:.0f}"
                )

    logging.info(
        f"[S2] done. docs={n_docs} parse_ok={n_parse_ok} "
        f"wall_sec={time.time()-t0_all:.2f} "
        f"sub_file={args.submission_file} probs_file={args.probs_file}"
    )


if __name__ == "__main__":
    main()
