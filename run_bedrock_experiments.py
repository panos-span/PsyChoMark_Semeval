# run_bedrock_experiments.py
"""
End-to-end, fast Bedrock runner for PsyCoMark.

Key improvements:
- Loads ACCESS_KEY_ID / SECRET_ACCESS_KEY / SESSION_TOKEN / MODEL_ID from .env (with python-dotenv).
- Supports both base model IDs and Inference Profile IDs in modelId.
- Single-call (joint) inference per doc for S1+S2; optional classification-only call when HF says "non" confidently.
- Tight prompts with stop sequences; small max_tokens.
- Concurrency via ThreadPoolExecutor.
- Optional gating using HF doc probabilities (skip S1 extraction when prob_conspiracy <= threshold).
- Robust JSON parsing/repair; caching responses to disk for reproducibility and cost control.

Usage examples:
    uv run ./run_bedrock_experiments.py
    uv run ./run_bedrock_experiments.py --model-id anthropic.claude-3-haiku-20240307-v1:0 --region eu-central-1 --concurrency 8
    uv run ./run_bedrock_experiments.py --hf-probs data/derived/.../hf_doc_probs.jsonl --gate-threshold 0.2
"""

import os
import sys
import json
import time
import hashlib
import logging
from tqdm import tqdm
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd

# --- .env loading (dev convenience; prefer IAM roles in prod) ---
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# ==============================================================================
# ENV HELPERS & JSON PARSING (Unchanged from your version)
# ==============================================================================
def load_env_and_map_for_boto3() -> str:
    if load_dotenv and find_dotenv:
        load_dotenv(find_dotenv(), override=False)
    ak, sk, st, rg = (
        os.getenv(k) or os.getenv(v)
        for k, v in [
            ("ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
            ("SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"),
            ("SESSION_TOKEN", "AWS_SESSION_TOKEN"),
            ("AWS_DEFAULT_REGION", "AWS_DEFAULT_REGION"),
        ]
    )
    if ak and not os.getenv("AWS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = ak
    if sk and not os.getenv("AWS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = sk
    if st and not os.getenv("AWS_SESSION_TOKEN"):
        os.environ["AWS_SESSION_TOKEN"] = st
    os.environ["AWS_DEFAULT_REGION"] = rg or "eu-central-1"
    return os.getenv("MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")


def parse_safe(text: str):
    import re

    text = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JsonExtractError("No JSON object found.")
    block = match.group(0)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[\}\]])", r"\1", block)
        return json.loads(repaired)


# ==============================================================================
# JSON PARSING / REPAIR (robust to markdown fences & minor commas)
# ==============================================================================
class JsonExtractError(Exception):
    pass


def strip_md_fences(s: str) -> str:
    return s.replace("```json", "").replace("```", "").strip()


def first_json_block(text: str):
    import re

    text = strip_md_fences(text)
    arr = re.search(r"\[.*\]", text, re.DOTALL)
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    # prefer object first for joint outputs
    if obj:
        return obj.group(0), "object"
    if arr:
        return arr.group(0), "array"
    raise JsonExtractError("No JSON block found.")


# ==============================================================================
# Bedrock Chat wrapper (system prompts, stop sequences, reset per call)
# ==============================================================================
class Chat:
    def __init__(
        self,
        model_id,
        client,
        temperature=0.0,
        max_tokens=600,
        system_text=None,
        stop_sequences=None,
    ):
        (
            self.model_id,
            self.client,
            self.temperature,
            self.max_tokens,
            self.system_text,
            self.stop_sequences,
        ) = (
            model_id,
            client,
            temperature,
            max_tokens,
            system_text,
            stop_sequences or [],
        )

    def _payload(self, user_text: str):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            ],
        }
        if self.system_text:
            body["system"] = self.system_text
        if self.stop_sequences:
            body["stop_sequences"] = self.stop_sequences
        return body

    def generate(self, user_text: str, retries=3, backoff=1.7) -> str:
        body = self._payload(user_text)
        for i in range(retries):
            try:
                resp = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                out = json.loads(resp["body"].read())
                return out.get("content", [{}])[0].get("text", "")
            except Exception as e:
                logging.warning(f"Bedrock call failed ({i+1}/{retries}): {e}")
                time.sleep(backoff**i)
        return ""


# run_bedrock_experiments_final_v2.py
import os, sys, json, time, hashlib, logging, argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import boto3
import pandas as pd

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = find_dotenv = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ... [ All helper functions from the top (load_env, parse_safe, Chat, HFGate, cache functions) are correct. No changes needed there. ] ...
# ... [ Paste them here ] ...


# = an=============================================================================
# *** NEW HELPER FUNCTION TO FIX THE DATA FORMAT MISMATCH ***
# ==============================================================================
def _coerce_few_shot_examples(examples) -> list[dict]:
    """
    Normalises few-shot data into the combined schema this runner expects (a list of dicts).
    This handles the case where the input JSON is a dict with 's1' and 's2' keys.
    """
    if isinstance(examples, list):
        return examples  # Already in the correct format

    if not isinstance(examples, dict):
        logging.warning(
            "Few-shot examples were neither list nor dict; defaulting to empty list."
        )
        return []

    # Merge s1 and s2 data based on the text content
    combined_examples = {}
    for item in examples.get("s1", []):
        text = item.get("text")
        if text and text not in combined_examples:
            combined_examples[text] = {"text": text, "markers": item.get("markers", [])}

    for item in examples.get("s2", []):
        text = item.get("text")
        if text and text not in combined_examples:
            combined_examples[text] = {"text": text}  # Initialize if not seen in s1

        # Add classification info
        combined_examples[text]["doc_label"] = item.get("doc_label", "non")
        combined_examples[text]["rationale"] = item.get("rationale", "")

    return list(combined_examples.values())


# ==============================================================================
# *** NEW & IMPROVED PROMPT BUILDERS ***
# ==============================================================================
def build_prompts(fewshots: list | None) -> tuple[str, str]:
    """Builds both the joint and classify-only prompts using our advanced CoT logic."""
    fewshots = fewshots or []

    # --- Helper to create a single example block ---
    def create_example_str(ex: dict, is_joint: bool) -> str:
        markers_out = [
            {"label": m["label"], "start": m["start"], "end": m["end"]}
            for m in ex.get("markers", [])
        ]
        class_out = {
            "label": ex.get("doc_label", "non"),
            "rationale": ex.get("rationale", ""),
        }

        output_json = (
            {"classification": class_out, "spans": markers_out}
            if is_joint
            else class_out
        )

        # This is our insight-driven, position-aware CoT
        scratchpad = (
            (
                f"1.  **Initial Read & Classification:** The document appears to be '{class_out['label']}'. Rationale: {class_out['rationale']}\n"
                f"2.  **Positional Scan for Spans (Narrative Arc):**\n"
                f"    - **Beginning:** I will look for the main `Actor` and their `Action` early in the text.\n"
                f"    - **Middle:** I will scan the middle for supporting `Evidence` and identifying the `Victim`.\n"
                f"    - **End:** I will look for the final negative consequence or `Effect` towards the end of the text.\n"
                f"3.  **Final JSON Construction:** Based on this structured analysis, I will now build the complete JSON object."
            )
            if is_joint
            else f"1. **Analyze Classification:** The document is '{class_out['label']}' because {class_out['rationale']}"
        )

        return (
            f"<example>\n"
            f"<document>\n{ex.get('text', '').strip()}\n</document>\n"
            f"<scratchpad>\n{scratchpad}\n</scratchpad>\n"
            f"<output>\n{json.dumps(output_json, indent=2)}\n</output>\n"
            f"</example>\n"
        )

    # --- Build JOINT Prompt ---
    examples_str_joint = "\n".join(
        [create_example_str(ex, is_joint=True) for ex in fewshots]
    )
    prompt_joint = (
        "You are an expert NLP analyst. Perform two tasks: classification and span extraction.\n"
        "First, think step-by-step in a private <scratchpad> following the narrative arc. Then, produce a single JSON object in an <output> section.\n\n"
        "<rules>\n"
        "1. Your final response MUST contain ONLY the <output> section with the JSON object.\n"
        "2. The JSON must have two keys: 'classification' (object) and 'spans' (list).\n"
        "3. If no spans, `spans` must be an empty list: `[]`.\n"
        "</rules>\n\n"
        "Here are examples:\n"
        f"{examples_str_joint}\n"
        "Now, analyze the following document.\n\n"
        "<document>\n{TEXT}\n</document>"
    )

    # --- Build CLASSIFY-ONLY Prompt (for gating) ---
    examples_str_classify = "\n".join(
        [create_example_str(ex, is_joint=False) for ex in fewshots]
    )
    prompt_classify = (
        "You are an expert analyst classifying a document as 'conspiracy' or 'non'.\n"
        "First, think in a <scratchpad>. Then, produce a single JSON object in an <output> section.\n\n"
        "<rules>\n"
        "1. Your final response MUST contain ONLY the <output> section with the JSON object.\n"
        "2. The JSON must have two keys: 'label' and 'rationale'.\n"
        "</rules>\n\n"
        "Here are examples:\n"
        f"{examples_str_classify}\n"
        "Now, analyze the following document.\n\n"
        "<document>\n{TEXT}\n</document>"
    )

    return prompt_joint, prompt_classify


# ==============================================================================
# Prompt builders (tight, fast)
# ==============================================================================
SYSTEM_TEXT_JOINT = (
    "You extract spans and classify documents. Follow rules exactly. Be concise."
)

JOINT_PROMPT_TMPL = (
    "Given <doc>…</doc>, do TWO tasks:\n"
    "1) Extract spans for labels: Actor, Action, Effect, Victim, Evidence.\n"
    "2) Classify the document as 'conspiracy' or 'non'.\n\n"
    "RULES:\n"
    "- Return ONLY a valid JSON object:\n"
    '  {"spans":[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}, ...],\n'
    '   "doc":{"label":"conspiracy|non","rationale":"<=2 sentences"}}\n'
    "- Offsets are 0-based char indices on EXACT text.\n"
    "- Overlaps/nesting allowed. If no spans, use [] (max 8 spans).\n"
    "- No extra text outside JSON.\n\n"
    "{EXAMPLES_BLOCK}"
    "<doc>\n{TEXT}\n</doc>\n<output>\n"
)

CLASSIFY_ONLY_TMPL = (
    "Classify the document as 'conspiracy' or 'non'.\n"
    'Return ONLY JSON: {"label":"conspiracy|non","rationale":"<=2 sentences"}\n\n'
    "{EXAMPLES_BLOCK}"
    "<doc>\n{TEXT}\n</doc>\n<output>\n"
)


def build_examples_block(fewshots, n_s1=2, n_s2=2, max_chars_per_example=800):
    """
    Build a compact examples block from best_fewshot_examples.json.
    We include up to n_s1 span examples and n_s2 doc examples.
    """
    if not fewshots:
        return ""
    s1 = fewshots.get("s1", [])[: max(0, n_s1)]
    s2 = fewshots.get("s2", [])[: max(0, n_s2)]

    parts = []
    if s1:
        parts.append("Examples (spans):")
        for ex in s1:
            text = ex["text"]
            if len(text) > max_chars_per_example:
                text = text[:max_chars_per_example] + "..."
            # Keep only label/start/end for brevity
            m = [
                {"label": m["label"], "start": int(m["start"]), "end": int(m["end"])}
                for m in ex["markers"][:1]
            ]
            parts.append(
                f"<example>\n<document>\n{text}\n</document>\n<output>\n{json.dumps(m)}\n</output>\n</example>\n"
            )

    if s2:
        parts.append("Examples (classification):")
        for ex in s2:
            text = ex["text"]
            if len(text) > max_chars_per_example:
                text = text[:max_chars_per_example] + "..."
            out = {"label": ex["doc_label"], "rationale": ex.get("rationale", "")[:200]}
            parts.append(
                f"<example>\n<document>\n{text}\n</document>\n<output>\n{json.dumps(out)}\n</output>\n</example>\n"
            )

    return "\n".join(parts) + ("\n" if parts else "")


# ==============================================================================
# Caching
# ==============================================================================
def cache_read(cache_dir: Path, key: str) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".txt")
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if text.strip():
            return text
        # Prune empty/failed cache entries so we can retry the call.
        try:
            path.unlink()
        except OSError:
            pass
    return None


def cache_write(cache_dir: Path, key: str, text: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".txt")
    if text.strip():
        path.write_text(text, encoding="utf-8")


# ==============================================================================
# Gating using HF doc probabilities (optional)
# ==============================================================================
class HFGate:
    def __init__(self, hf_probs_path: Path | None, threshold: float):
        self.active = hf_probs_path is not None
        self.th = threshold
        self.p = {}
        if self.active:
            dfp = pd.read_json(hf_probs_path, lines=True)
            # expect columns: doc_id, prob_conspiracy
            for _, r in dfp.iterrows():
                self.p[r["doc_id"]] = float(r.get("prob_conspiracy", 0.5))

    def skip_spans(self, doc_id) -> bool:
        if not self.active:
            return False
        prob = self.p.get(doc_id, None)
        return (prob is not None) and (prob <= self.th)


# ==============================================================================
# Worker
# ==============================================================================
# ==============================================================================
# *** REVISED WORKER ***
# ==============================================================================
def render_prompt(template: str, text: str) -> str:
    """Safely injects document text into a prompt template containing {TEXT}."""
    if "{TEXT}" not in template:
        raise ValueError("Prompt template missing {TEXT} placeholder.")
    return template.replace("{TEXT}", text)


def run_one(
    doc_row,
    client,
    model_id,
    prompt_joint,
    prompt_classify,
    cache_dir: Path,
    gate: HFGate,
    max_tokens_joint: int,
    max_tokens_class: int,
    stop_sequences: list[str],
):
    doc_id, text = doc_row["doc_id"], doc_row["text"]

    # --- Step 1: Decide which prompt to use (Gating) ---
    use_classify_only = gate.skip_spans(doc_id)
    if use_classify_only:
        prompt = render_prompt(prompt_classify, text)
        system_text = "You classify documents into 'conspiracy' or 'non'."
        max_tokens = max_tokens_class
        cache_key_type = "CLASSIFY_ONLY"
    else:
        prompt = render_prompt(prompt_joint, text)
        system_text = "You extract spans and classify documents."
        max_tokens = max_tokens_joint
        cache_key_type = "JOINT"

    # --- Step 2: Check cache or call API ---
    cache_key = (
        f"{model_id}::{cache_key_type}::{hashlib.sha256(prompt.encode()).hexdigest()}"
    )
    raw = cache_read(cache_dir, cache_key)
    if raw is None:
        chat = Chat(
            model_id,
            client,
            temperature=0.0,
            max_tokens=max_tokens,
            system_text=system_text,
            stop_sequences=stop_sequences,
        )
        raw = chat.generate(prompt)
        if raw.strip():
            cache_write(cache_dir, cache_key, raw)
        else:
            logging.warning(
                "Bedrock returned an empty response; skipping cache write for %s",
                cache_key_type,
            )

    # --- Step 3: Parse and standardize the output ---
    prediction, error = None, None
    try:
        parsed = parse_safe(raw)
        if use_classify_only:
            # If we only classified, create the full structure with empty spans
            prediction = {"spans": [], "doc": parsed}
        else:
            # If joint, ensure both keys are present
            prediction = {
                "spans": parsed.get("spans", []),
                "doc": parsed.get("doc", {}),
            }
    except Exception as e:
        error = str(e)
        prediction = {
            "spans": [],
            "doc": {"label": "non", "rationale": "Fallback due to parsing error."},
        }

    # Final safety clamps on the structure
    if not isinstance(prediction.get("spans"), list):
        prediction["spans"] = []
    if not isinstance(prediction.get("doc"), dict):
        prediction["doc"] = {}
    if prediction["doc"].get("label") not in {"conspiracy", "non"}:
        prediction["doc"]["label"] = "non"

    return {"doc_id": doc_id, "prediction": prediction, "raw": raw, "error": error}


# ==============================================================================
# Main
# ==============================================================================
def main():
    # Load env, map creds, pick default model from .env if set
    default_model_id = load_env_and_map_for_boto3()
    logging.info("Found credentials in environment variables.")

    ap = argparse.ArgumentParser(
        description="Fast PsyCoMark Bedrock runner (joint S1+S2, concurrency, gating)."
    )
    ap.add_argument(
        "--model-id",
        default=default_model_id,
        help="Bedrock model ID or Inference Profile ID.",
    )
    ap.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--max-docs", type=int, default=0, help="For quick dry runs; 0 = all."
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("./.bedrock_cache"))
    ap.add_argument(
        "--stop",
        nargs="*",
        default=["</output>"],
        help="Stop sequences to truncate generation.",
    )
    ap.add_argument("--max-tokens-joint", type=int, default=600)
    ap.add_argument("--max-tokens-class", type=int, default=160)
    ap.add_argument(
        "--fewshots-path",
        type=Path,
        default=None,
        help="Path to best_fewshot_examples.json",
    )
    ap.add_argument(
        "--s1-shots",
        type=int,
        default=2,
        help="Span examples to include (kept tiny for speed).",
    )
    ap.add_argument(
        "--s2-shots", type=int, default=2, help="Classification examples to include."
    )
    ap.add_argument(
        "--hf-probs",
        type=Path,
        default=None,
        help="Optional HF doc probs jsonl for gating spans.",
    )
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=0.2,
        help="Skip S1 when prob_conspiracy <= threshold.",
    )
    args = ap.parse_args()

    # Bedrock client
    try:
        client = boto3.client(service_name="bedrock-runtime", region_name=args.region)
    except Exception as e:
        logging.error(
            "Failed to create Bedrock client. Check AWS env/.env credentials and region."
        )
        logging.error(e)
        sys.exit(1)

    # Locate latest derived data dir (created by data_pipeline.py)
    latest_ptr = Path("./data/derived/psycomark_latest.txt")
    if not latest_ptr.exists():
        raise FileNotFoundError(
            "Run data_pipeline.py first to create data/derived/* and psycomark_latest.txt."
        )
    outdir = Path(latest_ptr.read_text().strip())
    dev_path = outdir / "dev.jsonl"
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing dev.jsonl under {outdir}")

    # Few-shots (optional but helpful). Expect {"s1":[...], "s2":[...]}.
    raw_fewshots = None
    fewshot_path_to_load = args.fewshots_path or outdir / "best_fewshot_examples.json"
    if fewshot_path_to_load.exists():
        with open(fewshot_path_to_load, "r", encoding="utf-8") as f:
            # This correctly loads the list of dicts
            raw_fewshots = json.load(f)

        # COERCE the loaded data into the format our prompt builder expects
        fewshots_for_prompting = _coerce_few_shot_examples(raw_fewshots)
        logging.info(
            f"Loaded and normalized {len(fewshots_for_prompting)} few-shot examples from {fewshot_path_to_load}"
        )
    else:
        logging.warning("No few-shot examples found; proceeding zero-shot.")

    prompt_joint_template, prompt_classify_template = build_prompts(
        fewshots_for_prompting
    )

    # Load dev set
    dev_df = pd.read_json(dev_path, lines=True)
    N = len(dev_df) if args.max_docs <= 0 else min(args.max_docs, len(dev_df))
    logging.info(
        f"Region={args.region} | model_id={args.model_id} | docs={N} | workers={args.concurrency}"
    )

    # Optional HF gating
    gate = HFGate(args.hf_probs, args.gate_threshold)
    if gate.active:
        logging.info(
            f"HF Gating is ACTIVE. Will skip S1 extraction when prob_conspiracy <= {gate.th}"
        )

    # Inference (parallel)
    jobs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        # Note: iterrows() returns a tuple (index, Series), so we pass row directly
        for _, row in dev_df.head(N).iterrows():
            jobs.append(
                ex.submit(
                    run_one,
                    row,
                    client,
                    args.model_id,
                    prompt_joint_template,  # Correct argument
                    prompt_classify_template,  # Correct argument
                    args.cache_dir,
                    gate,
                    args.max_tokens_joint,
                    args.max_tokens_class,
                    args.stop,
                )
            )

        # Use a list to store results in order to process them later
        results_list = [
            future.result()
            for future in tqdm(
                as_completed(jobs), total=len(jobs), desc="Bedrock Inference"
            )
        ]

    # ==============================================================
    # FINAL ENHANCEMENT: Unpack joint results into separate S1 & S2 files
    # ==============================================================

    preds_s1 = []
    preds_s2 = []
    for res in results_list:
        # Subtask 1 (Spans)
        preds_s1.append(
            {
                "doc_id": res["doc_id"],
                "prediction": res["prediction"]["spans"],
                "raw": res["raw"],
                "error": res["error"],
            }
        )
        # Subtask 2 (Classification)
        preds_s2.append(
            {
                "doc_id": res["doc_id"],
                "prediction": res["prediction"]["doc"],
                "raw": res["raw"],
                "error": res["error"],
            }
        )

    # Save S1 and S2 outputs
    model_safe = args.model_id.split(".")[-1].replace(":", "_")

    out_path_s1 = outdir / f"bedrock_preds_s1_{model_safe}.jsonl"
    pd.DataFrame(preds_s1).to_json(out_path_s1, orient="records", lines=True)

    out_path_s2 = outdir / f"bedrock_preds_s2_{model_safe}.jsonl"
    pd.DataFrame(preds_s2).to_json(out_path_s2, orient="records", lines=True)

    logging.info(f"Saved S1 (spans) predictions to: {out_path_s1}")
    logging.info(f"Saved S2 (classification) predictions to: {out_path_s2}")
    logging.info("Outputs are now ready for the official evaluation script.")


if __name__ == "__main__":
    # Ensure you have the full script content above this line
    main()
