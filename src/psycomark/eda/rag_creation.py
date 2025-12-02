#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prompt_artifacts_.py

A unified, batch-optimized artifact builder for SemEval-2026 PsyCoMark.
Handles RAG corpus creation (S1 & S2) using AWS Bedrock Batch Inference
for both text generation (Rationales/Whys) and embeddings (Titan V2).

Author: geofila (Student ID)
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import boto3
import chromadb
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def _load_dotenv_into_environ():
    """
    Loads environment variables from a .env file into os.environ.
    This allows boto3 to automatically pick up AWS_ACCESS_KEY_ID, etc.
    """
    # Look for .env in current dir or parent dirs
    curr = Path(__file__).resolve().parent
    env_path = None

    # Check current dir and up to 3 parents
    for _ in range(4):
        candidate = curr / ".env"
        if candidate.exists():
            env_path = candidate
            break
        curr = curr.parent

    if env_path:
        print(f"[env] Loading environment from {env_path}")
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                # Strip quotes if present
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k.strip(), v)
        except Exception as e:
            print(f"[env] Warning: Failed to read .env: {e}")
    else:
        print("[env] No .env file found. Assuming credentials are in env vars.")

    # Map generic names to AWS specific ones if needed
    alias = {
        "ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "SESSION_TOKEN": "AWS_SESSION_TOKEN",
        "REGION": "AWS_DEFAULT_REGION",
    }
    for src, dst in alias.items():
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]


# EXECUTE IMMEDIATELY
_load_dotenv_into_environ()

# ---------------------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------------------

DEFAULT_BUCKET = "ails-ntua-bedrock-batch-inference"
DEFAULT_ROLE_ARN = "arn:aws:iam::094808042282:role/Batch_Inference_Role"
STUDENT_ID = "geofila"  # Default Student ID

# Models
MODEL_EMBED = "amazon.titan-embed-text-v2:0"

# Paths
DATA_DIR = Path("data")
RAG_DIR = DATA_DIR / "rag"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BEDROCK_MODEL_ID = os.getenv("MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")


# ---------------------------------------------------------------------
# 2. AWS Batch & S3 Helpers
# ---------------------------------------------------------------------


def get_job_name(task: str, version: str = "v1") -> str:
    """
    Format: <task>-<student_id>-<date>-<version>
    Example: SEMEVAL_S1_WHYS-geofila-2025_03_01-v1
    """
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    return f"{task}-{STUDENT_ID}-{date_str}-{version}"


def upload_to_s3(local_path: Path, bucket: str, key: str):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    logger.info(f"[S3] Uploading {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def download_from_s3(bucket: str, key_prefix: str, local_dir: Path):
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[S3] Downloading from s3://{bucket}/{key_prefix} to {local_dir}...")

    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                # Calculate local path
                rel_path = key.replace(key_prefix, "").lstrip("/")
                if not rel_path:
                    # If prefix is a file, use filename
                    rel_path = Path(key).name

                dest = local_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))


def start_batch_job(
    job_name: str,
    model_id: str,
    role_arn: str,
    input_s3: str,
    output_s3: str,
) -> str:
    bedrock = boto3.client(
        "bedrock",
        region_name=AWS_REGION,
    )
    logger.info(f"[Batch] Submitting Job: {job_name}")
    logger.info(f"        Model: {model_id}")
    logger.info(f"        Input: {input_s3}")
    logger.info(f"        Output: {output_s3}")

    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        roleArn=role_arn,
        modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": input_s3}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3}},
    )
    arn = response.get("jobArn")
    logger.info(f"[Batch] Job ARN: {arn}")
    return arn


# ---------------------------------------------------------------------
# 3. MMR Selection (Online Embedding)
# ---------------------------------------------------------------------


class RAGEembedder:
    """Simple online embedder for the MMR Selection phase only."""

    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for txt in tqdm(texts, desc="[MMR] Embedding Candidates", leave=False):
            try:
                # Titan V2
                body = json.dumps(
                    {
                        "inputText": txt,  # Truncate for safety
                        "dimensions": 1024,
                        "normalize": True,
                    }
                )
                resp = self.client.invoke_model(
                    modelId=MODEL_EMBED,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )
                body = json.loads(resp["body"].read())
                results.append(body["embedding"])
            except Exception:
                results.append([0.0] * 1024)
        return results


class MMRSelector:
    def __init__(self, embedder):
        self.embedder = embedder

    def select(
        self,
        candidates: List[dict],  # dicts with 'text' and 'score'
        k: int,
        lambda_param: float = 0.5,
    ) -> List[dict]:
        if len(candidates) <= k:
            return candidates

        logger.info(
            f"[MMR] Selecting {k} diverse docs from {len(candidates)} candidates..."
        )
        texts = [c["text"] for c in candidates]
        embeddings = np.array(self.embedder.embed(texts))

        # Normalize scores
        scores = np.array([c["score"] for c in candidates])
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores.fill(1.0)

        selected_indices = [np.argmax(scores)]
        candidate_indices = [
            i for i in range(len(candidates)) if i != selected_indices[0]
        ]

        pbar = tqdm(total=k - 1, desc="[MMR] Iterating", leave=False)
        while len(selected_indices) < k and candidate_indices:
            rem_emb = embeddings[candidate_indices]
            sel_emb = embeddings[selected_indices]
            sim_matrix = cosine_similarity(rem_emb, sel_emb)
            max_sim = np.max(sim_matrix, axis=1)

            curr_quality = scores[candidate_indices]
            mmr_scores = (lambda_param * curr_quality) - ((1 - lambda_param) * max_sim)

            best_local = np.argmax(mmr_scores)
            selected_indices.append(candidate_indices[best_local])
            candidate_indices.pop(best_local)
            pbar.update(1)
        pbar.close()

        return [candidates[i] for i in selected_indices]


# ---------------------------------------------------------------------
# 4. S1 Pipeline: Logic & Prompts
# ---------------------------------------------------------------------

S1_WHY_SYSTEM = """
You are an expert computational psycholinguist generating training data.
Your task is to provide a "Teacher Rationale" explaining WHY a specific text span fits a conspiracy marker label.

<definitions>
- Actor: Agents alleged to secretly orchestrate events (e.g., "they", "the elite", "Big Pharma").
- Action: Deliberate, covert acts attributed to the Actor (verb phrases like "scheme", "cover up", "orchestrate"). Exclude unintended outcomes.
- Effect: Grand consequence or goal of the Action (e.g., "enslavement", "total control", "depopulation").
- Victim: Entity harmed or targeted by the Action.
- Evidence: Explicit support: URLs, named sources, quotes, or numeric facts tied to a source.
</definitions>

<task>
  Given the Text, Span, and Label:
  1. Identify the specific lexical cue or semantic feature in the span (e.g., "The verb 'plot' implies hostile intent").
  2. Explain how it fits the definition.
  3. Extract a brief verbatim context (5-10 words) from the document that frames the span.
</task>

Output Format: JSON only.
{
  "why": "Concise rationale citing specific cues (max 25 words).",
  "context": "Verbatim 5-10 word snippet from text."
}
"""


def _score_s1_doc(d: dict) -> float:
    # Heuristic: Prefer docs with complete narratives (Action+Effect) and diverse markers
    labels = {m["label"] for m in d["spans"]}
    score = len(d["spans"]) * 0.1
    if "Action" in labels and "Effect" in labels:
        score += 2.0
    if "Victim" in labels:
        score += 1.0
    return score


def prepare_s1_whys_payload(docs: List[dict]) -> List[dict]:
    payload = []
    for d in docs:
        doc_id = d["doc_id"]
        raw_text = d["text"]
        for i, span in enumerate(d["spans"]):
            # Unique ID for re-assembly
            record_id = f"{doc_id}::{i}"
            user_prompt = (
                f"Text: {raw_text}\n"
                f"Span: '{span['text']}'\n"
                f"Label: {span['label']}\n\n"
                "Explain why this span fits the label and provide context."
            )

            # Claude 3 Input Format
            model_input = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": user_prompt}],
                "system": S1_WHY_SYSTEM,
            }
            payload.append({"recordId": record_id, "modelInput": model_input})
    return payload


# ---------------------------------------------------------------------
# 5. S2 Pipeline: Logic & Prompts
# ---------------------------------------------------------------------

S2_RATIONALE_SYSTEM = """
You are a forensic analyst determining the author's STANCE toward a conspiracy narrative.
You are generating a rationale for a training example.

<inputs>
  1. Document Text: The raw user comment.
  2. Marker Summary: The "Plot" (Who is doing what).
  3. Gold Label: "conspiracy" (Endorsement) or "non" (Reporting/Mocking/Debunking).
</inputs>

<task>
  Write a 1-2 sentence rationale explaining the label based on STANCE.
  
  - If Label = "conspiracy": Explain that the author *endorses* the plot. Cite cues like "truth-telling" vocabulary ("Wake up", "The reality is") or urgent warnings.
  - If Label = "non": Explain that the author is *distanced* from the plot. Cite reporting verbs ("They claim", "Users say"), mockery ("This ridiculous theory"), or lack of endorsement.
</task>

Output Format: JSON only.
{ "rationale": "The author [endorses/reports]... because [specific cue]." }
"""


def prepare_s2_rats_payload(docs: List[dict]) -> List[dict]:
    payload = []
    for d in docs:
        record_id = d["doc_id"]
        summary = d.get("marker_summary", "No markers found.")
        user_prompt = (
            f"Document: {d['text']}\n"
            f"Gold Label: {d['label']}\n"
            f"Marker Summary: {summary}\n\n"
            "Generate the rationale."
        )

        model_input = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": user_prompt}],
            "system": S2_RATIONALE_SYSTEM,
        }
        payload.append({"recordId": record_id, "modelInput": model_input})
    return payload


# ---------------------------------------------------------------------
# 6. Common Embedding Payload Builder (Titan)
# ---------------------------------------------------------------------


def prepare_embedding_payload(docs: List[dict]) -> List[dict]:
    payload = []
    for d in docs:
        payload.append(
            {
                "recordId": str(d["doc_id"]),
                "modelInput": {
                    "inputText": d["text"],  # Titan limit safety
                    "dimensions": 1024,
                    "normalize": True,
                },
            }
        )
    return payload


# ---------------------------------------------------------------------
# 7. Workflow Orchestration
# ---------------------------------------------------------------------


def run_s1_rag_pipeline(args):
    """S1 Pipeline: Select -> Batch Whys -> Batch Embed -> Index"""
    rag_dir = Path(args.rag_out_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    f_selected = rag_dir / "s1_01_selected.jsonl"
    f_whys_in = rag_dir / "s1_02_whys_input.jsonl"
    f_enriched = rag_dir / "s1_03_enriched.jsonl"
    f_embed_in = rag_dir / "s1_04_embed_input.jsonl"

    # --- STAGE 1: SELECT (MMR) ---
    if args.s1_stage == "select":
        logger.info("[S1] Loading and Selecting Docs...")
        # Load
        raw = []
        with open(args.s1_train_jsonl, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                spans = d.get("spans") or d.get("markers") or []
                if spans:
                    raw.append(
                        {
                            "doc_id": str(d.get("doc_id") or d.get("_id")),
                            "text": d["text"],
                            "spans": spans,
                            "score": _score_s1_doc({"spans": spans}),
                        }
                    )

        # Filter Pool (Heuristic) -> MMR
        pool = sorted(raw, key=lambda x: x["score"], reverse=True)[: args.max_docs * 3]
        selector = MMRSelector(RAGEembedder())
        selected = selector.select(pool, k=args.max_docs, lambda_param=0.6)

        # Save
        with open(f_selected, "w", encoding="utf-8") as f:
            for d in selected:
                f.write(json.dumps(d) + "\n")
        logger.info(
            f"[S1] Selection complete. Saved {len(selected)} docs to {f_selected}"
        )

    # --- STAGE 2: SUBMIT WHYS (Text Gen) ---
    elif args.s1_stage == "submit-whys":
        logger.info("[S1] Preparing Whys Batch Job...")
        docs = [json.loads(line) for line in open(f_selected, encoding="utf-8")]
        payload = prepare_s1_whys_payload(docs)

        # Save & Upload
        with open(f_whys_in, "w", encoding="utf-8") as f:
            for p in payload:
                f.write(json.dumps(p) + "\n")

        s3_key = "inputs/s1_whys.jsonl"
        upload_to_s3(f_whys_in, args.bucket, s3_key)

        # Submit
        arn = start_batch_job(
            job_name=get_job_name("S1_WHYS"),
            model_id=BEDROCK_MODEL_ID,
            role_arn=args.role_arn,
            input_s3=f"s3://{args.bucket}/{s3_key}",
            output_s3=f"s3://{args.bucket}/outputs/s1_whys/",
        )
        logger.info(f"[S1] Whys Job Started! ARN: {arn}")
        logger.info("Wait for completion, then run --s1-stage merge-whys")

    # --- STAGE 3: MERGE WHYS ---
    elif args.s1_stage == "merge-whys":
        logger.info("[S1] Merging Whys...")
        # Download
        local_dl = rag_dir / "s1_whys_results"
        download_from_s3(args.bucket, "outputs/s1_whys/", local_dl)

        # Parse Results
        results = {}
        for fpath in local_dl.glob("*.out"):
            for line in open(fpath, encoding="utf-8"):
                res = json.loads(line)
                rid = res["recordId"]
                # Parse Claude Output
                text_out = res["modelOutput"]["content"][0]["text"]
                try:
                    # Robust JSON extraction
                    m = re.search(r"\{.*\}", text_out, re.DOTALL)
                    if m:
                        results[rid] = json.loads(m.group(0))
                except:
                    pass

        # Merge
        docs = [json.loads(line) for line in open(f_selected, encoding="utf-8")]
        for d in docs:
            for i, span in enumerate(d["spans"]):
                rid = f"{d['doc_id']}::{i}"
                if rid in results:
                    span.update(results[rid])  # Adds 'why' and 'context'

        with open(f_enriched, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        logger.info(f"[S1] Merged rationales. Saved to {f_enriched}")

    # --- STAGE 4: SUBMIT EMBED (Titan) ---
    elif args.s1_stage == "submit-embed":
        logger.info("[S1] Preparing Embedding Batch Job...")
        docs = [json.loads(line) for line in open(f_enriched, encoding="utf-8")]
        payload = prepare_embedding_payload(docs)

        with open(f_embed_in, "w", encoding="utf-8") as f:
            for p in payload:
                f.write(json.dumps(p) + "\n")

        s3_key = "inputs/s1_embed.jsonl"
        upload_to_s3(f_embed_in, args.bucket, s3_key)

        arn = start_batch_job(
            job_name=get_job_name("S1_EMBED"),
            model_id=MODEL_EMBED,
            role_arn=args.role_arn,
            input_s3=f"s3://{args.bucket}/{s3_key}",
            output_s3=f"s3://{args.bucket}/outputs/s1_embed/",
        )
        logger.info(f"[S1] Embedding Job Started! ARN: {arn}")
        logger.info("Wait for completion, then run --s1-stage index")

    # --- STAGE 5: INDEX ---
    elif args.s1_stage == "index":
        logger.info("[S1] Indexing to Chroma...")
        local_dl = rag_dir / "s1_embed_results"
        download_from_s3(args.bucket, "outputs/s1_embed/", local_dl)

        # Load Vectors
        vectors = {}
        for fpath in local_dl.glob("*.out"):
            for line in open(fpath, encoding="utf-8"):
                res = json.loads(line)
                vectors[res["recordId"]] = res["modelOutput"]["embedding"]

        # Build DB
        docs = [json.loads(line) for line in open(f_enriched, encoding="utf-8")]
        client = chromadb.PersistentClient(path=str(rag_dir))
        col = client.get_or_create_collection(
            "s1_markers", metadata={"hnsw:space": "cosine"}
        )

        ids, embs, texts, metas = [], [], [], []
        for d in docs:
            did = d["doc_id"]
            if did in vectors:
                ids.append(did)
                embs.append(vectors[did])
                texts.append(d["text"])
                metas.append({"doc_id": did, "spans_json": json.dumps(d["spans"])})

        # Batch Add
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            col.add(
                ids=ids[i : i + batch_size],
                embeddings=embs[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
            )
        logger.info(f"[S1] Indexed {len(ids)} documents successfully.")


# ---------------------------------------------------------------------
# S2 Marker Summary Logic
# ---------------------------------------------------------------------

S2_MARKER_SUMMARY_SYSTEM = """
You are an expert narrative analyst.
You are given a list of extracted forensic markers (Actor, Action, Effect) from a text.

Task: Synthesize these markers into ONE concise sentence (max 40 words) summarizing the alleged conspiracy plot.
- Focus on: Who (Actor) is doing What (Action) to Whom (Victim) and Why (Effect).
- If markers are sparse or unrelated, say "No coherent plot detected."

Output Format: JSON only.
{ "summary": "The markers indicate..." }
"""


def prepare_s2_summary_payload(docs: List[dict]) -> List[dict]:
    payload = []
    for d in docs:
        if not d.get("markers"):
            continue

        record_id = d["doc_id"]
        # Convert markers to a readable list for the model
        markers_text = "\n".join([f"- {m['label']}: {m['text']}" for m in d["markers"]])

        user_prompt = (
            f"Raw Text Segment: {d['text']}...\n\n"
            f"Extracted Markers:\n{markers_text}\n\n"
            "Summarize the narrative plot."
        )

        model_input = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": user_prompt}],
            "system": S2_MARKER_SUMMARY_SYSTEM,
        }
        payload.append({"recordId": record_id, "modelInput": model_input})
    return payload


def run_s2_rag_pipeline(args):
    """S2 Pipeline: Select -> Batch Summary -> Batch Rationale -> Batch Embed -> Index"""
    rag_dir = Path(args.rag_out_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate files
    f_selected = rag_dir / "s2_01_selected.jsonl"
    f_sum_in = rag_dir / "s2_02_summary_input.jsonl"
    f_summarized = rag_dir / "s2_03_summarized.jsonl"
    f_rats_in = rag_dir / "s2_04_rats_input.jsonl"
    f_enriched = rag_dir / "s2_05_enriched.jsonl"
    f_embed_in = rag_dir / "s2_06_embed_input.jsonl"

    # --- STAGE 1: SELECT ---
    if args.s2_stage == "select":
        logger.info("[S2] Loading Docs & Matching Markers...")
        # Load S1 Lookup
        s1_lookup = {}
        with open(args.s1_train_jsonl, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                spans = d.get("spans") or d.get("markers")
                if spans:
                    s1_lookup[str(d.get("doc_id") or d.get("_id"))] = spans

        # Load S2
        s2_docs = []
        with open(args.s2_train_docclf, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["label"] in ["conspiracy", "non"]:
                    d["doc_id"] = str(d.get("doc_id") or d.get("_id"))
                    d["markers"] = s1_lookup.get(d["doc_id"], [])
                    s2_docs.append(d)

        # Prioritize docs with markers
        s2_docs.sort(key=lambda x: len(x["markers"]), reverse=True)
        s2_docs = s2_docs[: args.max_docs]

        with open(f_selected, "w", encoding="utf-8") as f:
            for d in s2_docs:
                f.write(json.dumps(d) + "\n")
        logger.info(f"[S2] Selected {len(s2_docs)} docs. Saved to {f_selected}")

    # --- STAGE 2: SUBMIT SUMMARIES ---
    elif args.s2_stage == "submit-summaries":
        logger.info("[S2] Preparing Summary Batch Job...")
        docs = [json.loads(line) for line in open(f_selected, encoding="utf-8")]
        payload = prepare_s2_summary_payload(docs)

        with open(f_sum_in, "w", encoding="utf-8") as f:
            for p in payload:
                f.write(json.dumps(p) + "\n")

        s3_key = "inputs/s2_summaries.jsonl"
        upload_to_s3(f_sum_in, args.bucket, s3_key)

        arn = start_batch_job(
            job_name=get_job_name("S2_SUMMARIES"),
            model_id=BEDROCK_MODEL_ID,  # Claude Haiku
            role_arn=args.role_arn,
            input_s3=f"s3://{args.bucket}/{s3_key}",
            output_s3=f"s3://{args.bucket}/outputs/s2_summaries/",
        )
        logger.info(f"[S2] Summary Job Started! ARN: {arn}")

    # --- STAGE 3: MERGE SUMMARIES ---
    elif args.s2_stage == "merge-summaries":
        logger.info("[S2] Merging Summaries...")
        local_dl = rag_dir / "s2_summary_results"
        download_from_s3(args.bucket, "outputs/s2_summaries/", local_dl)

        summaries = {}
        for fpath in local_dl.glob("*.out"):
            for line in open(fpath, encoding="utf-8"):
                try:
                    res = json.loads(line)
                    txt = res["modelOutput"]["content"][0]["text"]
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if m:
                        summaries[res["recordId"]] = json.loads(m.group(0))["summary"]
                except:
                    pass

        docs = [json.loads(line) for line in open(f_selected, encoding="utf-8")]
        for d in docs:
            # If we sent it for summary, attach result. Else default.
            if d.get("markers"):
                d["marker_summary"] = summaries.get(d["doc_id"], "Summary failed.")
            else:
                d["marker_summary"] = "No markers found."

        with open(f_summarized, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        logger.info(f"[S2] Merged Summaries. Saved to {f_summarized}")

    # --- STAGE 4: SUBMIT RATIONALES ---
    elif args.s2_stage == "submit-rats":
        logger.info("[S2] Preparing Rationale Batch Job...")
        docs = [json.loads(line) for line in open(f_summarized, encoding="utf-8")]
        # Now we use the summaries generated in the previous step
        payload = prepare_s2_rats_payload(docs)

        with open(f_rats_in, "w", encoding="utf-8") as f:
            for p in payload:
                f.write(json.dumps(p) + "\n")

        s3_key = "inputs/s2_rats.jsonl"
        upload_to_s3(f_rats_in, args.bucket, s3_key)

        arn = start_batch_job(
            job_name=get_job_name("S2_RATS"),
            model_id=BEDROCK_MODEL_ID,
            role_arn=args.role_arn,
            input_s3=f"s3://{args.bucket}/{s3_key}",
            output_s3=f"s3://{args.bucket}/outputs/s2_rats/",
        )
        logger.info(f"[S2] Rationale Job Started! ARN: {arn}")

    # --- STAGE 5: MERGE RATIONALES ---
    elif args.s2_stage == "merge-rats":
        logger.info("[S2] Merging Rationales...")
        local_dl = rag_dir / "s2_rats_results"
        download_from_s3(args.bucket, "outputs/s2_rats/", local_dl)

        rats = {}
        for fpath in local_dl.glob("*.out"):
            for line in open(fpath, encoding="utf-8"):
                try:
                    res = json.loads(line)
                    txt = res["modelOutput"]["content"][0]["text"]
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if m:
                        rats[res["recordId"]] = json.loads(m.group(0))["rationale"]
                except:
                    pass

        docs = [json.loads(line) for line in open(f_summarized, encoding="utf-8")]
        for d in docs:
            d["rationale"] = rats.get(d["doc_id"], "Analysis failed.")

        with open(f_enriched, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        logger.info(f"[S2] Merged Rationales. Saved to {f_enriched}")

    # --- STAGE 6: SUBMIT EMBED ---
    elif args.s2_stage == "submit-embed":
        logger.info("[S2] Submitting Embedding Batch Job...")
        docs = [json.loads(line) for line in open(f_enriched, encoding="utf-8")]
        payload = prepare_embedding_payload(docs)

        with open(f_embed_in, "w", encoding="utf-8") as f:
            for p in payload:
                f.write(json.dumps(p) + "\n")

        s3_key = "inputs/s2_embed.jsonl"
        upload_to_s3(f_embed_in, args.bucket, s3_key)

        arn = start_batch_job(
            job_name=get_job_name("S2_EMBED"),
            model_id=MODEL_EMBED,
            role_arn=args.role_arn,
            input_s3=f"s3://{args.bucket}/{s3_key}",
            output_s3=f"s3://{args.bucket}/outputs/s2_embed/",
        )
        logger.info(f"[S2] Embed Job Started! ARN: {arn}")

    # --- STAGE 7: INDEX ---
    elif args.s2_stage == "index":
        logger.info("[S2] Indexing to Chroma...")
        local_dl = rag_dir / "s2_embed_results"
        download_from_s3(args.bucket, "outputs/s2_embed/", local_dl)

        vectors = {}
        for fpath in local_dl.glob("*.out"):
            for line in open(fpath, encoding="utf-8"):
                res = json.loads(line)
                vectors[res["recordId"]] = res["modelOutput"]["embedding"]

        docs = [json.loads(line) for line in open(f_enriched, encoding="utf-8")]
        client = chromadb.PersistentClient(path=str(rag_dir))
        col = client.get_or_create_collection(
            "s2_examples", metadata={"hnsw:space": "cosine"}
        )

        ids, embs, texts, metas = [], [], [], []
        for d in docs:
            did = d["doc_id"]
            if did in vectors:
                ids.append(did)
                embs.append(vectors[did])
                texts.append(d["text"])
                metas.append(
                    {
                        "label": d["label"],
                        "rationale": d["rationale"],
                        "markers_json": json.dumps(d["markers"]),
                        "marker_summary": d["marker_summary"],
                    }
                )

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            col.add(
                ids=ids[i : i + batch_size],
                embeddings=embs[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metas[i : i + batch_size],
            )
        logger.info(f"[S2] Indexed {len(ids)} documents.")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SemEval Batch RAG Builder")

    # Common Args
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role-arn", default=DEFAULT_ROLE_ARN)
    parser.add_argument("--rag-out-dir", default="data/rag")
    parser.add_argument("--max-docs", type=int, default=512)

    # S1 Args
    parser.add_argument("--s1-train-jsonl", help="Path to S1 train data")
    parser.add_argument(
        "--s1-stage",
        choices=["select", "submit-whys", "merge-whys", "submit-embed", "index"],
    )

    # S2 Args
    parser.add_argument("--s2-train-docclf", help="Path to S2 train data")
    parser.add_argument(
        "--s2-stage",
        choices=["select", "submit-rats", "merge-rats", "submit-embed", "index"],
    )

    args = parser.parse_args()

    if args.s1_stage:
        if not args.s1_train_jsonl and args.s1_stage == "select":
            logger.error("Error: --s1-train-jsonl required for selection.")
            return
        run_s1_rag_pipeline(args)

    if args.s2_stage:
        if not args.s2_train_docclf and args.s2_stage == "select":
            logger.error("Error: --s2-train-docclf required for selection.")
            if not args.s1_train_jsonl:
                logger.error(
                    "Error: --s1-train-jsonl required for S2 summary generation."
                )
            return
        run_s2_rag_pipeline(args)


if __name__ == "__main__":
    main()
