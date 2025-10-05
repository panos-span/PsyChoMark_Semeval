# PsyChoMark – SemEval 2026 Task 10 (Hypothetical)

Open, reproducible workflow for PsyCoMark span-level marker extraction (Actor / Action / Effect / Victim / Evidence) plus document-level conspiracy vs non classification.

## Table of Contents

- [PsyChoMark – SemEval 2026 Task 10 (Hypothetical)](#psychomark--semeval-2026-task-10-hypothetical)
  - [Table of Contents](#table-of-contents)
  - [1. Task Overview](#1-task-overview)
  - [2. Repository Structure](#2-repository-structure)
  - [3. End-to-End Workflow](#3-end-to-end-workflow)
  - [4. Data Processing Pipeline (`data_pipeline.py`)](#4-data-processing-pipeline-data_pipelinepy)
  - [5. Exploratory \& Insight Notebooks](#5-exploratory--insight-notebooks)
    - [`01-analysis-and-insights.ipynb`](#01-analysis-and-insightsipynb)
  - [6. Few-Shot Example Selection Notebook](#6-few-shot-example-selection-notebook)
    - [`02-select-fewshot-examples.ipynb`](#02-select-fewshot-examplesipynb)
  - [7. Bedrock Inference Runner (`run_bedrock_experiments.py`)](#7-bedrock-inference-runner-run_bedrock_experimentspy)
  - [8. Prompt \& Few-Shot Strategy](#8-prompt--few-shot-strategy)
  - [9. Caching, Reproducibility \& Determinism](#9-caching-reproducibility--determinism)
  - [10. Evaluation \& Outputs](#10-evaluation--outputs)
  - [11. Environment \& Installation](#11-environment--installation)
    - [Python Dependencies](#python-dependencies)
    - [AWS Bedrock Credentials](#aws-bedrock-credentials)
  - [12. Extending the System](#12-extending-the-system)
  - [13. Troubleshooting](#13-troubleshooting)
  - [14. Roadmap / Next Steps](#14-roadmap--next-steps)

## 1. Task Overview

We address a two-part (joint) problem:

1. **S1 – Span Extraction**: Identify spans belonging to one of five semantic role-like marker labels: `Actor`, `Action`, `Effect`, `Victim`, `Evidence`.
2. **S2 – Document Classification**: Label each document as `conspiracy` or `non` (optionally `cant_tell` in raw data, but excluded from few-shot exemplars and gated inference).

Our approach blends:

- Deterministic, auditable data preprocessing (leakage removal + duplicate control).
- Insight-driven exploratory analysis to surface linguistic and positional priors (e.g., Action/Effect overlap, absolutist language usage, marker positional distributions).
- Heuristic scoring for high-value few-shot exemplars, balancing coverage, ambiguity, and diversity.
- A *joint* AWS Bedrock prompt (Anthropic Claude 3 family) with optional gating: lightweight classification-only prompt when a local HF probability suggests non-conspiracy with high confidence.
- Robust parsing, response caching, and fallback strategies for production-style stability.

## 2. Repository Structure

```text
├── data_pipeline.py               # Cleans & versions official splits, creates priors & manifest
├── run_bedrock_experiments.py      # Joint S1+S2 Bedrock inference with gating, caching & concurrency
├── 01-analysis-and-insights.ipynb  # EDA: IoU heatmaps, absolutist language, hard example mining, span position
├── 02-select-fewshot-examples.ipynb# Scores & exports best few-shot exemplars (S1 & S2)
├── data/derived/                   # Versioned output runs (pointer: psycomark_latest.txt)
│   └── psycomark_official_split_YYYYMMDD_HHMMSS/
│       ├── train.jsonl / dev.jsonl
│       ├── class_weights.json / length_priors.json
│       ├── hard_examples.json
│       ├── best_fewshot_examples.json
│       ├── bedrock_preds_s1_*.jsonl / bedrock_preds_s2_*.jsonl
│       ├── mean_iou_matrix.png / absolutist_language_analysis.png / span_position_analysis.png
└── requirements.txt
```

Pointer file: `data/derived/psycomark_latest.txt` always contains the absolute path of the *latest* pipeline output directory for downstream automation.

## 3. End-to-End Workflow

High-level pipeline:

1. **Run data pipeline**: cleans, deduplicates, removes cross-split leakage, exports manifest & priors.
2. **Notebook 01**: derive analytical insights (overlap structure, ambiguity, linguistic signals, span positions) + export `hard_examples.json`.
3. **Notebook 02**: rank & select few-shot exemplars → produce `best_fewshot_examples.json` (merged S1+S2 schema).
4. **Run Bedrock experiments** with joint + gated classification-only prompts.
5. **Evaluate** predictions against official metrics (external script not yet included).
6. **Iterate**: adjust scoring heuristics, prompt structure, gating threshold, or priors.

## 4. Data Processing Pipeline (`data_pipeline.py`)

Core responsibilities:

- Loads `train_rehydrated.jsonl` and `dev_rehydrated.jsonl` (rehydrated raw sources).
- Normalizes identifiers, text, markers, and doc labels with resilient key search.
- Computes cross-split duplicate clusters using: exact hash + SimHash + LSH bucket passes.
- Removes *only* leaking training docs (dev preserved).
- Deduplicates training internally by retaining the most information-rich representative (prefers many markers + length).
- Generates class weight priors and length priors from *clean* train set only.
- Writes versioned output directory + `psycomark_latest.txt` pointer update.

Artifacts per run:

- `train.jsonl`, `dev.jsonl`, `manifest.json` (provenance, counts, removals)
- `class_weights.json`, `length_priors.json`
- (Downstream) analysis images & inference outputs.

Key design choices:

- **Immutable outputs**: Each run creates a timestamped folder for auditability.
- **Separation of concerns**: No model inference here—only cleaning and statistical priors.
- **Transparent leakage removal**: Leaky components logged; dev untouched.

## 5. Exploratory & Insight Notebooks

### `01-analysis-and-insights.ipynb`

Generates:

- Mean IoU matrix for marker-type overlaps (reveals ambiguity pairs e.g., Action↔Effect).
- Absolutist language counts + distribution by label (supports engineered features / prompt rationale).
- Hard example mining via three criteria: high Action/Effect IoU, high subreddit entropy, confident baseline misclassification.
- Span position KDEs (narrative arc intuition: Actor early, Evidence later, etc.).
- Exports: `hard_examples.json`, plots (`mean_iou_matrix.png`, `absolutist_language_analysis.png`, `span_position_analysis.png`).

Design emphasis: each analytic yields *actionable prompt / feature insight*.

## 6. Few-Shot Example Selection Notebook

### `02-select-fewshot-examples.ipynb`

- Loads `hard_examples.json` and joins with full dataset metadata.
- Scores candidates (ambiguity, label coverage, marker richness, size window, special bonuses for high Action/Effect overlap).
- Balances subreddit diversity & label coverage.
- Produces two collections:
  - `s1`: span exemplars (compact markers; at most 1–2 spans per label to control context window).
  - `s2`: classification exemplars (balanced conspiracy / non, rationale templated).
- Saves unified `best_fewshot_examples.json` consumed by the Bedrock runner (auto-coerced).

## 7. Bedrock Inference Runner (`run_bedrock_experiments.py`)

Features:

- Loads environment credentials automatically (`python-dotenv` fallback).
- Supports model IDs or (future) inference profiles.
- Builds *two* prompt templates: joint extraction+classification and classify-only (for gating).
- Optional **HF gating**: pass `--hf-probs` JSONL with per-doc `prob_conspiracy`; docs below a threshold skip span extraction.
- Concurrency via `ThreadPoolExecutor`.
- Response caching: hashed prompt → on-disk text file.
- Empty-response resilience: ignores & prunes blank cache artifacts.
- Robust JSON repair (`parse_safe`) with fallback structure on parse failure.
- Outputs separated automatically into S1 & S2 JSONL prediction files.

CLI (abridged):

```bash
uv run run_bedrock_experiments.py \
  --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
  --region us-east-1 \
  --concurrency 8 \
  --max-docs 100 \
  --cache-dir ./.bedrock_cache \
  --hf-probs path/to/hf_doc_probs.jsonl --gate-threshold 0.2
```

Key args: `--fewshots-path` (override), `--stop` (custom stop sequences), `--max-tokens-joint`, `--max-tokens-class`.

Returned JSON schema:

```json
{"spans": [{"label": "Actor", "start": 10, "end": 35}, ...],
 "doc": {"label": "conspiracy", "rationale": "..."}}
```

Written files:

- `bedrock_preds_s1_<model>.jsonl` → per line: `{doc_id, prediction=[...spans...], raw, error}`
- `bedrock_preds_s2_<model>.jsonl` → per line: `{doc_id, prediction={doc object}, raw, error}`

## 8. Prompt & Few-Shot Strategy

Principles:

- **Narrative arc scratchpad**: Encourage model to scan beginning→middle→end for role salience before emitting JSON.
- **Tight JSON contract**: Single object; enforced keys; empty spans as `[]`.
- **Exemplar curation**: Balanced marker coverage + ambiguous overlapping spans to teach boundaries.
- **Rationale constraint**: ≤2 sentences; avoids verbosity & token bloat.
- **Gating**: Saves cost/time by skipping span extraction for near-certain non-conspiracy cases.

## 9. Caching, Reproducibility & Determinism

- Disk cache keyed by SHA256 over full prompt string (includes text & few-shots).
- Empty responses not cached; stale empties purged on read.
- Versioned data directories prevent silent drift.
- Randomness controlled via pipeline `--seed` (affects duplicate resolution ordering & timestamp folder naming).

## 10. Evaluation & Outputs

Generated artifacts ready for downstream official scoring (external script not bundled yet). Suggested metrics:

- S1: span-level micro/macro F1, boundary IoU tolerance analysis.
- S2: accuracy, macro-F1 (conspiracy vs non).
- Joint error analysis: confusion pairs (Action vs Effect), missed Evidence late in docs.

## 11. Environment & Installation

### Python Dependencies

Install with `uv` (preferred) or pip:

```bash
uv venv
uv pip install -r requirements.txt
# or
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS) / .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### AWS Bedrock Credentials

`.env` example:

```ini
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...        # if temporary creds
AWS_DEFAULT_REGION=us-east-1
MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

Ensure the IAM principal has `bedrock:InvokeModel` for the target region; remove explicit denies in SCPs if applicable.

## 12. Extending the System

| Area | Idea | Effort |
|------|------|--------|
| Prompt Engineering | Add contrastive negative examples | Low |
| Span Post-Processing | Heuristic span merging / overlap suppression | Medium |
| HF Gating | Calibrate threshold via ROC on dev | Low |
| Model Ensemble | Combine Claude + small open model rational voting | Medium |
| Metrics | Add partial credit IoU-based scoring script | Low |
| Active Learning | Re-query uncertain docs to expand few-shot pool | Medium |
| Data Augmentation | Synthetic paraphrases preserving markers | High |

## 13. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| `AccessDeniedException` | IAM / SCP block | Attach policy with `bedrock:InvokeModel`, remove explicit deny |
| Empty cached outputs | Prior run under denied access | Cache now prunes empties; rerun after creds fixed |
| `No JSON object found.` | Model emitted commentary or nothing | Fallback kicks in; inspect `raw` and adjust prompt |
| Missing legend in span KDE | No spans after filtering | Check markers; verify upstream parsing |
| Few-shots skewed to one subreddit | Diversity cap too high / pool small | Increase pool or relax `MAX_PER_SUBREDDIT` |

## 14. Roadmap / Next Steps

- Add official evaluation harness + CI check.
- Implement partial-span matching metric (IoU ≥ 0.5) for robustness.
- Introduce lightweight local span tagger to pre-filter obvious negatives (further gating).
- Auto-refresh few-shot exemplars when drift detected (e.g., new high-IoU pairs).
- Provide model card & reproducibility manifest (hashes of prompt templates + exemplar set).

---

**Maintainers**: Provide issues / PRs for improvements. Re-run `data_pipeline.py` before committing new derived artifacts.
