## Reproducible Steps & Script Outputs

This section provides step-by-step instructions, PowerShell commands, and a summary of what each script produces.

### Overview

The baseline pipeline includes:

- **Data pipeline:** Prevents data leakage, creates cross-validation folds, computes priors, overlap statistics, and diagnostics.
- **Exploratory Data Analysis (EDA):** Quantifies span interactions and lexical signals to inform rule selection.
- **Few-shot selection:** Uses a frozen `cant_tell` policy for example selection.
- **S2 (Document-level):** DistilBERT baseline for document classification, outputs probability dumps, evaluation metrics, and optional calibration.
- **S1 (Span-level):** Trains per-label taggers, scores spans, merges results, applies rule-based post-processing, and evaluates span predictions.

### PowerShell Commands

All commands assume you are using PowerShell on Windows and running Python scripts via `uv`.

### Directory Structure

```markdown
.
├── scripts/
│   ├── data_pipeline.py              # Data processing and diagnostics
│   ├── analysis_and_insights.py      # EDA and insights
│   ├── make_docclf_views.py          # Document classification views
│   └── calibrate_temperature.py      # Calibration
├── src/
│   └── psycomark/
│       ├── eda/
│       │   ├── analysis-and-insights.py
│       │   └── select-fewshot-examples.py
│       ├── ensemble/
│       │   └── span_merger.py
│       └── postproc/
│           └── postprocess_spans.py
├── starter/
│   ├── train_binary.py               # S2 training
│   ├── infer_binary.py               # S2 inference
│   ├── eval_binary.py                # S2 evaluation
│   ├── train_one_span.py             # S1 training (per label)
│   ├── infer_one_span.py             # S1 inference (scored spans)
│   └── eval_token.py                 # S1 evaluation (token/span)
```

### Data Pipeline

**Purpose:**  
- Loads raw data, deduplicates, and removes train/dev leakage.
- Writes official span JSONL, document classification views, and cross-validation folds.
- Exports priors (length & start position), overlap statistics (IoU + 95% CI), and diagnostics.
- Updates `data/derived/psycomark_latest.txt`.

**Example Command:**
```powershell
uv python scripts/data_pipeline.py
```

**Outputs:**
- Processed data files for training and evaluation.
- Diagnostic reports and statistics.
- Updated derived data files.

**Example Full Command with Parameters:**
```powershell
uv run python scripts/data_pipeline.py `
    --data-dir data/raw `
    --output-root data/derived `
    --seed 42 `
    --lsh-bands 8 --lsh-ham 4
```

**Key outputs (in `data/derived/<timestamp>/`):**
- `train.jsonl`, `dev.jsonl`
- `train_docclf.jsonl`, `dev_docclf.jsonl` (binary view for S2)
- `folds.jsonl`, `folds_summary.json`
- `length_position_priors.json`
- `overlap_pair_stats.json`, `overlap_pair_stats_ci.json`
- `first_occurrence_cdf.csv`
- `manifest.json`

### EDA & Insights

**Purpose:**  
- Bootstraps CIs for mean IoU and IoU@{0, .1, .5}
- Analyzes boundary context tokens around span edges
- Computes first-occurrence CDFs, coverage, and lexical signal summaries
- Exports hard examples to seed few-shots

**Example Command:**
```powershell
uv run python src\psycomark\eda\analysis-and-insights.py
```

**Key outputs:**
- `overlap_pair_stats.json`, `overlap_pair_stats_ci.json`
- `boundary_context.json`
- `first_occurrence_cdf.csv`
- `hard_examples.json`
- Plots: `mean_iou_matrix.png`, `span_position_analysis.png`, `absolutist_*`

**Result meaning:**  
These statistics drive few-shot selection and post-processing rules (e.g., Action↔Effect and Actor↔Victim tie-breaks informed by priors and IoU rates).

3) Few-shot selection (frozen policy)

What it does

S2: balanced conspiracy/non selection; optionally inject a small number of cant_tell as negative examples with a rationale (never train on cant_tell as a positive class)

S1: prior-aware per-label selection, target ambiguous pairs, add outlier quota, short snippets

Run
```powershell
uv run python src\psycomark\eda\select-fewshot-examples.py `
  --shots-s2-per-class 8 `
  --shots-s1-per-label 2 `
  --shots-s1-outliers 1 `
  --cant-tell-negs 2 `
  --seed 42
```

Outputs

best_fewshot_examples.json — sections { "s1": [...], "s2": [...] }

fewshot_policy.json — documents the cant_tell policy and selection settings

Result meaning: we codified the cant_tell policy and curated mixed easy/hard exemplars that reflect the priors and the most ambiguous label pairs.

4) S2 — Document classification (DistilBERT baseline)
4.1 Train

````powershell
uv run python starter\train_binary.py
````

- Saves checkpoints under distilbert-conspiracy-classification\checkpoint-*
  
4.2 Inference (+ probability dump for calibration/ensembling)

```powershell
New-Item -ItemType Directory -Force -Path runs\s2_baseline | Out-Null

uv run python starter\infer_binary.py `
  --test-file "$latest\dev.jsonl" `
  --submission-file "runs\s2_baseline\submission.jsonl"
```