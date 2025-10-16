# prompt_sweep_joint.py Explained

## Pipeline Overview

`starter/prompt_sweep_joint.py` orchestrates a two-stage joint sweep for SemEval Task 10:

1. **S1 span extraction** – locate psycholinguistic markers (Actor, Action, Effect, Victim, Evidence).
2. **S2 document classification** – decide whether a document promotes a conspiracy narrative.

Both stages use LLM prompts that can be augmented with EDA-derived policies and few-shot examples.

## Detailed Flow

1. **Input loading**
   - `--test-file-s1` supplies labeled documents with markers for S1 evaluation (e.g., `dev.jsonl`).
   - `--test-file-s2` provides labeled documents for binary classification (e.g., `dev_docclf.jsonl`).
   - `--limit-docs N` optionally restricts processing to the first *N* documents to speed experimentation.

2. **Prompt construction**
   - If `--eda-root` is set, the script loads helper text and few-shot examples via `build_s1_policy`, `build_s2_policy`, and `load_fewshots`.
   - Techniques (comma-separated) control features such as few-shots (`fs_…`) or self-consistency (`sc5`, `sc10`).

3. **S1 inference**
   - For each document, it calls the Bedrock model with the constructed S1 prompt.
   - Outputs:
     - `runs/<root>/<tech>/s1/submission.jsonl`: full set of extracted spans (used for evaluation).
     - `runs/<root>/<tech>/s1/submission_pruned.jsonl`: spans pruned per label (for S2 conditioning).
   - Debug counters `raw/valid/pruned` show how many spans were produced, kept as valid, and retained after pruning.
   - Evaluation: invokes `starter/eval_token.py` (token IoU) and prints aggregate + macro F1. Also writes per-label CSV and a bar plot if matplotlib is available.

4. **S2 inference**
   - For each document, the S2 prompt includes the original text plus the pruned S1 markers.
   - The model returns a label and dual probabilities (`p_conspiracy`, `p_non`).
   - Predictions go to `runs/<root>/<tech>/s2/submission.jsonl`; probabilities to `runs/<root>/<tech>/s2/probs.jsonl`.
   - Probability diagnostics include mean `p_conspiracy` and fraction of samples with `p_conspiracy ≥ 0.5`.
   - Inline evaluation computes accuracy, F1 (binary, macro, weighted), and confusion matrix counts.

5. **Summary output**
   - Collects all metrics per technique into `runs/<root>/joint_prompt_sweep_summary.csv`.
   - Prints a compact table of S1/S2 headline scores for convenience.

## Interpreting Your Run

Example log excerpt:

```text
=== JOINT S1→S2 :: fs_boundary_policy ===
S1 done -> runs\joint_llm\tinydev\fs_boundary_policy\s1\submission.jsonl
S1 pruned-for-S2 -> runs\joint_llm\tinydev\fs_boundary_policy\s1\submission_pruned.jsonl
S1 debug: spans raw/valid/pruned = 42/42/39
S1 metrics: F1_aggregate=0.011 F1_macro=0.012
S2 prob stats: mean_p=0.100 frac_p>=0.5=0.000
S2 done -> runs\joint_llm\tinydev\fs_boundary_policy\s2\submission.jsonl
S2 metrics: acc=0.580 f1_bin=0.000 f1_macro=0.367 f1_weighted=0.426 tn=58 fp=0 fn=42 tp=0
```

### S1 interpretation

- **Low F1 (≈0.01)** means the extracted spans rarely overlap with the gold markers under the IoU threshold. Causes include prompt misalignment, insufficient few-shot guidance, boundary errors, or small sample size.
- The `raw/valid/pruned` counts show pruning is mainly for S2 input; S1 evaluation uses the full `submission.jsonl` output.

### S2 interpretation

- **Probabilities**: average `p_conspiracy` of 0.10 and zero cases above 0.5 indicate the model defaults to "non". Dual probabilities are captured in `probs.jsonl` for diagnostics.
- **Metrics**: accuracy is moderate, but binary F1 is zero because the model never predicted "conspiracy" (tp=0). Macro/weighted F1 remain nonzero owing to class imbalance.

### Next Steps

- Improve prompts and few-shots (especially for S1) to raise span recall and accuracy.
- Consider tuning the decision threshold on `p_conspiracy` using dev labels to avoid predicting only "non".
- Inspect probability distribution (`probs.jsonl`) and increase the document limit once confident in the setup.
