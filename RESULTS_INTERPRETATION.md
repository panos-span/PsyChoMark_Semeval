# PsyCoMark – Interpretation of Preliminary Results

## 1. Purpose of This Document

This report synthesizes the exploratory analyses, derived artifacts, and few-shot selection outcomes produced so far. It translates raw findings into modeling implications and prioritized next steps for both subtasks:

- **S1**: Psycholinguistic marker span extraction (Actor, Action, Effect, Evidence, Victim)
- **S2**: Conspiracy vs Non-Conspiracy document classification

## 2. Data Pipeline & Corpus Snapshot

Latest processed run: `psycomark_official_split_20250928_232947`

| Split | Raw Docs | Final Docs | Notes |
|-------|----------|------------|-------|
| Train | 3,682 | 3,360 | 20 removed (dev leakage); internal duplicate count not reported numerically |
| Dev   | 100 | 100 | Preserved verbatim |

Key pipeline properties:

- **Leakage control**: Dev overlap removed from train (20 docs). Critical for trustworthy validation.
- **Versioning**: Timestamped directory + pointer file ensures reproducibility.
- **Dedup strategy**: Keeps richer marker density when collapsing clusters (details in `data_pipeline.py`).

## 3. Marker Ambiguity & Overlap (IoU Analysis)

Artifact: `mean_iou_matrix.png`

Observations:

- Highest average overlap: **Action ↔ Effect** (expected—plans versus outcomes blend semantically in narrative text).
- Moderate incidental overlaps with Actor spans (nested attribution clauses).
- Near-perfect self-diagonal (sanity) and low cross-noise for distinct roles like Victim vs Evidence.

Implications for S1 modeling:

- **Boundary Confusion**: A principal error source will be distinguishing Action (process/intent) from Effect (realized or projected outcome). Consider auxiliary contrastive loss to push apart embeddings of co-occurring Action vs Effect spans.
- **Label Co-Occurrence**: Multi-head or multi-label token classification should allow overlapping spans; BIO with naive decoding may under-represent overlaps. A span-proposal approach (start/end pairs) could handle overlap more cleanly.

## 4. Linguistic Certainty / Absolutist Language

Artifact: `absolutist_language_analysis.png`

Findings:

- Conspiracy-labeled documents show a **higher median** and wider spread of absolutist lexical items (e.g., *always, proof, clearly, secret*).
- Distribution difference suggests stylistic certainty framing is a predictive—but not causative—signal.

Modeling implications:

- Add a **normalized absolutist count feature** to S2 classifier (or prompt note) while guarding against over-reliance (feature dropout / adversarial masking evaluation).
- Potential to calibrate decisions: cases with high classifier confidence but low certainty language → inspect for alternative cues (structural or marker-driven evidence).

## 5. Hard Example Mining Outcomes

Artifact: `hard_examples.json`

Criteria applied:

1. **High Action/Effect IoU**: 74 docs (ambiguity cluster)
2. **High Subreddit Label Entropy (>1.5)**: 1,124 docs (domain ambiguity)
3. **Baseline Confident Misclassifications**: (Counts not printed—should log explicitly in future runs)

Combined unique hard examples exported: **1,167** (filtered to 869 after label restrictions for few-shot selection in Notebook 2).

Utility:

- Supplies a **curriculum pool** for focused fine-tuning or for few-shot prompt augmentation.
- Enables building *difficulty-aware sampling* (upweight ambiguous cases early, anneal later).

## 6. Positional Structure of Spans

Artifact: `span_position_analysis.png`

Narrative tendencies (qualitatively expected—visual confirmation pending detailed quantification):

- **Actor** spans cluster earlier (introducing agents).
- **Evidence** tends to drift later (supporting justification after claim framing).
- **Effect** can appear mid-to-late as consequences are elaborated.

Modeling implications:

- Incorporate **relative position bins** (e.g., quintiles) as an additional embedding or feature for span boundary decisions.
- Positional priors can regularize improbable placements (e.g., Evidence at very first tokens—flag for uncertainty).

## 7. Few-Shot Exemplar Set Assessment

Artifact: `best_fewshot_examples.json`

Composition:

- **S1**: 10 span exemplars (2 per label; compact markers, capped to 1–2 spans per label) → concise, balanced, limited stylistically.
- **S2**: 16 classification exemplars (8 conspiracy / 8 non) with rationales.

Strengths:

- Balanced label coverage prevents early prompt bias.
- Inclusion of ambiguous Action/Effect cases helps boundary teaching.

Limitations:

- Small absolute count—risk of overfitting LLM latent pattern to a narrow stylistic band.
- No negative exemplar with *zero* markers (could help calibrate span abstention behavior).
- No intentionally adversarial or stylistically atypical samples (e.g., low-certainty conspiracy, high-certainty non-conspiracy).

Recommended refinements:

- Add 1–2 “empty marker” non-conspiracy docs to reinforce span precision.
- Rotate or ensemble exemplar sets (A/B) to measure variance in LLM outputs.
- Introduce explicit *contrastive pair* (same base claim wording with vs without conspiratorial inference).

## 8. Emerging Risk Factors

| Risk | Description | Mitigation |
|------|-------------|------------|
| Action/Effect confusion | Semantic blending reduces F1 for both | Auxiliary contrastive loss; label-specific boundary hints in prompt |
| Over-reliance on certainty words | Lexical stylistic shortcut | Feature masking evaluation; SHAP / attribution audits |
| r/conspiracy oversampling bias | Domain style leakage to classifier | Domain-adaptive weighting; subreddit adversarial head |
| Few-shot narrow style | LLM prompt brittleness | Exemplar diversification & rotation |
| Over-trimming markers | Capping spans per label may hide multiplicity patterns | Provide at least one multi-span Actor or Evidence example |
| Lack of baseline grounding | No supervised baseline yet—weak comparison | Implement lightweight DeBERTa / RoBERTa baselines ASAP |

## 9. Immediate Modeling Priorities

1. **Evaluation Harness**: Implement span IoU F1 (threshold sweep) + doc macro-F1 script.
2. **Supervised Baselines**:
   - S2: DeBERTa-v3-small (class weights from `class_weights.json`).
   - S1: BIO tagger (allow overlapping via multi-pass or span enumeration).
3. **Feature Engineering**:
   - Add absolutist count, positional bins, subreddit entropy (global) to document classifier.
4. **Few-Shot Expansion**:
   - Add negative zero-marker exemplar + one high-density multi-marker exemplar.
5. **Difficulty-Aware Sampling**:
   - Upweight high IoU & high entropy docs in early epochs.

## 10. Medium-Term Enhancements

| Area | Proposal | Benefit |
|------|----------|---------|
| Joint Multi-Task Model | Shared encoder + span & doc heads | Leverage marker cues for classification |
| Span Proposal + Refinement | Start/end candidate generation + scorer | Better overlap handling than flat BIO |
| Contrastive Role Disambiguation | Pull apart Action vs Effect embeddings | Reduce systematic label confusion |
| Gating Upgrade | Distill LLM predictions into local classifier | Lower latency & cost |
| Exemplar Hashing | Cache invalidation by set fingerprint | Reproducibility & audit |
| Robustness Tests | Style ablation (remove certainty words) | Measure reliance on superficial cues |

## 11. Quantitative KPIs to Track Going Forward

| KPI | Target (Initial) | Rationale |
|-----|------------------|-----------|
| S2 Macro-F1 (baseline) | > LLM few-shot by +5 points | Demonstrates added value of supervision |
| S1 Macro-F1 @ IoU≥0.5 | >= 0.55 initial | Establish viable baseline |
| Action↔Effect confusion rate | < 25% of span errors | Gauge role disambiguation success |
| Zero-span precision (non-conspiracy) | High (>0.9) | Avoid hallucinated spans |
| Cache hit rate (LLM runs) | > 60% during iterations | Cost control |

## 12. Actionable Next Steps (Ordered)

1. Create `scripts/eval_spans.py` & `scripts/eval_doc.py`.
2. Implement `data/dataset.py` with joint yield (tokens, BIO tags, doc label, features).
3. Train S2 baseline; log macro-F1 + calibration curve.
4. Train S1 baseline; produce label-wise and overall IoU F1 table.
5. Expand few-shot set (add zero-marker and multi-span exemplars); re-run LLM to compare effect.
6. Add heuristic feature ablation experiment (with and without certainty counts).
7. Prototype Action vs Effect contrastive auxiliary loss.

## 13. Open Questions / Decisions Needed

| Question | Options | Recommendation |
|----------|---------|----------------|
| Overlapping span encoding | Multi-pass BIO vs span proposals | Start with span proposals (start,end,label) using boundary classifier |
| Handling 'cant_tell' | Ignore vs third class | Ignore initially for simpler binary; revisit if many borderline cases |
| Evaluation IoU threshold | Single (0.5) vs sweep | Provide sweep (0.3/0.5/0.7) for diagnostic richness |
| Few-shot rationale length | Current 1–2 sentences | Keep but enforce via regex check to prevent drift |

## 14. Summary

The exploratory phase surfaces clear, actionable structure: (a) Action–Effect ambiguity is the dominant span challenge; (b) stylistic certainty cues are predictive but must be regularized; (c) positional signals and subreddit entropy can be leveraged for robustness; and (d) current few-shot exemplars are balanced but stylistically narrow. The immediate focus should shift to building a reproducible supervised baseline stack plus evaluation harness, then iteratively enriching prompts and multi-task models with targeted disambiguation strategies.

---
**Prepared for**: Modeling & Engineering team

**Next author task** (if approved): Scaffold evaluation + baseline training scripts.
