# PsyCoMark — SemEval-2025 Task 10

**Psycholinguistic Conspiracy Marker Extraction & Endorsement Detection**

## System Architecture

PsyCoMark is a two-stage pipeline built with **Pydantic-AI** (structured LLM outputs) and **LangGraph** (stateful workflow orchestration):

### Stage 1: Conspiracy Marker Extraction (S1)

Extracts five psycholinguistic marker types — **Actor**, **Action**, **Effect**, **Victim**, **Evidence** — using a Dynamic Discriminative Chain-of-Thought (DD-CoT) pipeline:

```
Generator → Critic → Refiner → Verifier
```

- **Generator**: DD-CoT extraction with contrastive reasoning and context anchoring
- **Critic**: Soft-gated audit (prevents false-negative span wipes)
- **Refiner**: Applies critique with narrative/complexity context injection
- **Verifier**: 5-strategy span localisation + aggressive cross-label deduplication

### Stage 2: Endorsement Detection (S2)

Anti-Echo Chamber adjudication pipeline classifying texts as `conspiracy` or `non`:

```
Forensic Profiler → Parallel Council → Calibrated Judge
```

- **Forensic Profiler**: Static linguistic metrics (attribution density, JAQ detection, epistemic intensity)
- **Parallel Council**: Four adversarial personas vote **independently** (Prosecutor, Defense, Literalist, Profiler)
- **Calibrated Judge**: Dissent-aware adjudication with programmatic confidence damping

### Key Components

- **Contrastive RAG**: ChromaDB + cross-encoder reranking (BAAI/bge-reranker-v2-m3) with MMR diversity
- **GEPA Optimisation**: Automated prompt evolution via MLflow
- **Robust Span Localisation**: 5-strategy alignment (exact, case-insensitive, regex, fuzzy, cluster)

## Package Structure

```
psycomark/
├── __init__.py              # Package metadata
├── config.py                # Environment, LLM init, semaphore, retry wrapper
├── run.py                   # CLI entry point
│
├── schemas/                 # Pydantic data models
│   ├── s1.py                # S1 span extraction schemas (DD-CoT)
│   └── s2.py                # S2 classification schemas (Council/Judge)
│
├── agents/                  # LLM agent factories & runners
│   ├── s1_agents.py         # DD-CoT generator, critic, refiner
│   ├── s2_agents.py         # Parallel council, calibrated judge, dossier
│   └── span_utils.py        # 5-strategy span localisation & dedup
│
├── graphs/                  # LangGraph workflow definitions
│   ├── s1_graph.py          # S1: Generator → Critic → Refiner → Verifier
│   └── s2_graph.py          # S2: Profiler → Council → Judge
│
├── prompts/                 # Prompt management
│   ├── builder.py           # Hardcoded prompt construction (fallback)
│   └── loader.py            # File-based loading from prompts/openai/
│
├── rag/                     # Retrieval-Augmented Generation
│   └── retrieval.py         # ChromaDB, cross-encoder, MMR, stratified retrieval
│
├── evaluation/              # Metrics
│   └── metrics.py           # S1 Macro Overlap F1 (IoU≥0.5), S2 label normalisation
│
├── data/                    # Data preparation / curation
│   └── pipeline.py          # Safe train/dev preparation (RAG-optimised)
│
└── eda/                     # Exploratory analysis
    └── analysis_and_insights.py  # Dataset + span overlap analysis artifacts
```

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run both S1 and S2 on development data
psycomark --data dev_rehydrated.jsonl --task both

# S1 only with RAG
psycomark --data dev_rehydrated.jsonl --task s1 \
    --s1-rag chroma_db/ --s1-rag-name s1_fewshots

# S2 only
psycomark --data dev_rehydrated.jsonl --task s2 \
    --s2-rag chroma_db/ --s2-rag-name s2_precedents

# Resume interrupted run
psycomark --data dev_rehydrated.jsonl --task both --resume
```

## Build RAG Indexes (Optional)

```bash
# Optional progress bars
pip install -e ".[ragbuilder]"

# Build ChromaDB collections: s1_patterns + s2_precedents
python -m psycomark.rag.builder_online \
    --s1-input data/clean_v2/train_clean_s1.jsonl \
    --s2-input data/clean_v2/train_clean_s2.jsonl \
    --output-dir data/rag_openai_contrastive \
    --max-docs 500
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `PSYCOMARK_MODEL` | Model name (default: `gpt-5.2`) |
| `AWS_ACCESS_KEY_ID` | For Bedrock backend (optional) |
| `AWS_SECRET_ACCESS_KEY` | For Bedrock backend (optional) |
| `AWS_DEFAULT_REGION` | AWS region (default: `us-east-1`) |

## Evaluation

- **S1**: Macro Overlap F1-Score (IoU ≥ 0.5) across 5 fixed categories
- **S2**: Weighted F1-Score (conspiracy vs. non)

## Citation

```bibtex
@inproceedings{psycomark2025,
    title={PsyCoMark at SemEval-2025 Task 10: LLM-Powered Psycholinguistic Conspiracy Marker Extraction},
    year={2025},
    booktitle={Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)},
}
```
