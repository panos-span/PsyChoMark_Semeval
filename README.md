# AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection

This repository contains the official implementation of the AILS-NTUA system for **SemEval-2026 Task 10: Psycholinguistic Conspiracy Marker Extraction and Detection**.

**Authors:** Panagiotis Spanakis, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou  
*National Technical University of Athens (NTUA)*

## Overview
We propose a novel two-stage agentic LLM architecture built to accurately separate complex psycholinguistic markers (S1) and detect conspiracy endorsement avoiding common pitfalls (S2).

Our approach features:
- **Subtask 1 (Marker Extraction):** A **Dynamic Discriminative Chain-of-Thought (DD-CoT)** framework that combines LLM reasoning with a deterministic span verifier to guarantee character-accurate marker extraction.
- **Subtask 2 (Conspiracy Detection):** An **Anti-Echo Chamber** architecture using a *Parallel Council* (Prosecutor, Defense Attorney, Literalist, and Stance Profiler) adjudicated by a *Calibrated Judge* to separate endorsement from objective reporting (the "Reporter Trap").

## System Architecture

Our agentic workflow isolates LLM semantic decisions from deterministic operations:

1. **S1 Pipeline (DD-CoT Self-Refine):** 
   - Uses stratified contrastive retrieval to fetch relevant few-shot examples.
   - The generator proposes marker spans and types (e.g., Actor, Action, Victim, Effect, Evidence).
   - An Enhanced Critic and Refiner loop perfects the generated extractions.
   - A Deterministic Verifier anchors the LLM-generated string precisely back to the source text.

2. **S2 Pipeline (Parallel Council):**
   - Retrieves "Hard Negatives" to help the model distinguish between simply discussing strings and actually endorsing a conspiracy.
   - A Forensic Profiler calculates lightweight deterministic signals (e.g., attribution density, shouting score, passive voice).
   - A Parallel Council independently debates the source text.
   - A Calibrated Judge issues the final verdict based on the council's diverse insights and forensic guidelines.

## Quickstart Using `uv`

This project is configured to be run and managed blazingly fast using [uv](https://github.com/astral-sh/uv).

### 1. Install `uv`
If you haven't already, install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup and Run
You can run the project directly without manually creating a virtual environment. `uv` will automatically read `pyproject.toml`, install dependencies, and execute the entry point.

Run the main pipeline:
```bash
# This will automatically create a virtual environment, sync dependencies, and run the command
uv run psycomark
```

### 3. Development Commands
To run specific components or scripts using `uv`:

```bash
# Run testing
uv run pytest

# Run experimental pipelines (e.g., bedrock experiments)
uv run python run_bedrock_experiments.py
```

## Project Structure
- `psycomark/` - Core package containing the agents, verifiers, and council architecture.
- `scripts/` - Assorted scripts for patching, extracting, and analyzing data.
- `configs/` - Configuration parameters.
- `data/` - Training, dev, and test dataset files.

## Citation
If you use this code or approach in your research, please refer to our paper:
> Spanakis, P., Lymperaiou, M., Filandrianos, G., Voulodimos, A., & Stamou, G. (2026). *AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection*. In Proceedings of SemEval-2026.
