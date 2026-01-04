#!/usr/bin/env python3
"""
End-to-end test of the S1 optimization pipeline.
Tests: Data Loading → Predict Wrapper → Scorer
"""

import sys
import pathlib
import json
import asyncio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from loguru import logger

# Configure logger for test output
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<level>{level: <8}</level> | {message}")

from pydanticai2.optimize_s1 import (
    load_eval_data,
    s1_rich_scorer,
    find_best_span,
    normalize_label,
)
from pydanticai2.psycomark_agents import (
    get_rag_collection,
    run_s1_discriminative,
    retrieve_stratified_s1,
)


def main():
    print("\n" + "=" * 70)
    print("🔬 END-TO-END PIPELINE TEST")
    print("=" * 70)

    # 1. Initialize RAG
    print("\n[1] Initializing RAG collection...")
    rag_dir = "data/rag_online_v3"
    rag_collection = None
    if pathlib.Path(rag_dir).exists():
        rag_collection = get_rag_collection(rag_dir, "s1_markers")
        print(f"    ✅ RAG loaded from {rag_dir}")
    else:
        print(f"    ⚠️ RAG directory not found: {rag_dir}")

    # 2. Load single test document
    print("\n[2] Loading test document...")
    data = load_eval_data("data/gold/optimization_set.jsonl", limit=1)

    if not data:
        print("    ❌ No data loaded!")
        return

    item = data[0]
    text = item["inputs"]["text"]
    passthrough_gold = item["inputs"]["passthrough_gold"]
    gold_data = json.loads(passthrough_gold)

    print(f"    Doc ID: {gold_data['doc_id']}")
    print(f"    Text length: {len(text)} chars")
    print(f"    Text preview: {text[:100]}...")
    print(f"    Gold spans: {len(gold_data['gold_spans'])}")

    for i, span in enumerate(gold_data["gold_spans"][:5]):
        print(
            f"      [{i+1}] {span['label']}: '{span['text'][:40]}...' @ [{span['start']}:{span['end']}]"
        )

    # 3. Run prediction (DIRECTLY calling run_s1_discriminative, not through wrapper)
    print("\n[3] Running run_s1_discriminative...")
    print("    (This will call the LLM - may take 10-30 seconds)")

    try:
        # Get few-shot examples
        few_shots = []
        if rag_collection:
            few_shots = retrieve_stratified_s1(rag_collection, text, k_total=6)
            print(f"    Retrieved {len(few_shots)} few-shot examples")

        # Run the actual model
        spans = asyncio.run(run_s1_discriminative(text, few_shots=few_shots))

        print(f"\n    ✅ Model returned {len(spans)} spans!")

        # Convert and localize spans
        pred_spans = []
        assigned_count = {}

        for s in spans:
            if hasattr(s, "model_dump"):
                span_dict = s.model_dump()
            elif hasattr(s, "dict"):
                span_dict = s.dict()
            else:
                span_dict = {"text": str(s), "label": "Unknown"}

            # Normalize label
            if "label" in span_dict:
                span_dict["label"] = normalize_label(span_dict["label"])

            # Localize span
            span_text = span_dict.get("text", "")
            if span_text:
                key = (span_dict.get("label", ""), span_text.strip())
                nth = assigned_count.get(key, 0)
                start, end = find_best_span(text, span_text, nth=nth)

                if start != -1:
                    span_dict["start"] = start
                    span_dict["end"] = end
                    span_dict["text"] = text[start:end]
                    assigned_count[key] = nth + 1
                else:
                    span_dict["start"] = -1
                    span_dict["end"] = -1

            pred_spans.append(span_dict)

        for i, span in enumerate(pred_spans[:8]):
            start = span.get("start", "?")
            end = span.get("end", "?")
            label = span.get("label", "?")
            text_preview = span.get("text", "")[:40]
            print(f"      [{i+1}] {label}: '{text_preview}...' @ [{start}:{end}]")

        if len(pred_spans) > 8:
            print(f"      ... and {len(pred_spans) - 8} more")

    except Exception as e:
        print(f"    ❌ Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Run scorer
    print("\n[4] Running scorer...")
    result = {
        "final_spans": pred_spans,
        "passthrough_gold_ref": passthrough_gold,
    }

    try:
        feedback = s1_rich_scorer(outputs=result, expectations={})

        print(f"\n    📊 SCORE: {feedback.value:.2f}")
        print(f"    📝 FEEDBACK: {feedback.rationale}")

    except Exception as e:
        print(f"    ❌ Scoring failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 5. Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"  Gold spans:      {len(gold_data['gold_spans'])}")
    print(f"  Predicted spans: {len(pred_spans)}")
    print(f"  Score:           {feedback.value:.2f}")
    print(
        f"  Status:          {'✅ PASS' if feedback.value > 0.5 else '⚠️ NEEDS IMPROVEMENT'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
