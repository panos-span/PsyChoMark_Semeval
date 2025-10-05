import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# --------------------
# Helpers
# --------------------


def find_latest_checkpoint(base_path, marker_type):
    """Return the latest checkpoint folder for a given single-type model."""
    full_path = f"{base_path}-{marker_type}"
    checkpoint_dirs = glob.glob(os.path.join(full_path, "checkpoint-*"))
    if not checkpoint_dirs:
        print(f"Warning: no 'checkpoint-*' found. Using: {full_path}")
        return full_path
    checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def load_data(file_path):
    """Load JSONL keeping order and an _id."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item["_id"] = item.get("_id", f"sample_{i}")
                item["text"] = item.get("text", "")
                item["markers"] = item.get("markers", [])
                item["conspiracy"] = item.get("conspiracy", "No")
                data.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i}: {line[:120]}...")
    print(f"Loaded {len(data)} samples for inference from {file_path}.")
    return data


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and keep offset mapping for span reconstruction."""
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
    )
    # Dummy labels for the Trainer API
    tokenized_inputs["labels"] = [
        [-100] * len(offset_map) for offset_map in tokenized_inputs["offset_mapping"]
    ]
    return tokenized_inputs


def reconstruct_spans(pred_ids, tokenized_dataset, id_to_label, pos_probs=None):
    """
    Convert token id predictions (0='O', 1='TYPE') into character spans.

    Returns TWO dicts indexed by sample idx:
      - submission_fmt[i] -> [{startIndex,endIndex,type,text}]
      - scored_fmt[i]     -> [{start,end,label,text,score}]
    """
    sub_markers = defaultdict(list)
    scored_markers = defaultdict(list)

    positive_label_type = id_to_label.get(1)
    if not positive_label_type or positive_label_type == "O":
        print("Error: id_to_label[1] must be the marker type for the binary model.")
        return sub_markers, scored_markers

    for i, ids in enumerate(pred_ids):
        offsets = tokenized_dataset[i]["offset_mapping"]
        original_text = tokenized_dataset[i]["text"]
        current_span_start_char = None
        token_scores = []  # collect per-token probs inside the current span

        for tok_idx, label_id in enumerate(ids):
            offset = offsets[tok_idx]
            is_special = (
                offset is None
                or offset[0] is None
                or offset[1] is None
                or (offset[0] == 0 and offset[1] == 0)
            )

            if is_special:
                # close any running span at previous token end
                if current_span_start_char is not None:
                    prev_end = None
                    if tok_idx > 0 and offsets[tok_idx - 1][1] is not None:
                        prev_end = offsets[tok_idx - 1][1]
                    if prev_end is not None and prev_end > current_span_start_char:
                        span_text = original_text[current_span_start_char:prev_end]
                        score = float(np.mean(token_scores)) if token_scores else 0.5
                        sub_markers[i].append(
                            {
                                "startIndex": current_span_start_char,
                                "endIndex": prev_end,
                                "type": positive_label_type,
                                "text": span_text,
                            }
                        )
                        scored_markers[i].append(
                            {
                                "start": current_span_start_char,
                                "end": prev_end,
                                "label": positive_label_type,
                                "text": span_text,
                                "score": score,
                            }
                        )
                    current_span_start_char = None
                    token_scores = []
                continue

            label = id_to_label[label_id]
            start_char = offset[0]

            if label == positive_label_type:
                if current_span_start_char is None:
                    current_span_start_char = start_char
                    token_scores = []
                if pos_probs is not None:
                    # collect this token's positive-class prob
                    try:
                        token_scores.append(float(pos_probs[i, tok_idx]))
                    except Exception:
                        pass
            elif label == "O":
                if current_span_start_char is not None:
                    prev_end = (
                        offsets[tok_idx - 1][1]
                        if tok_idx > 0 and offsets[tok_idx - 1][1] is not None
                        else start_char
                    )
                    if prev_end > current_span_start_char:
                        span_text = original_text[current_span_start_char:prev_end]
                        score = float(np.mean(token_scores)) if token_scores else 0.5
                        sub_markers[i].append(
                            {
                                "startIndex": current_span_start_char,
                                "endIndex": prev_end,
                                "type": positive_label_type,
                                "text": span_text,
                            }
                        )
                        scored_markers[i].append(
                            {
                                "start": current_span_start_char,
                                "end": prev_end,
                                "label": positive_label_type,
                                "text": span_text,
                                "score": score,
                            }
                        )
                    current_span_start_char = None
                    token_scores = []

        # close if span is still open
        if current_span_start_char is not None:
            last_valid_end = None
            for back in range(len(ids) - 1, -1, -1):
                off = offsets[back]
                if off and off[1] and off[1] != 0:
                    last_valid_end = off[1]
                    break
            if last_valid_end and last_valid_end > current_span_start_char:
                span_text = original_text[current_span_start_char:last_valid_end]
                score = float(np.mean(token_scores)) if token_scores else 0.5
                sub_markers[i].append(
                    {
                        "startIndex": current_span_start_char,
                        "endIndex": last_valid_end,
                        "type": positive_label_type,
                        "text": span_text,
                    }
                )
                scored_markers[i].append(
                    {
                        "start": current_span_start_char,
                        "end": last_valid_end,
                        "label": positive_label_type,
                        "text": span_text,
                        "score": score,
                    }
                )

    return sub_markers, scored_markers


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-base", default="distilbert-single-type-simplified")
    parser.add_argument(
        "--markers",
        nargs="+",
        default=["Action", "Actor", "Effect", "Evidence", "Victim"],
    )
    parser.add_argument("--test-file", default="dev_rehydrated.jsonl")
    parser.add_argument("--submission-file", default="submission.jsonl")
    parser.add_argument(
        "--scored-file",
        default=None,
        help="Unified spans for ensembling; default is predictions_s1_single.jsonl next to submission.",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # Resolve derived scored path
    if args.scored_file is None:
        out_dir = os.path.dirname(args.submission_file) or "."
        args.scored_file = os.path.join(out_dir, "predictions_s1_single.jsonl")

    # 1) Load data
    raw_data = load_data(args.test_file)
    if not raw_data:
        print("Error: no data loaded. Aborting.")
        sys.exit(1)

    unique_ids = [d["_id"] for d in raw_data]
    conspiracy_keys = [d["conspiracy"] for d in raw_data]
    test_dataset = Dataset.from_list(raw_data)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    tokenized_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=[
            c
            for c in test_dataset.column_names
            if c not in ["text", "offset_mapping", "_id", "conspiracy"]
        ],
        fn_kwargs={"tokenizer": tokenizer},
    )

    # holders for submissions and scored unified spans
    sub_agg = defaultdict(list)
    scored_agg = defaultdict(list)

    # 2) Per-marker inference
    for marker_type in args.markers:
        model_dir = find_latest_checkpoint(args.models_base, marker_type)
        print(f"\n--- Inference for: {marker_type} ---\nModel: {model_dir}")

        try:
            model = DistilBertForTokenClassification.from_pretrained(model_dir)
            id_to_label = {0: "O", 1: marker_type}
        except Exception as e:
            print(f"Error loading {marker_type} from '{model_dir}': {e}")
            continue

        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"./tmp_inference_span_{marker_type}",
                per_device_eval_batch_size=args.batch_size,
                report_to="none",
            ),
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        pred_out = trainer.predict(tokenized_test_dataset)
        logits = pred_out.predictions  # [N, T, 2]
        # softmax to probs
        exps = np.exp(logits - logits.max(axis=2, keepdims=True))
        probs = exps / exps.sum(axis=2, keepdims=True)
        pos_probs = probs[:, :, 1]  # P(label=1) per token

        pred_ids = np.argmax(logits, axis=2)

        sub_map, scored_map = reconstruct_spans(
            pred_ids, tokenized_test_dataset, id_to_label, pos_probs=pos_probs
        )

        # aggregate
        for i, lst in sub_map.items():
            sub_agg[i].extend(lst)
        for i, lst in scored_map.items():
            scored_agg[i].extend(lst)

    # 3) Write original submission JSONL (unchanged schema)
    print(f"\nSaving submission to {args.submission_file} ...")
    with open(args.submission_file, "w", encoding="utf-8") as f:
        for i in range(len(raw_data)):
            f.write(
                json.dumps(
                    {
                        "_id": unique_ids[i],
                        "conspiracy": conspiracy_keys[i],
                        "markers": sub_agg.get(i, []),
                    }
                )
                + "\n"
            )
    print("Submission file written.")

    # 4) Write scored unified spans for ensembling/merging
    print(f"Saving scored unified spans to {args.scored_file} ...")
    with open(args.scored_file, "w", encoding="utf-8") as f:
        for i in range(len(raw_data)):
            f.write(
                json.dumps(
                    {
                        "_id": unique_ids[i],
                        "markers": scored_agg.get(i, []),
                    }
                )
                + "\n"
            )
    print("Scored span file written.")
