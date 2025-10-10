#!/usr/bin/env python3
import json
from pathlib import Path


def map_label_to_yn(doc_label):
    if doc_label == "conspiracy":
        return "Yes"
    if doc_label == "non":
        return "No"
    return None  # cant_tell or missing


def convert_split(latest_dir: Path, split: str):
    src = latest_dir / f"{split}.jsonl"
    dst = latest_dir / f"{split}_docclf.jsonl"
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            _id = item.get("_id") or item.get("doc_id")
            lab = map_label_to_yn(item.get("doc_label"))
            if _id is None or lab is None:
                continue
            fout.write(json.dumps({"_id": _id, "conspiracy": lab}) + "\n")
            kept += 1
    print(f"[{split}] wrote {kept} rows → {dst}")


def main():
    latest_ptr = Path("data/derived/psycomark_latest.txt")
    latest_dir = Path(latest_ptr.read_text().strip())
    print(f"Using latest: {latest_dir}")
    convert_split(latest_dir, "train")
    convert_split(latest_dir, "dev")


if __name__ == "__main__":
    main()
