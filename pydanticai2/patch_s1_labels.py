import json
import argparse
import sys
import pathlib
import chromadb
from loguru import logger
from tqdm import tqdm


def patch_rag_labels(source_jsonl: str, rag_dir: str):
    p = pathlib.Path(source_jsonl)
    if not p.exists():
        logger.error(f"Source file not found: {source_jsonl}")
        return

    # 1. Build Lookup Map from Source Data
    # We map doc_id -> label
    logger.info(f"Reading source labels from {source_jsonl}...")
    id_to_label = {}

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Normalize ID logic to match your builder script
                doc_id = str(row.get("doc_id") or row.get("_id") or row.get("id"))

                # Normalize Label (Force lowercase 'conspiracy' or 'non')
                raw_label = row.get("label", "non")
                if str(raw_label).lower() in ["conspiracy", "yes", "true"]:
                    clean_label = "conspiracy"
                else:
                    clean_label = "non"

                id_to_label[doc_id] = clean_label
            except Exception:
                continue

    logger.info(f"Loaded {len(id_to_label)} labels from source.")

    # 2. Connect to RAG
    logger.info(f"Connecting to RAG at {rag_dir}...")
    try:
        client = chromadb.PersistentClient(path=rag_dir)
        # Note: We don't need the embedding function just to update metadata!
        collection = client.get_collection(name="s1_markers")
    except Exception as e:
        logger.error(f"Failed to load collection: {e}")
        return

    current_count = collection.count()
    logger.info(f"Found {current_count} documents in RAG.")

    # 3. Fetch Existing Metadata
    # We fetch everything (ids and metadatas)
    existing_data = collection.get()
    existing_ids = existing_data["ids"]
    existing_metas = existing_data["metadatas"]

    # 4. Prepare Updates
    ids_to_update = []
    metas_to_update = []

    matched_count = 0
    missing_count = 0

    for doc_id, meta in zip(existing_ids, existing_metas):
        if doc_id in id_to_label:
            # Found a match!
            new_label = id_to_label[doc_id]

            # CRITICAL: Copy existing meta so we don't lose 'spans_json'
            updated_meta = meta.copy() if meta else {}
            updated_meta["label"] = new_label

            ids_to_update.append(doc_id)
            metas_to_update.append(updated_meta)
            matched_count += 1
        else:
            missing_count += 1
            # Optional: Assume 'non' or leave as is?
            # Usually better to leave as is or log warning.
            logger.warning(f"ID {doc_id} exists in RAG but not in source file.")

    # 5. Commit Updates to ChromaDB
    if ids_to_update:
        logger.info(f"Patching {len(ids_to_update)} documents...")

        # ChromaDB allows batch updates. We do it in chunks to be safe.
        batch_size = 100
        for i in tqdm(range(0, len(ids_to_update), batch_size)):
            end = i + batch_size
            collection.update(
                ids=ids_to_update[i:end],
                metadatas=metas_to_update[i:end],
                # Note: We do NOT pass 'embeddings' or 'documents', so they remain unchanged.
            )
        logger.success("Patch Complete.")
    else:
        logger.warning(
            "No updates were prepared. Check if IDs match between RAG and Source."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch RAG Metadata Labels")
    parser.add_argument(
        "--source",
        default="data/clean/train_clean_s1.jsonl",
        help="Original source with labels",
    )
    parser.add_argument(
        "--rag-dir", default="data/rag_online_v3", help="Path to ChromaDB"
    )
    args = parser.parse_args()

    patch_rag_labels(args.source, args.rag_dir)
