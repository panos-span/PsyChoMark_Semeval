import sys
import chromadb
from loguru import logger

# Point this to your actual RAG directory
RAG_DIR = "data/rag_online_v3"


def inspect_rag():
    logger.info(f"Connecting to {RAG_DIR}...")
    try:
        client = chromadb.PersistentClient(path=RAG_DIR)

        # 1. Check Collections
        collections = client.list_collections()
        logger.info(f"Found Collections: {[c.name for c in collections]}")

        if "s1_markers" not in [c.name for c in collections]:
            logger.error("Collection 's1_markers' NOT found!")
            return

        # 2. Peek at Data (No Embedding Function needed for peek)
        coll = client.get_collection("s1_markers")
        count = coll.count()
        logger.info(f"Collection 's1_markers' contains {count} documents.")

        if count == 0:
            logger.warning("Collection is EMPTY. You need to run hydration.")
            return

        # 3. Dump Metadata
        logger.info("--- Peeking at first 5 items ---")
        peek = coll.peek(limit=5)

        if peek["metadatas"]:
            for i, meta in enumerate(peek["metadatas"]):
                print(f"[{i}] Metadata: {meta}")
        else:
            logger.warning("No metadata found in peek!")

    except Exception as e:
        logger.exception("Failed to inspect RAG")


if __name__ == "__main__":
    inspect_rag()
