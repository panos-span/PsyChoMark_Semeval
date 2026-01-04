import chromadb

client = chromadb.PersistentClient(path="data/rag_online_v3")
coll = client.get_collection("s1_markers")

# Get the first 5 items, including IDs
data = coll.get(limit=5, include=["metadatas"])

print(f"{'ID':<15} | {'LABEL':<10}")
print("-" * 30)
for i, doc_id in enumerate(data["ids"]):
    label = data["metadatas"][i].get("label", "MISSING")
    print(f"{doc_id:<15} | {label:<10}")
