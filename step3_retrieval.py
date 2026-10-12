"""
Step 3: Retrieving Relevant Context
Based on: RAG_Step3_Retrieving_Relevant_Context.ipynb.ipynb

This module shows how to:
1. Search ChromaDB using collection.query().
2. Convert cosine distance to similarity score: similarity = 1 - distance.
3. Filter chunks by similarity_threshold (drops irrelevant results).
4. Optionally filter by source document using where={"source": filename}.
5. Format retrieved chunks into a clean context block for the LLM.
"""

from step2_chromadb_storage import get_collection


def distance_to_similarity(distance):
    """Convert distance to a similarity score between 0 and 1."""
    return round(max(0.0, 1.0 - distance), 4)


def retrieve_context(
    query,
    collection=None,
    n_results=4,
    similarity_threshold=0.25,
    source_filter=None
):
    """
    Query ChromaDB for chunks matching the query string.
    Only keeps chunks where similarity >= similarity_threshold.
    """
    if collection is None:
        collection = get_collection()

    total_chunks = collection.count()
    if total_chunks == 0 or not query.strip():
        return []

    # Don't ask for more chunks than exist in the collection
    k = min(n_results, total_chunks)

    where_clause = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    context_chunks = []

    if results and "documents" in results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(docs, metas, distances):
            similarity = distance_to_similarity(distance)

            if similarity >= similarity_threshold:
                context_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(distance, 4),
                    "similarity": similarity
                })

    return context_chunks


def format_context_for_llm(context_chunks):
    """
    Format retrieved passages into a readable block for the LLM prompt.
    Matches the pattern from Step 3 and Step 4 notebooks.
    """
    if not context_chunks:
        return "No relevant context found in the reference documents."

    lines = []
    for c in context_chunks:
        header = f"[Source: {c['source']}, chunk {c['chunk_index']} | Similarity: {c['similarity']}]"
        lines.append(f"{header}\n{c['text']}")

    return "\n\n".join(lines)


# Quick test when running directly
if __name__ == "__main__":
    collection = get_collection()
    test_query = "leave application handover and emergency contact"
    print("Running query:", test_query)

    chunks = retrieve_context(test_query, collection=collection, n_results=3, similarity_threshold=0.25)
    print(f"Retrieved {len(chunks)} chunks above threshold.")

    for i, c in enumerate(chunks, 1):
        print(f"\n--- Match {i} [{c['similarity']}] {c['source']} #{c['chunk_index']} ---")
        print(c["text"][:120], "...")

    print("\nFormatted Context Block Preview:")
    print(format_context_for_llm(chunks)[:300], "...")
