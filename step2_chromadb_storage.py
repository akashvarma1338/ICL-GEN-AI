"""
Step 2: Storing Chunks in ChromaDB
Based on: RAG_Step2_Storing_Chunks_in_ChromaDB_completed.ipynb

This module shows how to:
1. Initialize a persistent ChromaDB client on disk (data/chroma_db/).
2. Load the Sentence Transformers embedding function with 'all-MiniLM-L6-v2'.
3. Create or get the collection named 'writing_assistant_chunks'.
4. Store chunks, metadata, and IDs using upsert.
5. Inspect stored chunks and count them.
6. Delete chunks for a specific document safely.
"""

import os

# Linux / Streamlit Community Cloud SQLite fix for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "writing_assistant_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_collection(db_path=DB_PATH, collection_name=COLLECTION_NAME):
    """
    Connect to persistent ChromaDB on disk and return the collection with all-MiniLM-L6-v2.
    """
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder
    )
    return collection


def add_document(collection, text, source_name, chunk_size=500, chunk_overlap=100):
    """
    Split a document, build metadata/IDs, and store its chunks in ChromaDB.
    Matches Exercise 3 in Step 2 notebook.
    """
    from step1_load_and_chunk import chunk_text

    chunks, metadatas, chunk_ids = chunk_text(
        text=text,
        source_name=source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Use upsert to avoid duplicate-ID errors if the document is re-uploaded
    collection.upsert(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )

    return chunks, metadatas, chunk_ids


def get_all_sources(collection):
    """List all unique source documents stored in the collection."""
    if collection.count() == 0:
        return []
    data = collection.get(include=["metadatas"])
    sources = sorted(list({m["source"] for m in data["metadatas"] if "source" in m}))
    return sources


def delete_document(collection, source_name):
    """Delete all chunks belonging to a specific source file without touching others."""
    data = collection.get(where={"source": source_name})
    ids_to_delete = data.get("ids", [])
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def seed_sample_docs_if_empty(sample_dir="sample_docs"):
    """
    Automatically index sample reference guidelines into ChromaDB if the collection is empty.
    Returns the count of chunks in the collection.
    """
    collection = get_collection()
    if collection.count() == 0 and os.path.exists(sample_dir):
        from step1_load_and_chunk import load_document
        for filename in sorted(os.listdir(sample_dir)):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                file_path = os.path.join(sample_dir, filename)
                try:
                    doc = load_document(file_path)
                    add_document(collection, doc["text"], doc["filename"])
                except Exception:
                    pass
    return collection.count()


# Quick test when running directly
if __name__ == "__main__":
    from step1_load_and_chunk import load_document

    collection = get_collection()
    print("Collection name:", collection.name)
    print("Initial chunk count in DB:", collection.count())

    sample_file = "sample_docs/email_writing_guide.txt"
    if os.path.exists(sample_file):
        doc = load_document(sample_file)
        chunks, metas, ids = add_document(collection, doc["text"], doc["filename"])
        print(f"Added {len(chunks)} chunks from {doc['filename']}")
        print("Total chunks now in ChromaDB:", collection.count())
        print("Stored sources:", get_all_sources(collection))
