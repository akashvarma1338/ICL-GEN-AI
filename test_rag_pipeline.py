"""
WriteWise AI — Full Pipeline Test Suite
Tests the complete RAG pipeline: chunking, storage, retrieval, and generation.
Run: python test_rag_pipeline.py
"""

import os
import sys
from pathlib import Path

# Ensure app dir is in path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(override=True)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def test_step1_loading():
    print("\n── Step 1: Document Loading & Chunking ──")
    from step1_load_and_chunk import load_document, chunk_text

    # Test TXT loading
    sample = "sample_docs/email_writing_guide.txt"
    if not os.path.exists(sample):
        print(f"  {WARN} Sample file not found: {sample}")
        return False

    doc = load_document(sample)
    assert doc["filename"] == "email_writing_guide.txt", "Filename mismatch"
    assert doc["char_count"] > 0, "Empty document"
    print(f"  {PASS} load_document: '{doc['filename']}' ({doc['char_count']} chars, {doc['page_count']} pages)")

    chunks, metas, ids = chunk_text(doc["text"], doc["filename"], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 0, "No chunks generated"
    assert len(chunks) == len(metas) == len(ids), "Mismatched chunk/meta/id lengths"
    assert all(ids[i].startswith("email_writing_guide") for i in range(min(3, len(ids)))), "Bad chunk ID prefix"
    print(f"  {PASS} chunk_text: {len(chunks)} chunks | ID sample: {ids[0]}")

    return True


def test_step2_storage():
    print("\n── Step 2: ChromaDB Storage ──")
    from step2_chromadb_storage import get_collection, add_document, get_all_sources, delete_document
    from step1_load_and_chunk import load_document

    collection = get_collection()
    initial_count = collection.count()
    print(f"  ℹ️  Initial chunk count: {initial_count}")

    sample = "sample_docs/email_writing_guide.txt"
    if not os.path.exists(sample):
        print(f"  {WARN} Sample file missing — skipping storage test.")
        return False

    doc = load_document(sample)
    chunks, metas, ids = add_document(collection, doc["text"], doc["filename"])
    after_count = collection.count()
    print(f"  {PASS} add_document: {len(chunks)} chunks upserted (DB now has {after_count} total)")

    sources = get_all_sources(collection)
    assert doc["filename"] in sources, f"Source not found in DB: {sources}"
    print(f"  {PASS} get_all_sources: {sources}")

    return True


def test_step3_retrieval():
    print("\n── Step 3: Semantic Retrieval ──")
    from step2_chromadb_storage import get_collection, seed_sample_docs_if_empty
    from step3_retrieval import retrieve_context, format_context_for_llm

    seed_sample_docs_if_empty("sample_docs")
    collection = get_collection()

    if collection.count() == 0:
        print(f"  {WARN} Knowledge base empty — seed sample docs first.")
        return False

    query = "leave application handover contact"
    chunks = retrieve_context(query, collection=collection, n_results=3, similarity_threshold=0.1)
    print(f"  {PASS} retrieve_context: {len(chunks)} chunk(s) returned for query '{query}'")

    if chunks:
        top = chunks[0]
        assert "text" in top and "source" in top and "similarity" in top, "Missing keys in chunk"
        assert 0.0 <= top["similarity"] <= 1.0, f"Bad similarity score: {top['similarity']}"
        print(f"  {PASS} Top chunk: [{top['similarity']}] {top['source']} | {top['text'][:60]}...")

    formatted = format_context_for_llm(chunks)
    assert len(formatted) > 0, "Empty formatted context"
    print(f"  {PASS} format_context_for_llm: {len(formatted)} chars")

    return True


def test_step4_generation():
    print("\n── Step 4: LLM Generation (Offline Fallback) ──")
    from step4_llm_generation import generate_writing, build_user_prompt

    reqs = {
        "Purpose": "Leave request",
        "Recipient": "Dr. Sarah Jenkins",
        "Sender": "Alex Taylor",
        "Subject": "Leave Application: Nov 12-14",
        "Details": "Personal family commitment. Colleague John will cover escalations.",
        "Tone": "Professional",
        "Length": "Detailed"
    }

    # Test offline fallback (no API key)
    result = generate_writing("email", reqs, provider="offline")
    assert "text" in result, "Missing 'text' in result"
    assert len(result["text"]) > 50, "Draft too short"
    assert "retrieved_chunks" in result, "Missing 'retrieved_chunks'"
    print(f"  {PASS} generate_writing (offline): {len(result['text'])} chars | "
          f"grounded={result['is_grounded']} | chunks={len(result['retrieved_chunks'])}")

    # Test with Groq API key if available
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        print(f"  ℹ️  Groq API key found. Testing live generation...")
        try:
            result_live = generate_writing("email", reqs, provider="groq", api_key=api_key)
            assert len(result_live["text"]) > 100, "Live draft too short"
            print(f"  {PASS} generate_writing (Groq live): {len(result_live['text'])} chars")
        except Exception as e:
            print(f"  {WARN} Groq live call failed: {e}")
    else:
        print(f"  {WARN} No GROQ_API_KEY found — skipping live API test.")

    return True


def test_history_manager():
    print("\n── History Manager ──")
    from history_manager import save_draft, list_history, delete_draft, history_count

    reqs = {"Subject": "Test Email", "Recipient": "Test User"}
    path = save_draft("email", reqs, "Test draft content.", False, 0)
    assert os.path.exists(path), f"History file not created: {path}"
    print(f"  {PASS} save_draft: {Path(path).name}")

    records = list_history()
    assert len(records) >= 1, "No records found"
    print(f"  {PASS} list_history: {len(records)} record(s)")

    fname = records[0]["_filename"]
    deleted = delete_draft(fname)
    assert deleted, "Deletion failed"
    print(f"  {PASS} delete_draft: '{fname}' deleted")

    return True


def run_all_tests():
    print("=" * 55)
    print("  WriteWise AI — RAG Pipeline Test Suite")
    print("=" * 55)

    results = {
        "Step 1: Loading & Chunking": test_step1_loading(),
        "Step 2: ChromaDB Storage": test_step2_storage(),
        "Step 3: Retrieval": test_step3_retrieval(),
        "Step 4: LLM Generation": test_step4_generation(),
        "History Manager": test_history_manager(),
    }

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{len(results)} tests passed.")
    print("=" * 55)
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

