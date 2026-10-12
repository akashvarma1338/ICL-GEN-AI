"""
Step 1: Document Loading and Chunking
Based on: RAG_Step1_Document_Loading_and_Chunking.ipynb

This module shows how to:
1. Read plain text (.txt) and PDF (.pdf) documents.
2. Extract clean text.
3. Split the text into overlapping chunks using RecursiveCharacterTextSplitter.
4. Create metadata and clean IDs for each chunk.
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default chunking settings (can be changed in settings)
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def load_txt(file_path):
    """Read plain text from a .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def load_pdf(file_path):
    """Extract text page by page from a .pdf file using pypdf."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n\n".join(pages_text).strip(), len(reader.pages)


def load_document(file_path):
    """
    Load a document (.txt or .pdf) and return its text, filename, and page count.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt":
        text = load_txt(file_path)
        page_count = 1
    elif ext == ".pdf":
        text, page_count = load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")

    if not text:
        raise ValueError(f"Document {filename} is empty or contains no readable text.")

    return {
        "filename": filename,
        "text": text,
        "char_count": len(text),
        "page_count": page_count
    }


def chunk_text(text, source_name, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Split text using RecursiveCharacterTextSplitter and create metadata and unique IDs.
    Returns: (chunks, metadatas, ids)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    # Clean the filename to make safe IDs
    stem = os.path.splitext(source_name)[0].replace(" ", "_").lower()

    metadatas = []
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": source_name,
            "chunk_index": i,
            "char_count": len(chunk)
        })
        chunk_ids.append(f"{stem}_chunk_{i}")

    return chunks, metadatas, chunk_ids


# Quick test when running directly
if __name__ == "__main__":
    sample_file = "sample_docs/email_writing_guide.txt"
    if os.path.exists(sample_file):
        print(f"Loading {sample_file}...")
        doc = load_document(sample_file)
        print(f"Loaded: {doc['filename']} ({doc['char_count']} chars, {doc['page_count']} pages)")

        chunks, metadatas, ids = chunk_text(doc["text"], doc["filename"], chunk_size=300, chunk_overlap=50)
        print(f"Created {len(chunks)} chunks.")
        print(f"\nExample Chunk 0 (ID: {ids[0]}):")
        print(chunks[0][:150], "...")
        print("Metadata:", metadatas[0])
