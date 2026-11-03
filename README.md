The project livelink is:https://emailassisstant.streamlit.app/
# WriteWise AI — RAG-Based Email & Letter Writing Assistant

WriteWise AI is an end-to-end **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **ChromaDB**, **Sentence Transformers**, and **Groq LLM**. It assists users in drafting grounded, grammatically polished, and professionally structured emails and formal letters using their own uploaded reference documents and style guidelines.

---

## 🌟 Key Features

1. **Email Drafting with RAG Grounding**:
   - Covers 10 distinct email types (Job application, Leave request, Follow-up, Meeting request, Permission, Apology, Complaint, College communication, etc.).
   - Gathers user requirements (Sender, Recipient, Subject, Main purpose, Details, Tone, Length, Language).
   - Generates separate, editable fields for Subject line and Email Body.

2. **Structured Formal Letter Drafting**:
   - Supports 10 formal letter categories (Formal request, Leave letters, Permissions, Complaints, Academic letters, Internship/Job applications).
   - Adopts proper letter etiquette: Sender info, Date, Recipient designation/organization, Subject line, Salutation, Structured paragraphs, and Complimentary close.

3. **Persistent Knowledge Base**:
   - Supports uploading multiple `.txt` and `.pdf` files.
   - Page-by-page extraction via `pypdf`.
   - Text chunking using LangChain's `RecursiveCharacterTextSplitter`.
   - Persistent vector storage in `data/chroma_db/` using ChromaDB's persistent client.
   - Built-in reference guidelines for instant testing.
   - Safe per-document deletion and database reset capabilities.

4. **Accurate Semantic Retrieval & Inspection**:
   - Generates embeddings using `all-MiniLM-L6-v2` (`SentenceTransformerEmbeddingFunction`).
   - Retrieves top matching chunks with configurable `top_k` and similarity threshold.
   - Expandable **"View Retrieved References"** drawer on every draft showing source filename, chunk index, similarity score, and exact text passages.
   - Live **Vector Search Explorer** in Knowledge Base for interactive similarity testing.

5. **Draft Refinement & Export**:
   - In-place editing of generated drafts.
   - One-click refinement actions:
     - 👔 *Make More Formal*
     - ✂️ *Make More Concise*
     - 📝 *Expand Content*
     - ✨ *Improve Grammar*
     - 🔄 *Regenerate*
   - Direct download as a clean `.txt` file.

6. **Local Writing History**:
   - Automatically archives all generated drafts with timestamps, recipient info, and grounding tags in `data/history/`.
   - Allows previewing, downloading, or reopening past drafts directly in the editor.

## 🛠️ Project Structure

```text
ICL/
│
├── app.py                     # Streamlit frontend, multi-page router, and interactive UI
├── config.py                  # Centralized configuration (paths, defaults, model constants)
├── document_processor.py      # Text extraction for TXT/PDF, chunking, and metadata generation
├── vector_store.py            # ChromaDB persistent client, upsert, query, filtering, deletion
├── llm_service.py             # Groq & Gemini API dispatch, anti-hallucination system prompt, refinement
├── rag_pipeline.py            # Search query construction, context formatting, and grounded generation
├── history_manager.py         # Local JSON history storage for previous drafts
├── requirements.txt           # Verified Python package dependencies
├── .env.example               # Template for API keys and environment variables
├── .gitignore                 # Files and directories excluded from version control
├── README.md                  # Detailed documentation and beginner guide
│
├── sample_docs/               # Ready-to-use reference documents
│   ├── email_writing_guide.txt
│   └── formal_letter_guide.txt
│
└── data/                      # Persistent runtime data
    ├── chroma_db/             # ChromaDB persistent vector database directory
    └── history/               # Saved draft history records
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- (Recommended) A free Groq API key from [console.groq.com](https://console.groq.com/keys).

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment configuration:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and insert your Groq API key:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

*(Note: You can also configure your API key directly within the app's **Settings** page at any time).*

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---


