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

---

## 🧠 Mapping from Reference Notebooks

WriteWise AI directly builds upon the concepts and code from the provided reference notebooks:

| Notebook Concept | Notebook Implementation | WriteWise AI Implementation |
|---|---|---|
| **Document Loading** | `pypdf.PdfReader` & UTF-8 `.txt` file reading | `document_processor.py` (`load_document`, `extract_text_from_pdf`, `extract_text_from_txt`) |
| **Text Chunking** | `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]` | `document_processor.py` (`chunk_document` with configurable `chunk_size=500`, `chunk_overlap=100`) |
| **Embeddings** | `SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")` | `vector_store.py` (`get_embedding_function`) |
| **Vector Database** | `chromadb.Client()` & collection creation | `vector_store.py` (`chromadb.PersistentClient(path="data/chroma_db/")` with collection `"writing_assistant_chunks"`) |
| **ID & Metadata Generation** | Unique chunk IDs (`f"{source}_chunk_{i}"`) and metadata dicts | `document_processor.py` (`stem_hash_chunk_{i}` with `source`, `chunk_index`, `char_count`, `doc_id`) |
| **Similarity Querying** | `collection.query(query_texts=[...], n_results=...)` | `vector_store.py` (`query_vector_store` with distance-to-similarity conversion `1 - distance`) |
| **Prompt Construction** | `build_user_prompt(query, context_chunks)` | `rag_pipeline.py` (`build_email_prompt`, `build_letter_prompt`) |
| **LLM Execution** | `groq.Groq` completions with system prompt | `llm_service.py` (`call_groq`, `call_llm` with strict anti-hallucination system prompt) |

---

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

## 📖 Walkthrough: Testing the Complete Workflow

### Step 1: Ingest Reference Guidelines
1. In the sidebar, select **"📚 Knowledge Base"**.
2. Click **"⚡ Load Built-in Reference Guidelines"** to instantly ingest standard email and formal letter conventions.
3. Switch to the **"📂 Stored Documents"** tab to see the ingested files (`email_writing_guide.txt` and `formal_letter_guide.txt`) and their chunk counts.
4. Try the **"🔎 Vector Search Explorer"** tab by entering a query like `"leave handover protocol"` to verify semantic similarity retrieval.

### Step 2: Generate an Email
1. In the sidebar, select **"✉️ Write an Email"**.
2. Choose a purpose (e.g., `Leave request`).
3. Fill in:
   - Recipient Name: `Dr. Marcus Vance`
   - Sender Name: `Alex Johnson`
   - Subject: `Leave Request: October 14 to October 16`
   - Main Message: `Requesting 3 days personal leave for a family commitment.`
   - Details: `Colleague Priya will cover urgent escalations. I will check urgent emails once daily.`
4. Click **"🚀 Generate Email"**.
5. Inspect the generated Subject line and Body.
6. Open **"🔍 View Retrieved References"** to see which chunks from the knowledge base were used.
7. Click **"👔 Make More Formal"** or **"✂️ Make More Concise"** to refine the text.
8. Click **"📥 Download as TXT"** to save your email.

### Step 3: Generate a Formal Letter
1. In the sidebar, select **"📜 Write a Letter"**.
2. Choose a type (e.g., `Formal request letters`).
3. Fill in:
   - Sender: `Alex Johnson, Department of Computer Science`
   - Recipient: `The Principal, Engineering College`
   - Subject: `Permission to participate in National AI Hackathon`
   - Details: `Dates: November 2-4, 2026. Requesting on-duty attendance.`
4. Click **"🚀 Generate Letter"**.
5. Review the structured formal layout (Heading, Date, Salutation, Body, Closing).

### Step 4: Review History
1. In the sidebar, select **"🕒 Writing History"**.
2. Inspect past entries. You can reopen any draft directly into the editor or download it again.

---

## 🔒 Security & Privacy

- **No Hardcoded Keys**: API keys are loaded solely from `.env` or in-memory Streamlit session state and are excluded from Git via `.gitignore`.
- **Local Storage**: ChromaDB vectors and draft histories are kept locally on your machine.
- **Prompt Injection Defense**: Uploaded documents are wrapped as strictly isolated reference material in prompts, preventing embedded text from overriding system boundaries.
- **Anti-Hallucination**: The system prompt strictly prohibits fabricating names, dates, promises, or personal facts not provided by the user.

---

## 📄 License
This project is open-source and intended for educational and personal productivity use.
