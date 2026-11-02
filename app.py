"""
WriteWise AI — Email and Letter Writing Assistant
Powered by RAG (ChromaDB + Sentence Transformers + Groq LLM).
API key is strictly loaded from the .env file.

Features:
- ✉️  Email drafting with RAG grounding (10 email types)
- 📜  Formal letter drafting (9 letter types)
- 📚  Knowledge Base management: upload docs, inspect, delete, vector search
- 🕒  Writing History: view, reopen, download past drafts
- ⚡  One-click refinements: formal, detailed, concise, grammar fix, regenerate
- 📥  Download drafts as .txt
"""

# Linux / Streamlit Community Cloud SQLite fix for ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure application directory is always in sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Load environment variables from .env
load_dotenv(override=True)


def find_groq_key():
    """Thoroughly search for the Groq API key across Streamlit secrets and environment."""
    load_dotenv(override=True)
    try:
        if hasattr(st, "secrets"):
            for k in ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"]:
                try:
                    if k in st.secrets:
                        val = str(st.secrets[k]).strip().strip("'\"")
                        if val:
                            return val
                except Exception:
                    pass
    except Exception:
        pass

    for k in ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"]:
        val = os.getenv(k, "").strip().strip("'\"")
        if val:
            return val
    return ""


from config import GROQ_MODELS, DEFAULT_GROQ_MODEL
from step1_load_and_chunk import load_document, chunk_text
from step2_chromadb_storage import (
    get_collection,
    add_document,
    get_all_sources,
    delete_document,
    seed_sample_docs_if_empty,
)
from step3_retrieval import retrieve_context, format_context_for_llm
from step4_llm_generation import generate_writing, refine_draft
from history_manager import save_draft, list_history, load_draft, delete_draft, clear_all_history, history_count


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="WriteWise — Email & Letter Assistant",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
    }
    .stButton button {
        border-radius: 6px;
    }
    .badge-grounded {
        background: #D1FAE5; color: #065F46;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.82rem; font-weight: 600;
    }
    .badge-offline {
        background: #FEF3C7; color: #92400E;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.82rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Shared State & Init
# ==============================================================================

@st.cache_resource(show_spinner=False)
def init_knowledge_base():
    """Seed sample docs into ChromaDB on startup if collection is empty."""
    try:
        return seed_sample_docs_if_empty("sample_docs")
    except Exception:
        return 0

init_knowledge_base()

# Initialize session state defaults
_defaults = {
    "current_email": None,
    "current_letter": None,
    "nav_page": "✉️ Write an Email",
    "email_rev": 0,
    "letter_rev": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("## ✍️ WriteWise AI")
    st.caption("RAG-Powered Professional Writing Assistant")
    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["✉️ Write an Email", "📜 Write a Letter", "📚 Knowledge Base", "🕒 Writing History"],
        key="nav_page",
    )

    st.divider()

    # API Key
    configured_key = find_groq_key()
    st.subheader("🔑 Groq API Key")
    if configured_key:
        masked = (configured_key[:6] + "••••" + configured_key[-4:]) if len(configured_key) > 10 else "••••••••"
        st.success(f"🟢 Connected (`{masked}`)")
        with st.expander("⚙️ Override Key", expanded=False):
            override_key = st.text_input(
                "Custom Key", value="", type="password",
                placeholder="Paste to override...",
            ).strip()
        active_api_key = override_key if override_key else configured_key
    else:
        st.warning("⚠️ No API Key detected")
        active_api_key = st.text_input(
            "Groq API Key", type="password", placeholder="gsk_...",
            help="Get a free key at console.groq.com/keys"
        ).strip()

    has_api_key = bool(active_api_key)
    if has_api_key:
        os.environ["GROQ_API_KEY"] = active_api_key

    st.divider()

    # Model selection
    st.subheader("🧠 Model")
    selected_model = st.selectbox(
        "Groq Model", GROQ_MODELS,
        index=GROQ_MODELS.index(DEFAULT_GROQ_MODEL) if DEFAULT_GROQ_MODEL in GROQ_MODELS else 0,
    )

    st.divider()
    kb_count = get_collection().count()
    st.metric("📦 KB Chunks", kb_count, help="Total chunks indexed in ChromaDB")


# ==============================================================================
# Helper: Save to history
# ==============================================================================
def _save_to_history(doc_type, requirements, result):
    try:
        save_draft(
            doc_type=doc_type,
            requirements=requirements,
            draft_text=result.get("text", ""),
            is_grounded=result.get("is_grounded", False),
            chunk_count=len(result.get("retrieved_chunks", [])),
        )
    except Exception:
        pass


# ==============================================================================
# Helper: RAG reference expander
# ==============================================================================
def _show_rag_expander(result, key_prefix="rag"):
    chunks = result.get("retrieved_chunks", [])
    label = f"📚 View Retrieved References ({len(chunks)} chunks)" if chunks else "📚 Retrieved References"
    with st.expander(label, expanded=False):
        if chunks:
            for i, c in enumerate(chunks, 1):
                st.markdown(
                    f"**Reference #{i}** · Source: `{c['source']}` · "
                    f"Chunk #{c['chunk_index']} · Similarity: `{c['similarity']}`"
                )
                st.caption(c["text"])
        else:
            st.info("No specific reference chunks met the similarity threshold. "
                    "Drafted using standard best practices.")


# ==============================================================================
# Helper: Refinement buttons
# ==============================================================================
def _refinement_buttons(state_key, current_text, rev_key, api_key, model, prefix):
    st.markdown("**⚡ Refine Draft:**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def _refine(instruction):
        with st.spinner("Refining draft with AI..."):
            try:
                revised = refine_draft(current_text, instruction, api_key=api_key, model=model)
                if revised and revised.strip() != current_text.strip():
                    st.session_state[state_key]["text"] = revised
                    st.session_state[rev_key] = st.session_state.get(rev_key, 0) + 1
                    st.rerun()
                elif revised and revised.strip() == current_text.strip():
                    st.info("The draft already aligns with this refinement instruction.")
                else:
                    st.warning("No revised draft returned.")
            except Exception as e:
                st.error(f"Refinement error: {e}")

    with c1:
        if st.button("👔 Formal", key=f"{prefix}_formal", use_container_width=True):
            _refine("Make this draft more formal, polite, and professional.")
    with c2:
        if st.button("📝 Detailed", key=f"{prefix}_detailed", use_container_width=True):
            _refine(
                "Substantially expand and elaborate this draft with comprehensive context, "
                "multi-paragraph explanations, background reasoning, timeline/coverage details, "
                "and structured next steps while preserving all core facts."
            )
    with c3:
        if st.button("✂️ Concise", key=f"{prefix}_concise", use_container_width=True):
            _refine("Make this draft concise, brief, and direct while preserving all essential details.")
    with c4:
        if st.button("✨ Grammar", key=f"{prefix}_grammar", use_container_width=True):
            _refine("Correct all grammar, spelling, punctuation, and phrasing issues while keeping the meaning intact.")
    with c5:
        if st.button("🔄 Regenerate", key=f"{prefix}_regen", use_container_width=True):
            _refine("Rewrite this draft completely from scratch maintaining all the same requirements, recipient, and purpose, but with a fresh voice and structure.")
    with c6:
        if st.button("🗑️ Clear", key=f"{prefix}_clear", use_container_width=True):
            st.session_state[state_key] = None
            st.session_state[rev_key] = st.session_state.get(rev_key, 0) + 1
            st.rerun()


# ==============================================================================
# PAGE 1: Write an Email
# ==============================================================================
if page == "✉️ Write an Email":
    st.markdown('<div class="main-header">✉️ Write an Email</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered professional email drafting, grounded in your reference documents.</div>', unsafe_allow_html=True)

    with st.form("email_form"):
        col1, col2 = st.columns(2)
        with col1:
            purpose = st.selectbox("Email Purpose / Category *", [
                "Leave request", "Job application", "Meeting request",
                "Follow-up email", "Thank-you email", "Apology email",
                "Complaint / Issue escalation", "Permission request",
                "Academic communication", "General professional email"
            ])
            recipient_name = st.text_input("Recipient Name / Designation *",
                                           placeholder="e.g., Dr. Sarah Jenkins / Hiring Manager")
            sender_name = st.text_input("Sender Name *", placeholder="e.g., Alex Taylor")
            subject = st.text_input("Subject or Topic *",
                                    placeholder="e.g., Sick Leave Application - Oct 14 / Project Update")
            date_str = st.text_input("Date", value=datetime.now().strftime("%B %d, %Y"))

        with col2:
            tone = st.selectbox("Tone", ["Professional", "Formal", "Polite", "Friendly", "Urgent"])
            length = st.selectbox("Length", ["Detailed", "Standard", "Concise"], index=0,
                                  help="'Detailed' gives 3-5 rich paragraphs.")
            details = st.text_area(
                "Important Details / Main Message *",
                placeholder="State reason, dates, context, handovers, contact details...",
                height=150
            )

        submit_email = st.form_submit_button("🚀 Generate Email", use_container_width=True, type="primary")

    if submit_email:
        missing = [f for f, v in [
            ("Recipient Name", recipient_name), ("Sender Name", sender_name),
            ("Subject", subject), ("Details", details)
        ] if not v.strip()]
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            reqs = {
                "Purpose": purpose, "Recipient": recipient_name.strip(),
                "Sender": sender_name.strip(), "Subject": subject.strip(),
                "Date": date_str.strip(), "Details": details.strip(),
                "Tone": tone, "Length": length
            }
            with st.spinner("Retrieving guidelines and drafting email..."):
                try:
                    result = generate_writing(
                        doc_type="email", requirements=reqs,
                        provider="groq" if has_api_key else "offline",
                        api_key=active_api_key, model=selected_model,
                        top_k=4, similarity_threshold=0.25
                    )
                    st.session_state.current_email = result
                    st.session_state["email_rev"] = st.session_state.get("email_rev", 0) + 1
                    _save_to_history("email", reqs, result)
                    st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "invalid_api_key" in err_msg:
                        st.error("❌ **Invalid Groq API Key (401)**. "
                                 "Get a free key at [console.groq.com/keys](https://console.groq.com/keys).")
                    else:
                        st.error(f"Error generating email: {e}")

    # Output Section
    if st.session_state.current_email:
        st.markdown("---")
        result = st.session_state.current_email
        grounded = result.get("is_grounded", False)
        badge = "badge-grounded" if grounded else "badge-offline"
        badge_txt = "✅ RAG Grounded" if grounded else "⚡ Standard Draft"
        st.markdown(f'<span class="{badge}">{badge_txt}</span>', unsafe_allow_html=True)
        st.subheader("📝 Your Email Draft")

        rev = st.session_state.get("email_rev", 0)
        draft_val = result.get("text", "")

        edited_email = st.text_area(
            "Draft Content (You can edit directly):",
            value=draft_val,
            height=340,
            key=f"email_editor_{rev}"
        )
        st.session_state.current_email["text"] = edited_email

        _refinement_buttons("current_email", edited_email, "email_rev", active_api_key, selected_model, "em")

        st.markdown("---")
        d1, d2 = st.columns([1, 4])
        with d1:
            st.download_button(
                "📥 Download .txt", data=edited_email,
                file_name=f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", use_container_width=True
            )
        with d2:
            _show_rag_expander(result, "email_rag")


# ==============================================================================
# PAGE 2: Write a Letter
# ==============================================================================
elif page == "📜 Write a Letter":
    st.markdown('<div class="main-header">📜 Write a Formal Letter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Draft structured formal letters grounded in standard letter-writing conventions.</div>', unsafe_allow_html=True)

    with st.form("letter_form"):
        col1, col2 = st.columns(2)
        with col1:
            letter_type = st.selectbox("Letter Type *", [
                "Leave letter", "Formal request letter", "Permission letter",
                "Job application letter", "Internship application letter",
                "Complaint letter", "Apology letter",
                "Recommendation request letter", "General formal letter"
            ])
            sender_details = st.text_area(
                "Sender Details (Name, Designation, Address/Contact) *",
                placeholder="e.g.,\nAlex Taylor\nSoftware Engineer\nTech Solutions Inc.\nalex@example.com",
                height=100
            )
            recipient_details = st.text_area(
                "Recipient Details (Name, Designation, Organization/Address) *",
                placeholder="e.g.,\nDr. Marcus Vance\nDean of Academic Affairs\nUniversity Institute",
                height=100
            )
            date_str = st.text_input("Date", value=datetime.now().strftime("%B %d, %Y"))

        with col2:
            subject = st.text_input("Subject *", placeholder="e.g., Request for Semester On-Duty Permission")
            tone = st.selectbox("Tone", ["Formal", "Respectful", "Firm", "Polite"])
            length = st.selectbox("Length", ["Detailed", "Standard", "Concise"], index=0)
            details = st.text_area(
                "Letter Purpose & Specific Details *",
                placeholder="Explain the background, reason, dates, attached documents, or specific requests...",
                height=170
            )

        submit_letter = st.form_submit_button("🚀 Generate Letter", use_container_width=True, type="primary")

    if submit_letter:
        missing = [f for f, v in [
            ("Sender Details", sender_details), ("Recipient Details", recipient_details),
            ("Subject", subject), ("Details", details)
        ] if not v.strip()]
        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            reqs = {
                "Letter Type": letter_type, "Sender": sender_details.strip(),
                "Recipient": recipient_details.strip(), "Date": date_str.strip(),
                "Subject": subject.strip(), "Details": details.strip(),
                "Tone": tone, "Length": length
            }
            with st.spinner("Retrieving guidelines and drafting formal letter..."):
                try:
                    result = generate_writing(
                        doc_type="formal letter", requirements=reqs,
                        provider="groq" if has_api_key else "offline",
                        api_key=active_api_key, model=selected_model,
                        top_k=4, similarity_threshold=0.25
                    )
                    st.session_state.current_letter = result
                    st.session_state["letter_rev"] = st.session_state.get("letter_rev", 0) + 1
                    _save_to_history("formal letter", reqs, result)
                    st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "invalid_api_key" in err_msg:
                        st.error("❌ **Invalid Groq API Key (401)**. "
                                 "Get a free key at [console.groq.com/keys](https://console.groq.com/keys).")
                    else:
                        st.error(f"Error generating letter: {e}")

    if st.session_state.current_letter:
        st.markdown("---")
        result = st.session_state.current_letter
        grounded = result.get("is_grounded", False)
        badge = "badge-grounded" if grounded else "badge-offline"
        badge_txt = "✅ RAG Grounded" if grounded else "⚡ Standard Draft"
        st.markdown(f'<span class="{badge}">{badge_txt}</span>', unsafe_allow_html=True)
        st.subheader("📝 Your Letter Draft")

        rev = st.session_state.get("letter_rev", 0)
        draft_val = result.get("text", "")

        edited_letter = st.text_area(
            "Draft Content (You can edit directly):",
            value=draft_val,
            height=420,
            key=f"letter_editor_{rev}"
        )
        st.session_state.current_letter["text"] = edited_letter

        _refinement_buttons("current_letter", edited_letter, "letter_rev", active_api_key, selected_model, "let")

        st.markdown("---")
        d1, d2 = st.columns([1, 4])
        with d1:
            st.download_button(
                "📥 Download .txt", data=edited_letter,
                file_name=f"letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", use_container_width=True
            )
        with d2:
            _show_rag_expander(result, "letter_rag")


# ==============================================================================
# PAGE 3: Knowledge Base
# ==============================================================================
elif page == "📚 Knowledge Base":
    st.markdown('<div class="main-header">📚 Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Manage your reference documents powering RAG-grounded drafts.</div>', unsafe_allow_html=True)

    collection = get_collection()
    kb_tab1, kb_tab2, kb_tab3 = st.tabs(["📂 Stored Documents", "📤 Upload Documents", "🔎 Vector Search Explorer"])

    # ── TAB 1: Stored Documents ───────────────────────────────────────────────
    with kb_tab1:
        sources = get_all_sources(collection)
        total = collection.count()

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.metric("Total Chunks Indexed", total)
        with col_b:
            if st.button("⚡ Load Built-in Reference Guidelines", use_container_width=True):
                with st.spinner("Seeding sample reference guides..."):
                    n = seed_sample_docs_if_empty("sample_docs")
                st.success(f"Knowledge base now has {n} chunks.")
                st.rerun()

        if not sources:
            st.info("No documents indexed yet. Upload files or load built-in reference guidelines.")
        else:
            st.markdown(f"**{len(sources)} source(s) indexed:**")
            for src in sources:
                src_data = collection.get(where={"source": src}, include=["metadatas"])
                chunk_cnt = len(src_data.get("ids", []))
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"📄 `{src}` &nbsp;&nbsp; <small>{chunk_cnt} chunks</small>", unsafe_allow_html=True)
                with c2:
                    if st.button("🗑️ Delete", key=f"del_{src}", use_container_width=True):
                        n = delete_document(collection, src)
                        st.success(f"Deleted {n} chunks from `{src}`.")
                        st.rerun()

        st.markdown("---")
        if total > 0:
            if st.button("💣 Reset Entire Knowledge Base", type="secondary"):
                all_ids = collection.get()["ids"]
                if all_ids:
                    collection.delete(ids=all_ids)
                st.success("Knowledge base cleared.")
                st.rerun()

    # ── TAB 2: Upload Documents ────────────────────────────────────────────────
    with kb_tab2:
        st.markdown("Upload `.txt` or `.pdf` files to add them to the knowledge base.")

        uploaded = st.file_uploader(
            "Choose files", type=["txt", "pdf"],
            accept_multiple_files=True, key="kb_upload"
        )
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        with u_col2:
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 10)

        if st.button("📥 Index Uploaded Files", type="primary", disabled=not uploaded):
            progress = st.progress(0, text="Starting...")
            errors = []
            total_added = 0
            for i, uf in enumerate(uploaded):
                progress.progress((i) / len(uploaded), text=f"Processing: {uf.name}")
                ext = Path(uf.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name
                try:
                    doc = load_document(tmp_path)
                    chunks, metas, ids = chunk_text(
                        doc["text"], uf.name,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    collection.upsert(documents=chunks, metadatas=metas, ids=ids)
                    total_added += len(chunks)
                except Exception as e:
                    errors.append(f"❌ `{uf.name}`: {e}")
                finally:
                    os.remove(tmp_path)

            progress.empty()
            if total_added:
                st.success(f"✅ Indexed {len(uploaded)} file(s) → {total_added} chunks added.")
                st.rerun()
            for err in errors:
                st.error(err)

    # ── TAB 3: Vector Search Explorer ─────────────────────────────────────────
    with kb_tab3:
        st.markdown("Test semantic retrieval from your knowledge base interactively.")
        search_q = st.text_input("Enter a test query:", placeholder="e.g., leave handover protocol")
        top_k_search = st.slider("Top-K results", 1, 10, 4)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.25, 0.05)

        if st.button("🔍 Search", type="primary", disabled=not search_q.strip()):
            with st.spinner("Searching..."):
                results = retrieve_context(
                    query=search_q, collection=collection,
                    n_results=top_k_search, similarity_threshold=threshold
                )
            if not results:
                st.warning("No chunks above similarity threshold. Try lowering the threshold or indexing more documents.")
            else:
                st.success(f"Found {len(results)} matching chunk(s):")
                for i, r in enumerate(results, 1):
                    with st.expander(
                        f"#{i} · `{r['source']}` · chunk {r['chunk_index']} · 🎯 {r['similarity']}",
                        expanded=(i == 1)
                    ):
                        st.text(r["text"])


# ==============================================================================
# PAGE 4: Writing History
# ==============================================================================
elif page == "🕒 Writing History":
    st.markdown('<div class="main-header">🕒 Writing History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">All your generated drafts, saved locally.</div>', unsafe_allow_html=True)

    records = list_history()
    count = len(records)

    if count == 0:
        st.info("No saved drafts yet. Generate an email or letter to see it here.")
    else:
        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.metric("Total Saved Drafts", count)
        with h_col2:
            if st.button("🗑️ Clear All History", type="secondary"):
                n = clear_all_history()
                st.success(f"Deleted {n} drafts.")
                st.rerun()

        st.markdown("---")
        for rec in records:
            fname = rec.get("_filename", "")
            doc_type = rec.get("doc_type", "draft").title()
            subject = rec.get("subject", "—")
            recipient = rec.get("recipient", "—")
            ts_display = rec.get("timestamp_display", rec.get("timestamp", "")[:16])
            grounded = rec.get("is_grounded", False)
            chunk_count = rec.get("chunk_count", 0)

            badge_txt = f"✅ Grounded ({chunk_count} refs)" if grounded else "⚡ Standard"

            with st.expander(f"**{doc_type}** · {subject} → {recipient} · {ts_display} · {badge_txt}"):
                draft_text = rec.get("draft_text", "")
                st.text_area("Draft:", value=draft_text, height=260,
                             key=f"hist_{fname}", disabled=True)

                hc1, hc2, hc3 = st.columns(3)
                with hc1:
                    st.download_button(
                        "📥 Download .txt",
                        data=draft_text,
                        file_name=f"{fname.replace('.json', '.txt')}",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"dl_{fname}"
                    )
                with hc2:
                    if doc_type.lower() == "email":
                        if st.button("✏️ Reopen in Editor", key=f"open_{fname}", use_container_width=True):
                            st.session_state.current_email = {"text": draft_text, "retrieved_chunks": [], "is_grounded": grounded}
                            st.session_state["email_rev"] = st.session_state.get("email_rev", 0) + 1
                            st.session_state.nav_page = "✉️ Write an Email"
                            st.rerun()
                    else:
                        if st.button("✏️ Reopen in Editor", key=f"open_{fname}", use_container_width=True):
                            st.session_state.current_letter = {"text": draft_text, "retrieved_chunks": [], "is_grounded": grounded}
                            st.session_state["letter_rev"] = st.session_state.get("letter_rev", 0) + 1
                            st.session_state.nav_page = "📜 Write a Letter"
                            st.rerun()
                with hc3:
                    if st.button("🗑️ Delete", key=f"del_h_{fname}", use_container_width=True):
                        delete_draft(fname)
                        st.rerun()
