"""
WriteWise AI — Email and Letter Writing Assistant
Powered by RAG (ChromaDB + Sentence Transformers + Groq LLM).
API key is strictly loaded from the .env file.
"""

import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

import sys
from pathlib import Path

# Ensure application directory is always in sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Load environment variables from .env
load_dotenv()

from step2_chromadb_storage import seed_sample_docs_if_empty
from step4_llm_generation import generate_writing, refine_draft

# Ensure reference guidelines are seeded into ChromaDB on startup
@st.cache_resource(show_spinner=False)
def init_knowledge_base():
    try:
        return seed_sample_docs_if_empty("sample_docs")
    except Exception:
        return 0

init_knowledge_base()

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="WriteWise — Email & Letter Assistant",
    page_icon="✍️",
    layout="wide"
)

# Custom Styling for clean, modern look
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
        font-family: inherit;
        font-size: 0.95rem;
    }
    .stButton button {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">✍️ WriteWise AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG-Powered Professional Email & Formal Letter Assistant</div>', unsafe_allow_html=True)

# API Key check from environment
has_api_key = bool(os.getenv("GROQ_API_KEY", "").strip())
if not has_api_key:
    st.warning("⚠️ No `GROQ_API_KEY` found in your `.env` file. Please set `GROQ_API_KEY` in `.env` to enable AI generation.")

# Initialize Session State
if "current_email" not in st.session_state:
    st.session_state.current_email = None

if "current_letter" not in st.session_state:
    st.session_state.current_letter = None


# Two primary tabs: Email and Letter
tab_email, tab_letter = st.tabs(["✉️ Write an Email", "📜 Write a Letter"])


# ==============================================================================
# 1. EMAIL TAB
# ==============================================================================
with tab_email:
    with st.form("email_form"):
        col1, col2 = st.columns(2)

        with col1:
            purpose = st.selectbox(
                "Email Purpose / Category *",
                [
                    "Leave request",
                    "Job application",
                    "Meeting request",
                    "Follow-up email",
                    "Thank-you email",
                    "Apology email",
                    "Complaint / Issue escalation",
                    "Permission request",
                    "Academic communication",
                    "General professional email"
                ]
            )
            recipient_name = st.text_input("Recipient Name / Designation *", placeholder="e.g., Dr. Sarah Jenkins / Hiring Manager")
            sender_name = st.text_input("Sender Name *", placeholder="e.g., Alex Taylor")
            subject = st.text_input("Subject or Topic *", placeholder="e.g., Sick Leave Application - Oct 14 / Project Update")

        with col2:
            tone = st.selectbox("Tone", ["Professional", "Formal", "Polite", "Friendly", "Urgent"])
            length = st.selectbox("Length", ["Standard", "Concise", "Detailed"])
            details = st.text_area(
                "Important Details / Main Message *",
                placeholder="State reason, dates, context, handovers, contact details...",
                height=130
            )

        submit_email = st.form_submit_button("🚀 Generate Email", use_container_width=True)

    if submit_email:
        if not recipient_name.strip() or not sender_name.strip() or not subject.strip() or not details.strip():
            st.error("Please fill in all required fields marked with *.")
        else:
            reqs = {
                "Purpose": purpose,
                "Recipient": recipient_name.strip(),
                "Sender": sender_name.strip(),
                "Subject": subject.strip(),
                "Details": details.strip(),
                "Tone": tone,
                "Length": length
            }

            with st.spinner("Retrieving guidelines and drafting email with AI..."):
                try:
                    result = generate_writing(
                        doc_type="email",
                        requirements=reqs,
                        provider="groq" if has_api_key else "offline",
                        top_k=4,
                        similarity_threshold=0.25
                    )
                    st.session_state.current_email = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating email: {e}")

    # Output Section
    if st.session_state.current_email:
        st.markdown("---")
        st.subheader("📝 Your Email Draft")

        draft_content = st.session_state.current_email.get("text", "")
        edited_email = st.text_area(
            "Draft Content (You can edit directly):",
            value=draft_content,
            height=320,
            key="email_edit_box"
        )

        # Refinement Actions
        st.markdown("**⚡ Refine Draft:**")
        ref_c1, ref_c2, ref_c3, ref_c4 = st.columns(4)

        with ref_c1:
            if st.button("👔 More Formal", key="btn_em_formal", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_email, "Make this email draft more formal, polite, and professional.")
                    st.session_state.current_email["text"] = revised
                    st.rerun()

        with ref_c2:
            if st.button("✂️ More Concise", key="btn_em_concise", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_email, "Make this email draft concise, brief, and direct while preserving all essential details.")
                    st.session_state.current_email["text"] = revised
                    st.rerun()

        with ref_c3:
            if st.button("✨ Fix Grammar", key="btn_em_grammar", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_email, "Correct all grammar, spelling, punctuation, and phrasing issues while keeping the meaning intact.")
                    st.session_state.current_email["text"] = revised
                    st.rerun()

        with ref_c4:
            if st.button("🗑️ Clear Draft", key="btn_em_clear", use_container_width=True):
                st.session_state.current_email = None
                st.rerun()

        # Download & Copy
        st.markdown("---")
        d_col1, d_col2 = st.columns([1, 4])
        with d_col1:
            st.download_button(
                label="📥 Download as TXT",
                data=edited_email,
                file_name=f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with d_col2:
            with st.expander("📚 View Reference Guidelines Used (RAG)"):
                chunks = st.session_state.current_email.get("retrieved_chunks", [])
                if chunks:
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Reference #{i}** (Source: `{c['source']}`, Similarity: `{c['similarity']}`)")
                        st.caption(c["text"])
                else:
                    st.info("No specific reference chunks met the similarity threshold; drafted using standard best practices.")


# ==============================================================================
# 2. LETTER TAB
# ==============================================================================
with tab_letter:
    with st.form("letter_form"):
        col1, col2 = st.columns(2)

        with col1:
            letter_type = st.selectbox(
                "Letter Type *",
                [
                    "Leave letter",
                    "Formal request letter",
                    "Permission letter",
                    "Job application letter",
                    "Internship application letter",
                    "Complaint letter",
                    "Apology letter",
                    "Recommendation request letter",
                    "General formal letter"
                ]
            )
            sender_details = st.text_area(
                "Sender Details (Name, Designation, Address/Contact) *",
                placeholder="e.g.,\nAlex Taylor\nSoftware Engineer\nTech Solutions Inc.\nalex@example.com",
                height=95
            )
            recipient_details = st.text_area(
                "Recipient Details (Name, Designation, Organization/Address) *",
                placeholder="e.g.,\nDr. Marcus Vance\nDean of Academic Affairs\nUniversity Institute",
                height=95
            )
            date_str = st.text_input("Date", value=datetime.now().strftime("%B %d, %Y"))

        with col2:
            subject = st.text_input("Subject *", placeholder="e.g., Request for Semester On-Duty Permission")
            tone = st.selectbox("Tone", ["Formal", "Respectful", "Firm", "Polite"])
            length = st.selectbox("Length", ["Standard", "Concise", "Detailed"])
            details = st.text_area(
                "Letter Purpose & Specific Details *",
                placeholder="Explain the background, reason, dates, attached documents, or specific requests...",
                height=160
            )

        submit_letter = st.form_submit_button("🚀 Generate Letter", use_container_width=True)

    if submit_letter:
        if not sender_details.strip() or not recipient_details.strip() or not subject.strip() or not details.strip():
            st.error("Please fill in all required fields marked with *.")
        else:
            reqs = {
                "Letter Type": letter_type,
                "Sender": sender_details.strip(),
                "Recipient": recipient_details.strip(),
                "Date": date_str.strip(),
                "Subject": subject.strip(),
                "Details": details.strip(),
                "Tone": tone,
                "Length": length
            }

            with st.spinner("Retrieving guidelines and drafting formal letter with AI..."):
                try:
                    result = generate_writing(
                        doc_type="formal letter",
                        requirements=reqs,
                        provider="groq" if has_api_key else "offline",
                        top_k=4,
                        similarity_threshold=0.25
                    )
                    st.session_state.current_letter = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating letter: {e}")

    # Output Section
    if st.session_state.current_letter:
        st.markdown("---")
        st.subheader("📝 Your Letter Draft")

        draft_content = st.session_state.current_letter.get("text", "")
        edited_letter = st.text_area(
            "Draft Content (You can edit directly):",
            value=draft_content,
            height=380,
            key="letter_edit_box"
        )

        # Refinement Actions
        st.markdown("**⚡ Refine Draft:**")
        ref_c1, ref_c2, ref_c3, ref_c4 = st.columns(4)

        with ref_c1:
            if st.button("👔 More Formal", key="btn_let_formal", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_letter, "Increase formal protocol, polite deference, and official terminology.")
                    st.session_state.current_letter["text"] = revised
                    st.rerun()

        with ref_c2:
            if st.button("✂️ More Concise", key="btn_let_concise", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_letter, "Make this letter concise, direct, and brief while maintaining formal structure.")
                    st.session_state.current_letter["text"] = revised
                    st.rerun()

        with ref_c3:
            if st.button("✨ Fix Grammar", key="btn_let_grammar", use_container_width=True):
                with st.spinner("Refining..."):
                    revised = refine_draft(edited_letter, "Correct grammatical structure, punctuation, sentence pacing, and formal salutations.")
                    st.session_state.current_letter["text"] = revised
                    st.rerun()

        with ref_c4:
            if st.button("🗑️ Clear Draft", key="btn_let_clear", use_container_width=True):
                st.session_state.current_letter = None
                st.rerun()

        # Download & Copy
        st.markdown("---")
        d_col1, d_col2 = st.columns([1, 4])
        with d_col1:
            st.download_button(
                label="📥 Download as TXT",
                data=edited_letter,
                file_name=f"letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with d_col2:
            with st.expander("📚 View Reference Guidelines Used (RAG)"):
                chunks = st.session_state.current_letter.get("retrieved_chunks", [])
                if chunks:
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**Reference #{i}** (Source: `{c['source']}`, Similarity: `{c['similarity']}`)")
                        st.caption(c["text"])
                else:
                    st.info("No specific reference chunks met the similarity threshold; drafted using standard best practices.")
