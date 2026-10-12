"""
Step 4: Passing Context to LLM (Groq / Gemini)
Based on: RAG_Step4_Passing_Context_to_Gemini_Groq_LLM.ipynb

This module shows how to:
1. Define a strict, grounded SYSTEM_PROMPT.
2. Build the user prompt combining retrieved context and user requirements.
3. Call Groq (with model fallback: qwen/qwen3.8-27b, allam-2-7b, openai/gpt-oss-120b).
4. Call Google Gemini (with model fallback: gemini-2.5-flash).
5. Provide a fallback offline generator if no API key is provided.
6. Provide a unified `generate_writing()` function.
"""

import os
from dotenv import load_dotenv
from step3_retrieval import retrieve_context, format_context_for_llm

load_dotenv()

from config import (
    GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
    GEMINI_MODELS,
    DEFAULT_GEMINI_MODEL
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_PROMPT = (
    "You are an expert executive email and formal letter writing assistant.\n\n"
    "Your objective is to craft comprehensive, high-quality, professional drafts that are rich in context, "
    "well-articulated, and thoroughly developed. Avoid terse, sparse, or generic one-line responses.\n\n"
    "Document Structure & Elaboration Standards:\n"
    "- Provide clear, descriptive subject lines or complete formal letterheads with date, addresses, and salutation.\n"
    "- Develop a polished introductory paragraph stating the background context, purpose, and timeline clearly.\n"
    "- Provide well-reasoned, multi-paragraph body text detailing the core message, specific circumstances, impact, "
    "and necessary arrangements (e.g. coverage handovers, deliverables, milestones, or attachments).\n"
    "- Use bullet points or formatted lists where enumerating key dates, deliverables, or items improves readability.\n"
    "- Conclude with a definite Call to Action (CTA), clear next steps, availability for follow-up, and an executive sign-off block.\n\n"
    "Groundedness and Integrity:\n"
    "- Use retrieved reference passages as guidance for appropriate phrasing, structure, and formal conventions.\n"
    "- Adapt reference guidelines naturally to the user's specific context; do not copy unrelated examples blindly.\n"
    "- Never invent verifiable personal facts, company policies, or fake accomplishments. "
    "If necessary information is missing, utilize standard brackets like [Contact Phone] or [Department Name].\n\n"
    "Output Rule: Return ONLY the final formatted document draft (including Subject/Header and Signature). "
    "Do not include conversational greetings, preambles, or conversational commentary outside the draft itself."
)


def build_user_prompt(doc_type, requirements, context_chunks):
    """
    Format the user prompt with context, detailed elaboration rules, and requirements.
    """
    has_context = len(context_chunks) > 0
    context_text = format_context_for_llm(context_chunks)

    req_lines = [f"- {k}: {v}" for k, v in requirements.items() if v]
    requirements_block = "\n".join(req_lines)

    length_pref = str(requirements.get("Length", "Detailed")).strip().lower()

    if "concise" in length_pref:
        depth_guide = (
            "LENGTH REQUIREMENT: Concise & Direct.\n"
            "- Keep the draft brief, focused, and crisp (1-2 compact paragraphs) while maintaining complete professional etiquette."
        )
    elif "standard" in length_pref:
        depth_guide = (
            "LENGTH REQUIREMENT: Standard Professional Length.\n"
            "- Provide a balanced, well-developed draft (2-3 paragraphs) explaining the situation clearly with professional courtesy."
        )
    else:  # Default to Detailed
        depth_guide = (
            "LENGTH REQUIREMENT: Detailed & Comprehensive (Extensive).\n"
            "- Produce an in-depth, thorough, and highly articulate draft (3-5 well-developed paragraphs).\n"
            "- Elaborate extensively on background context, justification/reasons, specific timeline, coverage/transition details, and anticipated outcomes.\n"
            "- Break complex items into structured bullet points or sub-sections where appropriate.\n"
            "- Provide a clear, actionable next step and comprehensive closing block."
        )

    prompt = f"""DOCUMENT TYPE: {doc_type.upper()}

REFERENCE CONTEXT:
\"\"\"
{context_text}
\"\"\"

USER REQUIREMENTS:
{requirements_block}

{depth_guide}

INSTRUCTIONS:
1. Generate the complete, polished {doc_type} fulfilling all stated requirements and depth guidelines.
2. Ground structure, tone, and conventions in the reference context where applicable.
3. Make the draft articulate, mature, and thoroughly fleshed out with complete explanations.
4. Output ONLY the complete draft text directly without conversational intro or outro."""

    return prompt


def ask_groq(user_prompt, api_key=None, model=None):
    """Call Groq chat completion API with model fallback."""
    key = api_key or GROQ_API_KEY
    if not key:
        return None

    from groq import Groq
    client = Groq(api_key=key)

    if model:
        models_to_try = [model] + [m for m in GROQ_MODELS if m != model]
    else:
        models_to_try = GROQ_MODELS
    last_error = None

    for m in models_to_try:
        if not m:
            continue
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Groq API error: {last_error}")


def ask_gemini(user_prompt, api_key=None, model=None):
    """Call Google Gemini API with model fallback."""
    key = api_key or GEMINI_API_KEY
    if not key:
        return None

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)

    models_to_try = [model] if model else GEMINI_MODELS
    last_error = None

    for m in models_to_try:
        if not m:
            continue
        try:
            response = client.models.generate_content(
                model=m,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=3000
                )
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Gemini API error: {last_error}")


def ask_offline_fallback(doc_type, requirements):
    """
    Simple rule-based fallback draft generator when no API key is provided.
    Allows testing the whole flow offline.
    """
    recipient = requirements.get("Recipient", "[Recipient Name]")
    sender = requirements.get("Sender", "[Your Name]")
    subject = requirements.get("Subject", "Professional Correspondence")
    details = requirements.get("Details", "the discussed matter")
    date_val = requirements.get("Date", "")

    if doc_type.lower() == "email":
        date_header = f"Date: {date_val}\n" if date_val else ""
        return f"""Subject: {subject}
{date_header}
Dear {recipient},

I hope this email finds you well.

I am writing regarding {subject}. Specifically, {details}.

Please let me know if any further information is needed. Thank you for your time.

Best regards,

{sender}"""
    else:
        return f"""From:
{sender}

Date: [Current Date]

To:
{recipient}

Subject: {subject}

Respected Sir/Madam,

I am writing to formally request your attention regarding {subject}.

{details}

I kindly request your favorable consideration on this matter.

Yours sincerely,

{sender}"""


def generate_writing(
    doc_type,
    requirements,
    provider="groq",
    api_key=None,
    model=None,
    top_k=4,
    similarity_threshold=0.25,
    source_filter=None
):
    """
    Complete RAG generation pipeline for emails or letters:
    1. Search vector DB for relevant chunks.
    2. Build grounded prompt.
    3. Call selected LLM (or offline fallback).
    4. Return generated draft and retrieved sources.
    """
    # 1. Search ChromaDB
    query = f"{requirements.get('Purpose', '')} {requirements.get('Subject', '')} {requirements.get('Details', '')}".strip()
    chunks = retrieve_context(
        query=query,
        n_results=top_k,
        similarity_threshold=similarity_threshold,
        source_filter=source_filter
    )

    # 2. Build prompt
    user_prompt = build_user_prompt(doc_type, requirements, chunks)

    # 3. Call LLM
    answer = None
    if provider == "gemini":
        answer = ask_gemini(user_prompt, api_key=api_key, model=model or DEFAULT_GEMINI_MODEL)
    elif provider == "groq":
        answer = ask_groq(user_prompt, api_key=api_key, model=model or DEFAULT_GROQ_MODEL)

    # If no API key or provider is offline, use offline fallback
    if not answer:
        answer = ask_offline_fallback(doc_type, requirements)

    return {
        "text": answer,
        "retrieved_chunks": chunks,
        "is_grounded": len(chunks) > 0,
        "prompt_used": user_prompt
    }


def refine_draft(current_draft, instruction, provider="groq", api_key=None, model=None):
    """
    Refine an existing draft (email or letter) based on a specific instruction.
    e.g. 'Make it more formal', 'Make it concise', 'Fix grammar'.
    """
    prompt = f"""You are a professional editor. Revise the following draft according to this instruction:
INSTRUCTION: {instruction}

ORIGINAL DRAFT:
\"\"\"
{current_draft}
\"\"\"

RULES:
- Return ONLY the revised draft.
- Maintain any existing placeholders like names, dates, or contact details.
- Do not add conversational intro or outro."""

    key = api_key or os.getenv("GROQ_API_KEY", "")
    if provider == "groq" and key:
        try:
            revised = ask_groq(prompt, api_key=key, model=model or DEFAULT_GROQ_MODEL)
            if revised:
                return revised
        except Exception:
            pass

    return current_draft


# Quick test when running directly
if __name__ == "__main__":
    sample_requirements = {
        "Purpose": "Leave request",
        "Recipient": "Dr. Sarah Jenkins",
        "Sender": "Alex Taylor",
        "Subject": "Leave Application: Nov 12-14",
        "Details": "Personal family commitment. Colleague John will cover escalations."
    }

    print("Testing generate_writing (offline mode)...")
    res = generate_writing("email", sample_requirements, provider="offline")
    print("\nResult Draft:")
    print(res["text"])
    print(f"\nGrounded in Knowledge Base: {res['is_grounded']} ({len(res['retrieved_chunks'])} chunks)")
