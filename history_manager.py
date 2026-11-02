"""
History Manager for WriteWise AI
Saves, loads, lists, and deletes draft history records as JSON files in data/history/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from config import HISTORY_DIR


def _history_dir() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORY_DIR


def save_draft(doc_type: str, requirements: dict, draft_text: str, is_grounded: bool, chunk_count: int) -> str:
    """
    Save a generated draft to local history.

    Returns:
        Path to the saved history JSON file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subject = (
        requirements.get("Subject", requirements.get("Letter Type", "draft"))
        .replace(" ", "_")
        .replace("/", "-")[:40]
    )
    filename = f"{timestamp}_{doc_type.replace(' ', '_')}_{safe_subject}.json"
    filepath = _history_dir() / filename

    record = {
        "id": timestamp,
        "doc_type": doc_type,
        "timestamp": datetime.now().isoformat(),
        "timestamp_display": datetime.now().strftime("%b %d, %Y %I:%M %p"),
        "requirements": requirements,
        "draft_text": draft_text,
        "is_grounded": is_grounded,
        "chunk_count": chunk_count,
        "subject": requirements.get("Subject", requirements.get("Letter Type", "—")),
        "recipient": requirements.get("Recipient", requirements.get("Recipient", "—")),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return str(filepath)


def list_history(limit: int = 50) -> list:
    """
    List saved drafts sorted by newest first.

    Returns:
        List of history record dicts.
    """
    hdir = _history_dir()
    records = []
    for f in sorted(hdir.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                rec = json.load(fp)
                rec["_filename"] = f.name
                records.append(rec)
        except Exception:
            pass
    return records


def load_draft(filename: str) -> dict:
    """Load a specific history record by filename."""
    filepath = _history_dir() / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_draft(filename: str) -> bool:
    """Delete a specific history record. Returns True if deleted."""
    filepath = _history_dir() / filename
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def clear_all_history() -> int:
    """Delete all history records. Returns count deleted."""
    hdir = _history_dir()
    count = 0
    for f in hdir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def history_count() -> int:
    """Return count of saved drafts."""
    return len(list(_history_dir().glob("*.json")))

