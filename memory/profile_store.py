"""
Mid-term memory: candidate profile persistence across sessions.

ProfileStore keeps candidate profiles in a SQLite DB (database/profiles.db) keyed
by an MD5 hash of the uploaded resume, so a returning candidate is recognised
even after closing and reopening the app. Stores their parsed resume, analysis
history, and the follow-up questions/answers they've given.

Usage:
    from memory.profile_store import ProfileStore
    store = ProfileStore()
    h = store.hash_resume(pdf_bytes)
    profile = store.find_by_hash(h) or {"session_id": store.create_profile(...)}
"""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# On HuggingFace Spaces, SPACE_ID is set; persist profiles.db in /data/ so
# candidate data survives container restarts.
_IS_HF_SPACE = bool(os.getenv("SPACE_ID"))
_DEFAULT_DB = (
    Path("/data/profiles.db") if _IS_HF_SPACE
    else Path(__file__).resolve().parents[1] / "database" / "profiles.db"
)

_CREATE = """
CREATE TABLE IF NOT EXISTS candidate_profiles (
    session_id          TEXT PRIMARY KEY,
    candidate_name      TEXT,
    resume_hash         TEXT,
    parsed_resume_json  TEXT,
    analyses_history    TEXT,
    followup_history    TEXT,
    created_at          TEXT,
    last_accessed       TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProfileStore:
    """Persistent candidate profiles backed by SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or _DEFAULT_DB)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE)

    def _conn(self) -> sqlite3.Connection:
        # New connection per operation — safe across Streamlit reruns/threads.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Identity ─────────────────────────────────────────────────────────────

    @staticmethod
    def hash_resume(pdf_bytes: bytes) -> str:
        """MD5 hash of the raw PDF bytes, used to recognise the same resume."""
        return hashlib.md5(pdf_bytes).hexdigest()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "session_id": row["session_id"],
            "candidate_name": row["candidate_name"],
            "resume_hash": row["resume_hash"],
            "parsed_resume": json.loads(row["parsed_resume_json"] or "{}"),
            "analyses_history": json.loads(row["analyses_history"] or "[]"),
            "followup_history": json.loads(row["followup_history"] or "[]"),
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
        }

    def find_by_hash(self, resume_hash: str) -> Optional[dict]:
        """Return an existing profile for this resume hash, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_profiles WHERE resume_hash = ? "
                "ORDER BY last_accessed DESC LIMIT 1",
                (resume_hash,),
            ).fetchone()
            if row is None:
                return None
            # Touch last_accessed.
            conn.execute(
                "UPDATE candidate_profiles SET last_accessed = ? "
                "WHERE session_id = ?",
                (_now(), row["session_id"]),
            )
            return self._row_to_dict(row)

    def create_profile(self, resume_hash: str, parsed_resume: dict,
                       name: str) -> str:
        """Create a new profile and return its session_id (uuid)."""
        session_id = str(uuid.uuid4())
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO candidate_profiles VALUES (?,?,?,?,?,?,?,?)",
                (session_id, name, resume_hash,
                 json.dumps(parsed_resume, ensure_ascii=False),
                 "[]", "[]", now, now),
            )
        return session_id

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_field(self, session_id: str, field: str) -> list:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {field} FROM candidate_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return []
        return json.loads(row[0])

    def _append_field(self, session_id: str, field: str, item: dict) -> None:
        items = self._get_field(session_id, field)
        items.append(item)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE candidate_profiles SET {field} = ?, last_accessed = ? "
                "WHERE session_id = ?",
                (json.dumps(items, ensure_ascii=False), _now(), session_id),
            )

    # ── Writes ───────────────────────────────────────────────────────────────

    def save_analysis(self, session_id: str, job_title: str, company: str,
                      fit_score: int, recommendation: str,
                      synthesis_dict: dict) -> None:
        """Append an analysis to the profile's analyses_history."""
        self._append_field(session_id, "analyses_history", {
            "timestamp": _now(),
            "job_title": job_title,
            "company": company,
            "fit_score": int(fit_score),
            "recommendation": recommendation,
            "synthesis": synthesis_dict,
        })

    def save_followup_qa(self, session_id: str, question: str, answer: str,
                         gap_addressed: str) -> None:
        """Append a follow-up question/answer to followup_history."""
        self._append_field(session_id, "followup_history", {
            "timestamp": _now(),
            "question": question,
            "answer": answer,
            "gap_addressed": gap_addressed,
        })

    # ── Reads ────────────────────────────────────────────────────────────────

    def get_asked_questions(self, session_id: str) -> list:
        """All follow-up questions previously asked this candidate."""
        return [qa.get("question", "")
                for qa in self._get_field(session_id, "followup_history")
                if qa.get("question")]

    def get_followup_history(self, session_id: str) -> list:
        """All previous follow-up QA entries (question, answer, gap_addressed)."""
        return self._get_field(session_id, "followup_history")

    def get_analysis_history(self, session_id: str) -> list:
        """All past analyses, sorted oldest -> newest by timestamp."""
        history = self._get_field(session_id, "analyses_history")
        return sorted(history, key=lambda a: a.get("timestamp", ""))

    def get_best_match(self, session_id: str) -> Optional[dict]:
        """The past analysis with the highest fit_score, or None."""
        history = self._get_field(session_id, "analyses_history")
        if not history:
            return None
        return max(history, key=lambda a: a.get("fit_score", 0))
