"""
AI Hiring Co-pilot — Streamlit UI (redesigned).

Candidate mode: upload a resume, pick/paste a JD, get an evidence-based fit
analysis, answer follow-up questions to refine the score, and chat about the
result. Recruiter mode: batch-analyze several resumes against one role.

Run:  streamlit run app.py
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Make the project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env for local development.
load_dotenv(_PROJECT_ROOT / ".env")

# ── HuggingFace Spaces detection ─────────────────────────────────────────────
IS_HF_SPACE = bool(os.getenv("SPACE_ID"))

# ── API key — read from env ONLY (no UI input) ───────────────────────────────
_ENV_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
IS_DEMO = not bool(_ENV_API_KEY)

# Pre-load the demo result.
_DEMO_RESULT_PATH = _PROJECT_ROOT / "data" / "demo_result.json"
_DEMO_RESULT: dict = {}
if _DEMO_RESULT_PATH.exists():
    try:
        with open(_DEMO_RESULT_PATH, encoding="utf-8") as _f:
            _DEMO_RESULT = json.load(_f)
    except Exception:
        _DEMO_RESULT = {}

_DB_PATH = _PROJECT_ROOT / "database" / "jobs.db"
_MAX_MB = 5

st.set_page_config(
    page_title="AI Hiring Co-pilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base typography ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ─── */
section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] .stMarkdown h2 { font-size: 1rem; }

/* ── Score card ─── */
.score-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ── Strength / gap item cards ─── */
.item-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left-width: 4px;
}
.item-card.strength { border-left-color: #22c55e; }
.item-card.critical { border-left-color: #ef4444; }
.item-card.moderate { border-left-color: #f59e0b; }
.item-card.minor    { border-left-color: #94a3b8; }

/* ── Progress stepper ─── */
.step-row { display:flex; gap:8px; align-items:center; margin-bottom:12px; }
.step { padding:6px 14px; border-radius:20px; font-size:.82rem; font-weight:600;
        background:#f1f5f9; color:#94a3b8; }
.step.active { background:#3b82f6; color:#fff; }
.step.done   { background:#dcfce7; color:#16a34a; }

/* ── Input cards ─── */
.input-card {
    background: #ffffff;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 22px;
}

/* ── Metric pill ─── */
.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 0;
    text-align: center;
}

/* ── Summary box ─── */
.summary-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    color: #374151;
    line-height: 1.65;
}

/* ── Chip / tag ─── */
.chip {
    display: inline-block;
    border: 1.5px solid #cbd5e1;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .82rem;
    color: #475569;
    margin: 3px 4px 3px 0;
}

/* ── Chat suggestion chips ─── */
.suggestion-chip {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: .8rem;
    cursor: pointer;
    margin: 2px 4px 2px 0;
}

/* ── Table alternating rows ─── */
.evidence-table td, .evidence-table th {
    padding: 8px 12px; font-size: .85rem; border-bottom: 1px solid #f1f5f9;
}
.evidence-table tr:nth-child(even) { background: #f8fafc; }

/* ── Banners ─── */
.banner-success {
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 10px; padding: 14px 18px; color: #166534;
}
.banner-neutral {
    background: #f8fafc; border: 1px solid #cbd5e1;
    border-radius: 10px; padding: 14px 18px; color: #475569;
}
.banner-error {
    background: #fef2f2; border: 1px solid #fca5a5;
    border-radius: 10px; padding: 14px 18px; color: #991b1b;
}

/* ── Recruiter drag-drop zone ─── */
.upload-zone {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
}

/* ── Start over button ─── */
div[data-testid="stButton"] button.start-over {
    background: transparent;
    border: 1.5px solid #cbd5e1;
    color: #64748b;
    font-size: .82rem;
    padding: 4px 14px;
    border-radius: 6px;
}

/* ── Hide streamlit branding ─── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "resume_dict": None,
        "resume_filename": None,
        "job_description": None,
        "analysis_results": None,
        "followup_questions": None,
        "reeval_results": None,
        "chat_history": [],
        "api_key_set": False,
        "analysis_complete": False,
        "current_mode": "Candidate",
        "memory_context": "",
        "chat_handler": None,
        # Mid-term (cross-session) profile memory.
        "profile_session_id": None,
        "returning_profile": None,
        "previously_asked": [],
        "job_title": None,
        "job_company": None,
        "analysis_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# Set api_key_set from env on every load.
st.session_state.api_key_set = bool(_ENV_API_KEY)
if _ENV_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = _ENV_API_KEY


# ── Lazy backend imports ──────────────────────────────────────────────────────

def _backend():
    """Import backend modules lazily so the app starts fast."""
    from agents.reflector import re_evaluate
    from agents.resume_parser import parse_resume
    from memory.chat_handler import ConversationHandler
    from workflow import run_copilot
    return run_copilot, parse_resume, re_evaluate, ConversationHandler


@st.cache_resource(show_spinner=False)
def get_profile_store():
    """One persistent ProfileStore instance for the app."""
    from memory.profile_store import ProfileStore
    return ProfileStore()


# ── Helpers ───────────────────────────────────────────────────────────────────

_STAGE_LABEL = {"seed": "Seed", "series-a": "Series A",
                "series-b": "Series B", "mnc": "MNC"}


@st.cache_data(show_spinner=False)
def get_job_options() -> list:
    """Return [(label, job_row_dict)] for the role dropdown, from the SQL DB."""
    if not _DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY title"
    ).fetchall()
    conn.close()
    options = []
    for r in rows:
        stage = _STAGE_LABEL.get(r["company_stage"], r["company_stage"])
        label = f'{r["title"]} at {r["company"]} ({stage}, {r["location"]})'
        options.append((label, dict(r)))
    return options


def _job_to_jd(job: dict) -> str:
    """Build a job-description string from a jobs-table row."""
    stage = _STAGE_LABEL.get(job["company_stage"], job["company_stage"])
    return (
        f'{job["title"]} at {job["company"]} ({stage}, {job["location"]}).\n'
        f'Experience required: {job["experience_min"]}-{job["experience_max"]} '
        f'years. Salary: {job["salary_min"]}-{job["salary_max"]} LPA.\n'
        f'Required skills: {job["skills_required"]}.\n'
        f'Industry: {job["industry"]}. '
        f'Remote friendly: {"yes" if job["remote_friendly"] else "no"}.\n\n'
        f'{job["full_description"]}'
    )


def _ensure_api_key(key_input: str) -> bool:
    """Set the API key into the environment; return True if a key is available."""
    if key_input:
        os.environ["OPENROUTER_API_KEY"] = key_input.strip()
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _score_label(score: int):
    """Return (emoji, label, hex_color) for a fit score."""
    if score >= 75:
        return "✓", "Strong Fit", "#22c55e"
    if score >= 55:
        return "~", "Possible Fit", "#f59e0b"
    if score >= 35:
        return "△", "Weak Fit", "#ef4444"
    return "✗", "Not a Fit", "#ef4444"


def _score_color(score: int) -> str:
    if score >= 75:
        return "#22c55e"
    if score >= 55:
        return "#f59e0b"
    return "#ef4444"


def _colored_bar(pct: int, color: str, height: int = 8):
    pct = max(0, min(100, int(pct)))
    st.markdown(
        f'<div style="background:#f1f5f9;border-radius:99px;height:{height}px;'
        f'width:100%;overflow:hidden;margin-top:4px">'
        f'<div style="background:{color};width:{pct}%;height:{height}px;'
        f'border-radius:99px;transition:width .3s"></div></div>',
        unsafe_allow_html=True,
    )


# Source badges — updated labels (no "SQL" / "RAG" jargon visible to users)
_SOURCE_STYLES = {
    "sql":    ("Job Data",    "#eff6ff", "#2563eb"),   # blue
    "rag":    ("Culture Docs","#f5f3ff", "#7c3aed"),   # purple
    "resume": ("Resume",      "#f0fdf4", "#15803d"),   # green
}


def _source_badge(source: str) -> str:
    label, bg, fg = _SOURCE_STYLES.get(
        (source or "").lower(), ("Resume", "#f0fdf4", "#15803d")
    )
    return (
        f'<span style="background:{bg};color:{fg};border-radius:20px;'
        f'padding:2px 10px;font-size:0.72rem;font-weight:600;'
        f'letter-spacing:0.03em">{label}</span>'
    )


def _badge(text: str) -> str:
    """Generic gray badge (kept for backwards compat)."""
    return (f'<span style="background:#f1f5f9;color:#475569;border-radius:10px;'
            f'padding:2px 9px;font-size:0.75rem">{text}</span>')


# ── No-key error screen ───────────────────────────────────────────────────────

def _render_no_key_screen():
    """Full-page error when OPENROUTER_API_KEY is missing from .env."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 40px;background:#fff;
                        border:1px solid #e2e8f0;border-radius:16px;
                        box-shadow:0 2px 12px rgba(0,0,0,.06)">
                <div style="font-size:3.5rem;margin-bottom:16px">🔒</div>
                <h2 style="margin-bottom:8px;color:#111827">Configuration Required</h2>
                <p style="color:#6b7280;line-height:1.6;max-width:380px;margin:0 auto 24px">
                    This app requires an OpenRouter API key to run.<br>
                    Please add it to your <code>.env</code> file:
                </p>
                <div style="background:#f8fafc;border:1px solid #e2e8f0;
                            border-radius:8px;padding:14px 20px;
                            font-family:monospace;font-size:.9rem;
                            color:#374151;text-align:left">
                    OPENROUTER_API_KEY=your_key_here
                </div>
                <p style="color:#9ca3af;font-size:.82rem;margin-top:18px">
                    Get a free key at
                    <a href="https://openrouter.ai" target="_blank"
                       style="color:#2563eb">openrouter.ai</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 2px'>"
            "<span style='font-size:1.25rem;font-weight:700;color:#111827'>"
            "🎯 AI Hiring Co-pilot</span><br>"
            "<span style='font-size:.82rem;color:#6b7280'>"
            "Evidence-based job fit analysis</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:14px 0;border-color:#e2e8f0'>",
                    unsafe_allow_html=True)

        # Mode toggle
        st.markdown(
            "<p style='font-size:.8rem;font-weight:600;color:#374151;"
            "text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px'>"
            "MODE</p>",
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Mode",
            ["Candidate", "Recruiter"],
            index=0 if st.session_state.current_mode == "Candidate" else 1,
            label_visibility="collapsed",
        )
        st.session_state.current_mode = mode

        st.markdown("<hr style='margin:14px 0;border-color:#e2e8f0'>",
                    unsafe_allow_html=True)

        with st.expander("How it works"):
            st.markdown(
                """
- **Upload** your resume — we parse it instantly
- **Choose** a role from our database or paste a JD
- **AI analysis** cross-references market data and company culture to score your fit
- **Refine** by answering follow-up questions — your score updates in real time
                """,
                unsafe_allow_html=False,
            )

    return mode


# ── Parse resume ──────────────────────────────────────────────────────────────

def _parse_uploaded_resume(uploaded) -> bool:
    """
    Validate the PDF, recognise a returning candidate by resume hash, and either
    load their stored profile (skipping re-parse) or parse + create a new one.
    """
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > _MAX_MB:
        st.markdown(
            f'<div class="banner-error">⚠ Please upload a PDF under {_MAX_MB} MB '
            f'(this file is {size_mb:.1f} MB).</div>',
            unsafe_allow_html=True,
        )
        return False
    if not st.session_state.api_key_set:
        st.markdown(
            '<div class="banner-error">⚠ API key not configured. '
            'Add OPENROUTER_API_KEY to your .env file.</div>',
            unsafe_allow_html=True,
        )
        return False

    store = get_profile_store()
    pdf_bytes = uploaded.getvalue()
    resume_hash = store.hash_resume(pdf_bytes)

    existing = store.find_by_hash(resume_hash)
    if existing:
        st.session_state.resume_dict = existing["parsed_resume"]
        st.session_state.resume_filename = uploaded.name
        st.session_state.profile_session_id = existing["session_id"]
        st.session_state.returning_profile = existing
        st.session_state.previously_asked = store.get_asked_questions(
            existing["session_id"])
        return True

    _, parse_resume, _, _ = _backend()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        with st.spinner("Reading your resume…"):
            resume = parse_resume(tmp_path)
        os.unlink(tmp_path)
    except Exception as exc:  # noqa: BLE001
        st.markdown(
            f'<div class="banner-error">⚠ We had trouble reading your resume. '
            f'Try a different PDF format. (Details: {exc})</div>',
            unsafe_allow_html=True,
        )
        return False

    if not resume or not resume.get("name"):
        st.markdown(
            '<div class="banner-error">⚠ We had trouble reading your resume. '
            'Try a different PDF format or copy-paste your resume text.</div>',
            unsafe_allow_html=True,
        )
        return False

    session_id = store.create_profile(resume_hash, resume,
                                      resume.get("name", "Candidate"))
    st.session_state.resume_dict = resume
    st.session_state.resume_filename = uploaded.name
    st.session_state.profile_session_id = session_id
    st.session_state.returning_profile = None
    st.session_state.previously_asked = []
    return True


# ── Run analysis ──────────────────────────────────────────────────────────────

def _run_analysis(resume_dict: dict, job_description: str):
    """Run the full pipeline with a clean progress display."""
    run_copilot, _, _, ConversationHandler = _backend()

    step_labels = [
        ("1", "Parse"),
        ("2", "Plan"),
        ("3", "Research"),
        ("4", "Analyze"),
        ("5", "Refine"),
    ]
    status_messages = [
        "Parsing your resume…",
        "Planning the investigation strategy…",
        "Researching company culture and job requirements…",
        "Synthesizing your fit analysis…",
        "Running quality checks…",
        "Generating your personalized questions…",
    ]

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    def _render_steps(active_idx: int):
        parts = []
        for i, (num, label) in enumerate(step_labels):
            if i < active_idx:
                cls = "done"
                display = f"✓ {label}"
            elif i == active_idx:
                cls = "active"
                display = f"{num}. {label}"
            else:
                cls = ""
                display = f"{num}. {label}"
            parts.append(f'<span class="step {cls}">{display}</span>')
        progress_placeholder.markdown(
            f'<div class="step-row">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )

    _render_steps(0)
    status_placeholder.caption(status_messages[0])

    try:
        for i in range(1, len(step_labels)):
            _render_steps(i)
            status_placeholder.caption(status_messages[i])

        result = run_copilot(
            resume_dict, job_description,
            previously_asked=st.session_state.previously_asked,
        )
    except Exception as exc:  # noqa: BLE001
        progress_placeholder.empty()
        status_placeholder.empty()
        st.markdown(
            f'<div class="banner-error">⚠ Analysis failed. Please try again. '
            f'<br><small style="opacity:.7">{exc}</small></div>',
            unsafe_allow_html=True,
        )
        return

    progress_placeholder.empty()
    status_placeholder.empty()

    st.session_state.analysis_results = result
    st.session_state.followup_questions = result.get("followup_questions", {})
    st.session_state.memory_context = result.get("memory_context", "")
    st.session_state.reeval_results = None
    st.session_state.chat_history = []
    st.session_state.chat_handler = ConversationHandler()
    st.session_state.analysis_complete = True

    sid = st.session_state.profile_session_id
    if sid:
        synth = result.get("synthesis", {})
        get_profile_store().save_analysis(
            sid,
            st.session_state.job_title or "Custom JD",
            st.session_state.job_company or "—",
            int(synth.get("fit_score", 0)),
            synth.get("recommendation", "not_fit"),
            synth,
        )


# ── Results rendering ─────────────────────────────────────────────────────────

def _render_results():
    result = st.session_state.analysis_results
    synth = result.get("synthesis", {})
    active = st.session_state.reeval_results or synth
    score = int(active.get("fit_score", 0))
    _, label, color = _score_label(score)

    # ── Score card ──
    rec_descriptions = {
        "strong_fit":   "You closely match the requirements. Go ahead and apply confidently.",
        "possible_fit": "You meet some requirements. Address the gaps to strengthen your application.",
        "weak_fit":     "There are significant gaps. Consider upskilling or targeting similar roles.",
        "not_fit":      "This role may not be the right match right now. See alternatives below.",
    }
    rec_key = active.get("recommendation", "not_fit")
    rec_desc = rec_descriptions.get(rec_key, "")

    bd = active.get("score_breakdown", {})

    st.markdown(
        f"""
        <div class="score-card">
          <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap">
            <div style="min-width:140px">
              <div style="font-size:4rem;font-weight:800;line-height:1;color:{color}">{score}</div>
              <div style="font-size:1rem;color:#94a3b8;font-weight:500;margin-top:2px">/ 100</div>
              <div style="margin-top:10px;font-size:1.15rem;font-weight:700;color:{color}">{label}</div>
            </div>
            <div style="flex:1;min-width:220px;padding-top:6px">
              <p style="color:#374151;margin:0 0 18px;line-height:1.5">{rec_desc}</p>
              <div style="display:flex;gap:12px;flex-wrap:wrap">
                <div class="metric-pill" style="flex:1;min-width:90px">
                  <div style="font-size:1.4rem;font-weight:700;color:#1e293b">{int(bd.get("skills_match", 0))}</div>
                  <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em">Skills Match</div>
                </div>
                <div class="metric-pill" style="flex:1;min-width:90px">
                  <div style="font-size:1.4rem;font-weight:700;color:#1e293b">{int(bd.get("experience_match", 0))}</div>
                  <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em">Experience</div>
                </div>
                <div class="metric-pill" style="flex:1;min-width:90px">
                  <div style="font-size:1.4rem;font-weight:700;color:#1e293b">{int(bd.get("culture_fit", 0))}</div>
                  <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em">Culture Fit</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Re-eval delta banner
    if st.session_state.reeval_results:
        delta = st.session_state.reeval_results.get("score_delta", 0)
        if delta:
            old_score = int(synth.get("fit_score", 0))
            new_score = score
            banner_cls = "banner-success" if delta > 0 else "banner-neutral"
            arrow = f"+{delta}" if delta > 0 else str(delta)
            st.markdown(
                f'<div class="{banner_cls}" style="margin-bottom:16px">'
                f'Your score improved from <strong>{old_score}</strong> to '
                f'<strong>{new_score}</strong> ({arrow} points)</div>',
                unsafe_allow_html=True,
            )

    # Reflection cycle note
    ref_cycles = (st.session_state.analysis_results or {}).get("reflection_cycles", 0)
    if ref_cycles and ref_cycles > 1:
        st.markdown(
            f'<div style="font-size:.8rem;color:#94a3b8;margin-bottom:12px">'
            f'ℹ Analysis refined through {ref_cycles} reasoning cycles for higher accuracy.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Strengths + Gaps ──
    left, right = st.columns(2)

    with left:
        st.markdown(
            "<p style='font-size:.85rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.06em;color:#64748b;margin-bottom:12px'>"
            "What works in your favor</p>",
            unsafe_allow_html=True,
        )
        strengths = active.get("strengths", [])
        if not strengths:
            st.markdown(
                '<div class="banner-neutral">No specific strengths identified.</div>',
                unsafe_allow_html=True,
            )
        for s in strengths:
            src = (s.get("source") or "resume").lower()
            st.markdown(
                f'<div class="item-card strength">'
                f'<div style="font-size:.92rem;font-weight:600;color:#111827;margin-bottom:6px">'
                f'{s.get("point","")}</div>'
                f'<div style="margin-bottom:6px">{_source_badge(src)}</div>'
                + (f'<div style="font-size:.82rem;color:#64748b;font-style:italic">'
                   f'{s["evidence"]}</div>' if s.get("evidence") else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            "<p style='font-size:.85rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.06em;color:#64748b;margin-bottom:12px'>"
            "Areas to address</p>",
            unsafe_allow_html=True,
        )
        gaps = active.get("gaps", [])
        if not gaps:
            st.markdown(
                '<div class="banner-success">No significant gaps identified.</div>',
                unsafe_allow_html=True,
            )
        for g in gaps:
            sev = g.get("severity", "moderate")
            src = (g.get("source") or "resume").lower()
            resolved_html = (
                ' <span style="color:#22c55e;font-size:.8rem">✓ resolved</span>'
                if g.get("resolved") else ""
            )
            st.markdown(
                f'<div class="item-card {sev}">'
                f'<div style="font-size:.92rem;font-weight:600;color:#111827;margin-bottom:6px">'
                f'{g.get("point","")}{resolved_html}</div>'
                f'<div style="margin-bottom:6px">{_source_badge(src)}</div>'
                + (f'<div style="font-size:.82rem;color:#64748b;font-style:italic">'
                   f'{g["evidence"]}</div>' if g.get("evidence") else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary ──
    summary = active.get("summary", "")
    if summary:
        st.markdown(
            "<p style='font-size:.78rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.06em;color:#64748b;margin-bottom:6px'>Summary</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="summary-box">{summary}</div>',
            unsafe_allow_html=True,
        )

    # Citation count
    cit = synth.get("citations", {})
    n_sql = len(cit.get("from_sql", []) or [])
    n_rag = len(cit.get("from_rag", []) or [])
    if n_sql or n_rag:
        st.markdown(
            f'<div style="font-size:.78rem;color:#94a3b8;margin-top:8px">'
            f'Analysis based on {n_sql} market data point(s) and {n_rag} culture insight(s).</div>',
            unsafe_allow_html=True,
        )

    # Alternative roles
    if rec_key != "strong_fit":
        alts = active.get("alternative_roles", [])
        if alts:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.82rem;font-weight:600;color:#374151;margin-bottom:6px'>"
                "You may be a stronger fit for:</p>",
                unsafe_allow_html=True,
            )
            chips = "".join(f'<span class="chip">{a}</span>' for a in alts)
            st.markdown(chips, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Evidence expander ──
    with st.expander("View Evidence"):
        sql_cits = cit.get("from_sql", []) or []
        rag_cits = cit.get("from_rag", []) or []
        all_rows = (
            [("Job Data", c) for c in sql_cits]
            + [("Culture Docs", c) for c in rag_cits]
        )
        if all_rows:
            rows_html = "".join(
                f"<tr><td style='padding:8px 12px;font-size:.8rem;white-space:nowrap;"
                f"color:#475569'>{_source_badge(('sql' if src=='Job Data' else 'rag'))}"
                f"</td><td style='padding:8px 12px;font-size:.82rem;color:#374151'>"
                f"{txt}</td></tr>"
                for src, txt in all_rows
            )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse">'
                f'<thead><tr>'
                f'<th style="padding:8px 12px;font-size:.78rem;text-align:left;'
                f'color:#94a3b8;border-bottom:1px solid #e2e8f0">Source</th>'
                f'<th style="padding:8px 12px;font-size:.78rem;text-align:left;'
                f'color:#94a3b8;border-bottom:1px solid #e2e8f0">Finding</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No evidence citations recorded for this analysis.")

        # Reflection quality notes
        reflection = (st.session_state.analysis_results or {}).get("reflection", {})
        weak_pts = (reflection or {}).get("weak_points", [])
        if weak_pts:
            st.markdown(
                "<p style='font-size:.78rem;color:#f59e0b;font-weight:600;"
                "margin-top:14px'>Quality notes:</p>",
                unsafe_allow_html=True,
            )
            for wp in weak_pts:
                st.caption(f"• {wp.get('claim', '')}")


# ── Welcome back ──────────────────────────────────────────────────────────────

def _render_previously_addressed():
    """Acknowledge gaps the candidate already addressed in a previous session."""
    prof = st.session_state.returning_profile
    if not prof:
        return
    history = prof.get("followup_history", [])
    if not history:
        return
    current_gaps = (st.session_state.analysis_results or {}).get(
        "synthesis", {}).get("gaps", [])
    gap_points = [g.get("point", "").lower() for g in current_gaps]
    shown = False
    for qa in history:
        ga = (qa.get("gap_addressed", "") or "").lower()
        if not ga:
            continue
        if any(ga in gp or gp in ga or gp[:25] == ga[:25] for gp in gap_points):
            if not shown:
                st.markdown(
                    "<p style='font-size:.85rem;font-weight:600;color:#374151;"
                    "margin-bottom:8px'>From your earlier sessions</p>",
                    unsafe_allow_html=True,
                )
                shown = True
            st.markdown(
                f'<div class="banner-success" style="margin-bottom:8px">'
                f'Previously you mentioned: <em>"{qa.get("answer","")}"</em> '
                f'— counting this toward your score. ✓</div>',
                unsafe_allow_html=True,
            )


def _render_welcome_back():
    """Warm welcome + analysis history for a returning candidate."""
    prof = st.session_state.returning_profile
    store = get_profile_store()
    sid = prof["session_id"]
    name = prof.get("candidate_name") or st.session_state.resume_dict.get("name", "")
    history = store.get_analysis_history(sid)
    best = store.get_best_match(sid)

    msg = f"Welcome back, {name}! You've analyzed fit for **{len(history)}** role(s)."
    if best:
        msg += (f" Your strongest match was **{best.get('job_title','?')}** "
                f"at **{best.get('company','?')}** (**{best.get('fit_score','?')}%**).")
    st.markdown(
        f'<div class="banner-success" style="margin-bottom:12px">'
        f'👋 {msg}</div>',
        unsafe_allow_html=True,
    )

    if history:
        with st.expander("Your analysis history"):
            for a in reversed(history):
                st.markdown(
                    f"- **{a.get('job_title','?')}** at {a.get('company','?')} — "
                    f"**{a.get('fit_score','?')}%** "
                    f"({a.get('recommendation','?')}) "
                    f"· {a.get('timestamp','')[:10]}")


# ── Follow-ups ────────────────────────────────────────────────────────────────

def _render_followups():
    fq = (st.session_state.followup_questions or {}).get("questions", [])

    st.markdown("<hr style='margin:28px 0;border-color:#e2e8f0'>",
                unsafe_allow_html=True)
    _render_previously_addressed()

    if not fq:
        st.markdown(
            '<div class="banner-neutral" style="font-size:.85rem">'
            'No new follow-up questions — your earlier answers already cover the open gaps.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        "<h3 style='margin-bottom:4px'>Help us refine your score</h3>"
        "<p style='color:#64748b;font-size:.88rem;margin-bottom:20px'>"
        "Answer these questions to potentially improve your fit assessment.</p>",
        unsafe_allow_html=True,
    )

    for i, q in enumerate(fq):
        char_key = f"answer_{i}"
        ans_val = st.session_state.get(char_key, "")
        char_count = len(ans_val)
        st.markdown(
            f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:16px 18px;margin-bottom:12px">'
            f'<p style="font-size:.9rem;font-weight:600;color:#111827;margin-bottom:10px">'
            f'{i+1}. {q.get("question","")}</p>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Your answer",
            key=char_key,
            label_visibility="collapsed",
            placeholder="Your answer…",
            height=100,
        )
        st.markdown(
            f'<div style="font-size:.72rem;color:#94a3b8;text-align:right;'
            f'margin-top:-8px;margin-bottom:4px">{char_count} characters</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Update My Score", type="primary"):
        _, _, re_evaluate, _ = _backend()
        followup_qa, qa_meta = [], []
        for i, q in enumerate(fq):
            ans = st.session_state.get(f"answer_{i}", "").strip()
            if ans:
                followup_qa.append({"question": q["question"], "answer": ans})
                qa_meta.append((q["question"], ans,
                                q.get("gap_addressed", "")))
        if not followup_qa:
            st.markdown(
                '<div class="banner-error">Please answer at least one question first.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Updating your score…"):
                updated = re_evaluate(
                    st.session_state.analysis_results["synthesis"],
                    followup_qa,
                    st.session_state.followup_questions,
                )
            st.session_state.reeval_results = updated
            sid = st.session_state.profile_session_id
            if sid:
                store = get_profile_store()
                for question, answer, gap in qa_meta:
                    store.save_followup_qa(sid, question, answer, gap)
            st.rerun()

    # Score update outcome
    rr = st.session_state.reeval_results
    if rr:
        old = int(st.session_state.analysis_results["synthesis"]["fit_score"])
        new = int(rr["fit_score"])
        delta = rr.get("score_delta", new - old)
        if delta > 0:
            st.markdown(
                f'<div class="banner-success" style="margin-top:12px">'
                f'Your score improved from <strong>{old}</strong> to '
                f'<strong>{new}</strong> (+{delta} points)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner-neutral" style="margin-top:12px">'
                f'Your score remains at <strong>{old}</strong>. '
                f'Consider addressing the remaining gaps.</div>',
                unsafe_allow_html=True,
            )
        resolved = rr.get("resolved_gaps", [])
        if resolved:
            st.markdown(
                "<p style='font-size:.85rem;font-weight:600;color:#374151;"
                "margin:12px 0 6px'>Gaps you addressed:</p>",
                unsafe_allow_html=True,
            )
            for g in resolved:
                st.markdown(
                    f'<div style="font-size:.85rem;color:#16a34a;'
                    f'text-decoration:line-through;margin-bottom:3px">'
                    f'✓ {g.get("point","")}</div>',
                    unsafe_allow_html=True,
                )


# ── Chat ──────────────────────────────────────────────────────────────────────

def _render_chat():
    st.markdown("<hr style='margin:28px 0;border-color:#e2e8f0'>",
                unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-bottom:4px'>Ask a question about your analysis</h3>",
        unsafe_allow_html=True,
    )

    # Suggestion chips
    suggestions = [
        "What should I focus on?",
        "Is this role right for me?",
        "What can I improve?",
    ]
    chip_html = "".join(
        f'<span class="suggestion-chip">{s}</span>' for s in suggestions
    )
    st.markdown(
        f'<div style="margin-bottom:12px">{chip_html}</div>',
        unsafe_allow_html=True,
    )
    # Chip click: populate input via selectbox trick
    def _handle_chip():
        if st.session_state.chip_select:
            st.session_state.pending_prompt = st.session_state.chip_select
            st.session_state.chip_select = ""

    st.selectbox(
        "Quick questions",
        [""] + suggestions,
        label_visibility="collapsed",
        key="chip_select",
        on_change=_handle_chip,
    )

    handler = st.session_state.chat_handler
    for turn in handler.get_history():
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Ask about your fit analysis…")
    
    if st.session_state.get("pending_prompt"):
        if not prompt:
            prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = handler.answer_question(
                    prompt, st.session_state.memory_context)
            st.markdown(answer)
        st.rerun()


# ── Demo mode ─────────────────────────────────────────────────────────────────

def _render_demo_banner():
    """Show demo mode banner at the top of the page."""
    st.markdown(
        '<div class="banner-neutral" style="margin-bottom:20px">'
        '🎮 <strong>Demo Mode</strong> — viewing a pre-computed example analysis '
        '(Arjun Mehta applying to ML Engineer at a Series A fintech). '
        'Add <code>OPENROUTER_API_KEY</code> to your <code>.env</code> '
        'file for live analysis of your own resume.</div>',
        unsafe_allow_html=True,
    )


def _render_demo_mode():
    """Display the pre-computed demo result when no API key is available."""
    if not _DEMO_RESULT:
        st.markdown(
            '<div class="banner-error">Demo result file not found. '
            'Add your API key to run a live analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    st.session_state.analysis_results = _DEMO_RESULT
    st.session_state.followup_questions = _DEMO_RESULT.get("followup_questions", {})
    st.session_state.memory_context = _DEMO_RESULT.get("memory_context", "")
    st.session_state.reeval_results = None
    st.session_state.analysis_complete = True

    _render_results()

    st.markdown("<hr style='margin:28px 0;border-color:#e2e8f0'>",
                unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-bottom:4px'>Help us refine your score</h3>"
        "<p style='color:#64748b;font-size:.88rem;margin-bottom:20px'>"
        "These are the questions the system would ask to refine the score. "
        "Add your API key to answer them and re-evaluate.</p>",
        unsafe_allow_html=True,
    )
    fq = _DEMO_RESULT.get("followup_questions", {}).get("questions", [])
    for i, q in enumerate(fq, 1):
        st.markdown(
            f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:16px 18px;margin-bottom:10px">'
            f'<p style="font-size:.9rem;font-weight:600;color:#111827;margin-bottom:4px">'
            f'{i}. {q.get("question","")}</p>'
            f'<p style="font-size:.78rem;color:#94a3b8;margin:0">'
            f'Addresses: {q.get("gap_addressed","")}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Candidate mode ────────────────────────────────────────────────────────────

def candidate_mode():
    # ── No API key: full-page error ──
    if not st.session_state.api_key_set:
        _render_no_key_screen()
        return

    # ── Demo mode ──
    if IS_DEMO and not st.session_state.api_key_set:
        st.markdown(
            "<h1 style='font-size:2.2rem;font-weight:800;margin-bottom:4px'>"
            "AI Hiring Co-pilot</h1>"
            "<p style='color:#64748b;font-size:1rem;margin-bottom:24px'>"
            "Know your fit before you apply</p>",
            unsafe_allow_html=True,
        )
        _render_demo_banner()
        _render_demo_mode()
        return

    if not st.session_state.analysis_complete:
        # ── Landing: heading ──
        col_h, col_btn = st.columns([4, 1])
        with col_h:
            st.markdown(
                "<h1 style='font-size:2.2rem;font-weight:800;margin-bottom:4px'>"
                "AI Hiring Co-pilot</h1>"
                "<p style='color:#64748b;font-size:1rem;margin-bottom:28px'>"
                "Know your fit before you apply</p>",
                unsafe_allow_html=True,
            )

        # ── Input cards ──
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                '<div class="input-card">'
                '<p style="font-weight:700;font-size:.95rem;color:#111827;margin-bottom:2px">'
                'Your Resume</p>'
                '<p style="font-size:.82rem;color:#64748b;margin-bottom:14px">'
                'Upload your PDF resume</p>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Resume upload",
                type=["pdf"],
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if uploaded is not None:
                size_kb = uploaded.size / 1024
                st.markdown(
                    f'<div style="font-size:.82rem;color:#64748b;margin-top:6px">'
                    f'📎 {uploaded.name} ({size_kb:.0f} KB)</div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.resume_filename != uploaded.name:
                    if _parse_uploaded_resume(uploaded):
                        st.rerun()
                if (st.session_state.resume_dict
                        and st.session_state.resume_filename == uploaded.name):
                    if st.session_state.returning_profile:
                        _render_welcome_back()
                    else:
                        name = st.session_state.resume_dict.get("name", "")
                        st.markdown(
                            f'<div class="banner-success" style="margin-top:8px;'
                            f'font-size:.85rem">Resume parsed — hi, {name}!</div>',
                            unsafe_allow_html=True,
                        )

        with right:
            st.markdown(
                '<div class="input-card">'
                '<p style="font-weight:700;font-size:.95rem;color:#111827;margin-bottom:2px">'
                'Target Role</p>'
                '<p style="font-size:.82rem;color:#64748b;margin-bottom:14px">'
                'Select from database or paste a JD</p>',
                unsafe_allow_html=True,
            )
            choice = st.radio(
                "Role source",
                ["Select a role", "Paste a JD"],
                label_visibility="collapsed",
            )
            jd = None
            if choice == "Select a role":
                options = get_job_options()
                if not options:
                    st.markdown(
                        '<div class="banner-error" style="font-size:.85rem">'
                        'No roles found in the database.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    labels = [o[0] for o in options]
                    sel = st.selectbox("Select a role", labels,
                                       label_visibility="collapsed")
                    job = dict(options[labels.index(sel)][1])
                    jd = _job_to_jd(job)
                    st.session_state.job_title = job["title"]
                    st.session_state.job_company = job["company"]
            else:
                jd = st.text_area(
                    "Job description",
                    height=160,
                    label_visibility="collapsed",
                    placeholder="Paste the full job description here (min 100 characters)…",
                )
                if jd and len(jd.strip()) < 100:
                    st.markdown(
                        f'<div style="font-size:.75rem;color:#f59e0b;margin-top:2px">'
                        f'{len(jd.strip())} / 100 characters minimum</div>',
                        unsafe_allow_html=True,
                    )
                st.session_state.job_title = "Custom JD"
                st.session_state.job_company = "—"

            if jd and len(jd.strip()) >= 100:
                st.session_state.job_description = jd.strip()
            elif choice == "Paste a JD":
                st.session_state.job_description = None

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Analyze button ──
        ready = bool(
            st.session_state.resume_dict
            and st.session_state.job_description
            and st.session_state.api_key_set
        )
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button(
                "Analyze My Fit →",
                disabled=not ready,
                use_container_width=True,
                type="primary",
            ):
                _run_analysis(
                    st.session_state.resume_dict,
                    st.session_state.job_description,
                )
                st.rerun()

        if not ready and st.session_state.resume_dict:
            st.markdown(
                '<div style="text-align:center;font-size:.82rem;color:#94a3b8;'
                'margin-top:6px">Provide a role to enable analysis.</div>',
                unsafe_allow_html=True,
            )

    else:
        # ── Results view ──
        top_l, top_r = st.columns([6, 1])
        with top_l:
            st.markdown(
                "<h2 style='font-size:1.4rem;font-weight:700;margin-bottom:0'>"
                "Your Fit Analysis</h2>",
                unsafe_allow_html=True,
            )
        with top_r:
            if st.button("Start Over", key="start_over"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

        _render_results()
        _render_followups()
        _render_chat()


# ── Recruiter mode ────────────────────────────────────────────────────────────

_ROW_COLOR = {
    "strong_fit":  "background-color: #f0fdf4",
    "possible_fit": "background-color: #fffbeb",
    "weak_fit":    "background-color: #fff7ed",
    "not_fit":     "background-color: #fef2f2",
}


def recruiter_mode():
    # ── No API key guard ──
    if not st.session_state.api_key_set:
        _render_no_key_screen()
        return

    st.markdown(
        "<h1 style='font-size:2rem;font-weight:800;margin-bottom:4px'>"
        "Batch Analysis</h1>"
        "<p style='color:#64748b;font-size:.95rem;margin-bottom:28px'>"
        "Analyze multiple candidates for one role</p>",
        unsafe_allow_html=True,
    )

    # Upload zone
    st.markdown(
        '<div class="upload-zone">',
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Drop resumes here or click to upload",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    options = get_job_options()
    if not options:
        st.markdown(
            '<div class="banner-error">No roles found in the database.</div>',
            unsafe_allow_html=True,
        )
        return
    labels = [o[0] for o in options]
    sel = st.selectbox("Select a role", labels)
    job = dict(options[labels.index(sel)][1])
    jd = _job_to_jd(job)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Analyze All →", type="primary", disabled=not files):
        run_copilot, parse_resume, _, _ = _backend()
        rows = []
        progress = st.progress(0.0, text="Starting…")
        total = len(files)
        for idx, f in enumerate(files, 1):
            progress.progress((idx - 1) / total,
                              text=f"Analyzing candidate {idx} of {total}…")
            size_mb = f.size / (1024 * 1024)
            if size_mb > _MAX_MB:
                rows.append({"Candidate": f.name, "Score": 0,
                             "Recommendation": "not_fit",
                             "Top Strength": "(file too large)",
                             "Key Gap": f">{_MAX_MB} MB",
                             "Skills Match": 0, "Experience Match": 0})
                continue
            try:
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                resume = parse_resume(tmp_path)
                os.unlink(tmp_path)
                result = run_copilot(resume, jd)
                synth = result["synthesis"]
                bd = synth.get("score_breakdown", {})
                strengths = synth.get("strengths", [])
                crit = [g for g in synth.get("gaps", [])
                        if g.get("severity") == "critical"]
                rows.append({
                    "Candidate": resume.get("name", f.name),
                    "Score": int(synth.get("fit_score", 0)),
                    "Recommendation": synth.get("recommendation", "not_fit"),
                    "Top Strength": (strengths[0]["point"] if strengths else "—"),
                    "Key Gap": (crit[0]["point"] if crit else "—"),
                    "Skills Match": int(bd.get("skills_match", 0)),
                    "Experience Match": int(bd.get("experience_match", 0)),
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({"Candidate": f.name, "Score": 0,
                             "Recommendation": "not_fit",
                             "Top Strength": "(analysis failed)",
                             "Key Gap": str(exc)[:60],
                             "Skills Match": 0, "Experience Match": 0})
        progress.progress(1.0, text="Done.")

        if rows:
            df = pd.DataFrame(rows).sort_values(
                "Score", ascending=False).reset_index(drop=True)

            # Download button at the top right
            dl_col, _ = st.columns([1, 4])
            with dl_col:
                st.download_button(
                    "Export as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="candidate_analysis.csv",
                    mime="text/csv",
                )

            def _color_row(row):
                return [_ROW_COLOR.get(row["Recommendation"], "")] * len(row)

            st.dataframe(
                df.style.apply(_color_row, axis=1),
                use_container_width=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = render_sidebar()
    if mode == "Candidate":
        candidate_mode()
    else:
        recruiter_mode()


if __name__ == "__main__":
    main()
