"""
Resume parser agent.

Extracts raw text from a PDF resume with pdfplumber, cleans it, and uses the
project LLM (OpenRouter) to produce a strict structured JSON profile.

Usage:
    from agents.resume_parser import parse_resume
    profile = parse_resume("data/raw/sample_resume.pdf")
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

import pdfplumber
from dotenv import load_dotenv

# Make the project root importable whether run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.llms.openrouter import OpenRouter  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# Same model as the rest of the project (see project memory: the spec's
# llama-3.1-70b:free was retired; gpt-oss-120b:free is the working free model).
# A larger token budget than the SQL tool, since a full resume JSON is long.
_MODEL = "openai/gpt-oss-120b:free"


def _get_extraction_llm() -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(
        api_key=api_key,
        model=_MODEL,
        max_tokens=2048,
        temperature=0.0,
    )


# ── Expected output schema ───────────────────────────────────────────────────
# Used to fill any missing fields with null/empty defaults — never hallucinated.

def _empty_profile() -> dict:
    return {
        "name": None,
        "email": None,
        "phone": None,
        "education": [],
        "total_experience_years": None,
        "skills": {
            "programming_languages": [],
            "frameworks_and_libraries": [],
            "tools_and_platforms": [],
            "databases": [],
            "soft_skills": [],
        },
        "work_experience": [],
        "projects": [],
        "certifications": [],
        "target_roles": [],
        "experience_level": None,
    }


# ── PDF extraction + cleaning ────────────────────────────────────────────────

def _extract_page_text(page: "pdfplumber.page.Page") -> str:
    """
    Extract text from a single page, handling multi-column layouts.

    pdfplumber's default extract_text follows reading order which usually works,
    but for multi-column pages we reconstruct lines by clustering words by their
    vertical position and sorting left-to-right within each line.
    """
    text = page.extract_text() or ""
    if text.strip():
        return text

    # Fallback: rebuild from words (helps with awkward/multi-column layouts).
    words = page.extract_words(use_text_flow=False) or []
    if not words:
        return ""
    lines: dict[int, list] = {}
    for w in words:
        # Bucket words into lines by rounded vertical position.
        key = round(w["top"] / 3)
        lines.setdefault(key, []).append(w)
    out_lines = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda x: x["x0"])
        out_lines.append(" ".join(w["text"] for w in row))
    return "\n".join(out_lines)


def _clean_text(text: str) -> str:
    """Normalize encoding, strip page numbers, collapse excess whitespace."""
    # Fix encoding / normalize unicode (smart quotes, ligatures, NBSP, etc.).
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "•": "-", "\xa0": " ",
        "ﬁ": "fi", "ﬂ": "fl",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop page-number lines like "1", "Page 2", "Page 2 of 5".
        if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", stripped, re.IGNORECASE):
            continue
        # Collapse runs of internal whitespace.
        stripped = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(stripped)

    # Collapse 3+ blank lines down to a single blank line.
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_resume_text(pdf_path: str) -> str:
    """Extract and clean all text from a resume PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(_extract_page_text(page))
    return _clean_text("\n".join(parts))


# ── LLM structured extraction ────────────────────────────────────────────────

_SCHEMA_BLOCK = """\
{
  "name": string,
  "email": string,
  "phone": string,
  "education": [
    {"degree": string, "field": string, "institution": string,
     "year": string, "cgpa": string or null}
  ],
  "total_experience_years": float,
  "skills": {
    "programming_languages": [],
    "frameworks_and_libraries": [],
    "tools_and_platforms": [],
    "databases": [],
    "soft_skills": []
  },
  "work_experience": [
    {"company": string, "role": string, "duration_months": int,
     "key_responsibilities": [], "technologies_used": []}
  ],
  "projects": [
    {"name": string, "description": string, "technologies": [],
     "impact": string or null}
  ],
  "certifications": [],
  "target_roles": [],
  "experience_level": "fresher"|"junior"|"mid"|"senior"
}"""


def _build_prompt(resume_text: str, strict: bool = False) -> str:
    strict_note = ""
    if strict:
        strict_note = (
            "\nIMPORTANT: Respond with ONLY raw JSON. No markdown, no code "
            "fences, no explanation, no text before or after the JSON.\n"
        )
    return f"""\
You are a resume-parsing engine. Extract the resume below into JSON.

Output a single JSON object with EXACTLY this structure and keys:
{_SCHEMA_BLOCK}

Rules:
- Output ONLY the JSON object, nothing else.
- Use only information present in the resume. Do NOT invent or guess any value.
- If a field is missing in the resume, use null (or an empty list [] for list
  fields). Never fabricate names, emails, numbers, or skills.
- total_experience_years: sum of professional/internship experience in years as a
  float (e.g. two internships of 3 and 2 months -> 0.42). Exclude projects.
- duration_months: integer number of months for each work experience.
- experience_level: choose "fresher" (no full-time experience, only internships, <1 yr),
  "junior" (~1-3 yrs), "mid" (~3-6 yrs), or "senior" (6+ yrs) based on real
  full-time experience.
- Classify each skill into the correct skills sub-category.
{strict_note}
RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"
"""


def _extract_json(raw: str) -> Optional[dict]:
    """Parse a JSON object out of an LLM response, tolerating fences/prose."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the substring between the first { and the last }.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _coerce(value: Any, default: Any) -> Any:
    """Use the parsed value if present, else the schema default."""
    return default if value is None else value


def _fill_missing(data: dict) -> dict:
    """Ensure every schema field exists; fill gaps with null/empty defaults."""
    profile = _empty_profile()
    if not isinstance(data, dict):
        return profile

    for key in ("name", "email", "phone", "total_experience_years",
                "experience_level"):
        if data.get(key) is not None:
            profile[key] = data[key]

    for key in ("education", "work_experience", "projects",
                "certifications", "target_roles"):
        val = data.get(key)
        if isinstance(val, list):
            profile[key] = val

    skills = data.get("skills")
    if isinstance(skills, dict):
        for sub in profile["skills"]:
            val = skills.get(sub)
            if isinstance(val, list):
                profile["skills"][sub] = val

    return profile


def parse_resume(pdf_path: str) -> dict:
    """
    Parse a resume PDF into a structured profile dict matching the schema.

    Extracts + cleans the PDF text, asks the LLM for strict JSON, validates it,
    and retries once with a stricter prompt if the first parse fails. Missing
    fields are filled with null/empty — never hallucinated.
    """
    resume_text = extract_resume_text(pdf_path)
    llm = _get_extraction_llm()

    # Attempt 1: normal prompt.
    raw = str(llm.complete(_build_prompt(resume_text, strict=False)))
    data = _extract_json(raw)

    # Attempt 2 (retry once): stricter "raw JSON only" prompt.
    if data is None:
        raw = str(llm.complete(_build_prompt(resume_text, strict=True)))
        data = _extract_json(raw)

    if data is None:
        # Both attempts failed to produce valid JSON — return an empty, fully
        # null profile rather than crashing or hallucinating.
        print("[resume_parser] WARNING: could not parse valid JSON from the LLM; "
              "returning an empty profile.")
        return _empty_profile()

    return _fill_missing(data)


# ── Sample PDF generator (for testing) ───────────────────────────────────────

def make_sample_pdf(
    txt_path: Optional[str] = None, pdf_path: Optional[str] = None
) -> str:
    """Render data/raw/sample_resume.txt to a PDF using fpdf2. Returns pdf path."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    txt_path = txt_path or str(_PROJECT_ROOT / "data" / "raw" / "sample_resume.txt")
    pdf_path = pdf_path or str(_PROJECT_ROOT / "data" / "raw" / "sample_resume.pdf")

    content = Path(txt_path).read_text(encoding="utf-8")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in content.splitlines():
        # latin-1 safe encoding for the core fonts.
        safe = line.encode("latin-1", "replace").decode("latin-1")
        # Collapse long unbreakable runs (e.g. dash separators) so multi_cell
        # can wrap; fpdf2 raises on a single "word" wider than the page.
        safe = re.sub(r"([-=_])\1{3,}", r"\1\1\1", safe)
        if not safe.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 5, safe, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.output(pdf_path)
    return pdf_path


def _test() -> None:
    pdf_path = make_sample_pdf()
    print(f"Sample PDF written to: {pdf_path}\n")
    print("Parsing resume...\n")
    profile = parse_resume(pdf_path)
    print("=" * 65)
    print("  PARSED RESUME JSON")
    print("=" * 65)
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _test()
