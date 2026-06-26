"""
Synthesizer agent.

Takes the resume, JD, SQL + RAG tool results, the investigation plan and the
session memory, and produces a strict-JSON fit analysis (score, breakdown,
strengths/gaps with citations, recommendation, alternatives, confidence).

Usage:
    from agents.synthesizer import synthesize_fit
    analysis = synthesize_fit(inputs)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Make the project root importable whether run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.llms.openrouter import OpenRouter  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# Same free model as the rest of the project (see project memory).
_MODEL = "openai/gpt-oss-120b:free"

_VALID_RECOMMENDATIONS = {"strong_fit", "possible_fit", "weak_fit", "not_fit"}
_VALID_SEVERITIES = {"critical", "moderate", "minor"}
_VALID_SOURCES = {"sql", "rag", "resume"}


def _get_llm(max_tokens: int = 1800) -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(api_key=api_key, model=_MODEL,
                      max_tokens=max_tokens, temperature=0.1)


# ── Shared score/recommendation helpers (reused by reflector.re_evaluate) ─────

def clamp_score(x, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except (TypeError, ValueError):
        return lo


def recommendation_from_score(score: int) -> str:
    if score >= 85:
        return "strong_fit"
    if score >= 65:
        return "possible_fit"
    if score >= 45:
        return "weak_fit"
    return "not_fit"


def _jd_seniority(jd: str) -> str:
    """Rough seniority of the role from the JD text (fresher/junior/mid/senior)."""
    low = (jd or "").lower()
    if any(k in low for k in ("senior", "lead", "staff", "principal", "sr.")):
        return "senior"
    # Look for a minimum years figure like "3-5 years" or "5+ years".
    nums = [int(n) for n in re.findall(r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?years?", low)]
    if nums:
        lo = min(nums)
        if lo >= 6:
            return "senior"
        if lo >= 3:
            return "mid"
        if lo >= 1:
            return "junior"
        return "fresher"
    return "mid"


# Hard caps (safety net on top of the prompt's calibration rules).
_SCORE_CAPS = {
    ("fresher", "senior"): 45,
    ("fresher", "mid"): 45,
    ("junior", "mid"): 65,
    ("junior", "senior"): 45,
}


def _apply_calibration_cap(score: int, resume_level: str, jd_level: str) -> int:
    cap = _SCORE_CAPS.get((resume_level, jd_level))
    return min(score, cap) if cap is not None else score


# ── Prompt + schema ──────────────────────────────────────────────────────────

_SCHEMA = """\
{
  "fit_score": int (0-100),
  "score_breakdown": {
    "skills_match": int (0-100),
    "experience_match": int (0-100),
    "culture_fit": int (0-100)
  },
  "strengths": [
    {"point": string, "evidence": string, "source": "sql"|"rag"|"resume"}
  ],
  "gaps": [
    {"point": string, "severity": "critical"|"moderate"|"minor",
     "evidence": string, "source": "sql"|"rag"|"resume"}
  ],
  "summary": string (2-3 sentences),
  "recommendation": "strong_fit"|"possible_fit"|"weak_fit"|"not_fit",
  "alternative_roles": [string],
  "citations": {"from_sql": [string], "from_rag": [string]},
  "confidence": int (0-100)
}"""


def _build_prompt(inputs: dict, strict: bool = False) -> str:
    resume_json = json.dumps(inputs.get("resume", {}), indent=2, ensure_ascii=False)
    sql_block = "\n".join(f"- {r}" for r in inputs.get("sql_results", [])) or "(none)"
    rag_block = "\n".join(f"- {r}" for r in inputs.get("rag_results", [])) or "(none)"
    plan_json = json.dumps(inputs.get("investigation_plan", {}), ensure_ascii=False)
    memory = inputs.get("memory_context", "") or "(none)"
    strict_note = ("\nIMPORTANT: respond with ONLY raw JSON, no markdown, no "
                   "explanation.\n" if strict else "")

    return f"""\
You are a senior technical recruiter producing a rigorous, evidence-based fit
analysis of a candidate for a specific role.

Output ONLY a single JSON object with EXACTLY this structure:
{_SCHEMA}

EVIDENCE & HONESTY RULES:
- Use ONLY the provided resume, SQL results and RAG results as evidence. Do NOT
  invent skills, experience, companies, or numbers not present in them.
- Every strength must cite which source supports it ("sql", "rag", or "resume")
  and give concrete evidence.
- Every gap must cite specific evidence and a severity
  ("critical"|"moderate"|"minor").
- citations.from_sql / from_rag must quote or closely paraphrase the actual
  tool results you relied on.
- alternative_roles must be realistic given the candidate's actual resume.

SCORE CALIBRATION (be strict and honest):
- A fresher (only internships, <1 yr) applying to a SENIOR role: fit_score <= 45.
- A junior (~1-3 yrs) applying to a MID-level role: fit_score <= 65.
- A genuinely good match: 70-85.
- An excellent match: 85+.
- experience_match must reflect the gap between required and actual years.

CANDIDATE RESUME (parsed JSON):
{resume_json}

JOB DESCRIPTION:
\"\"\"
{inputs.get("job_description", "")}
\"\"\"

INVESTIGATION PLAN:
{plan_json}

SQL JOB-DATABASE RESULTS:
{sql_block}

CULTURE / CONTEXT (RAG) RESULTS:
{rag_block}

SESSION MEMORY (for continuity):
{memory}
{strict_note}"""


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _empty_synthesis() -> dict:
    return {
        "fit_score": 0,
        "score_breakdown": {"skills_match": 0, "experience_match": 0,
                            "culture_fit": 0},
        "strengths": [],
        "gaps": [],
        "summary": "Unable to synthesize a fit analysis.",
        "recommendation": "not_fit",
        "alternative_roles": [],
        "citations": {"from_sql": [], "from_rag": []},
        "confidence": 0,
    }


def _validate(data: Optional[dict], inputs: dict) -> dict:
    out = _empty_synthesis()
    if not isinstance(data, dict):
        return out

    out["fit_score"] = clamp_score(data.get("fit_score", 0))

    bd = data.get("score_breakdown", {})
    if isinstance(bd, dict):
        for k in ("skills_match", "experience_match", "culture_fit"):
            out["score_breakdown"][k] = clamp_score(bd.get(k, 0))

    def clean_items(items, severity=False):
        cleaned = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            entry = {"point": str(it.get("point", "")).strip(),
                     "evidence": str(it.get("evidence", "")).strip()}
            src = it.get("source")
            entry["source"] = src if src in _VALID_SOURCES else "resume"
            if severity:
                sev = it.get("severity")
                entry["severity"] = sev if sev in _VALID_SEVERITIES else "moderate"
            if entry["point"]:
                cleaned.append(entry)
        return cleaned

    out["strengths"] = clean_items(data.get("strengths"))
    out["gaps"] = clean_items(data.get("gaps"), severity=True)

    if isinstance(data.get("summary"), str):
        out["summary"] = data["summary"].strip()
    if isinstance(data.get("alternative_roles"), list):
        out["alternative_roles"] = [str(x) for x in data["alternative_roles"]]
    cit = data.get("citations", {})
    if isinstance(cit, dict):
        for k in ("from_sql", "from_rag"):
            v = cit.get(k)
            if isinstance(v, list):
                out["citations"][k] = [str(x) for x in v]
    out["confidence"] = clamp_score(data.get("confidence", 50))

    # Calibration safety net: cap the score for clear seniority mismatches.
    resume_level = (inputs.get("resume", {}) or {}).get("experience_level", "")
    jd_level = _jd_seniority(inputs.get("job_description", ""))
    capped = _apply_calibration_cap(out["fit_score"], resume_level, jd_level)
    if capped != out["fit_score"]:
        out["fit_score"] = capped
        out["score_breakdown"]["experience_match"] = min(
            out["score_breakdown"]["experience_match"], capped
        )

    # Keep recommendation consistent with the (possibly capped) score.
    rec = data.get("recommendation")
    score_rec = recommendation_from_score(out["fit_score"])
    if rec in _VALID_RECOMMENDATIONS:
        # Never report a more optimistic recommendation than the score supports.
        order = ["not_fit", "weak_fit", "possible_fit", "strong_fit"]
        out["recommendation"] = min(rec, score_rec, key=order.index)
    else:
        out["recommendation"] = score_rec

    return out


def synthesize_fit(inputs: dict) -> dict:
    """Produce a strict-JSON fit analysis from the synthesizer inputs dict."""
    llm = _get_llm()
    raw = str(llm.complete(_build_prompt(inputs)))
    data = _extract_json(raw)
    if data is None:
        raw = str(llm.complete(_build_prompt(inputs, strict=True)))
        data = _extract_json(raw)
    if data is None:
        print("[synthesizer] WARNING: could not parse a valid analysis; "
              "returning an empty one.")
    return _validate(data, inputs)
