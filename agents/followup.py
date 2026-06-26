"""
Follow-up question generator.

Takes the synthesizer's gaps list and produces up to 3 specific, constructive
follow-up questions for the candidate (only for critical/moderate gaps). Each
question carries a hidden "what_good_answer_looks_like" used later by
re_evaluate (not shown to the user).

Usage:
    from agents.followup import generate_followups
    followups = generate_followups(gaps)
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

_MODEL = "openai/gpt-oss-120b:free"
_MAX_QUESTIONS = 3


def _get_llm() -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(api_key=api_key, model=_MODEL,
                      max_tokens=900, temperature=0.3)


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


def _build_prompt(gaps: list, previously_asked: Optional[list] = None) -> str:
    gaps_json = json.dumps(gaps, indent=2, ensure_ascii=False)
    asked_block = ""
    if previously_asked:
        asked_list = "\n".join(f"  - {q}" for q in previously_asked)
        asked_block = (
            "\nThis candidate has answered follow-up questions in PREVIOUS "
            "sessions. Do NOT ask any of these previously asked questions "
            "again (or close paraphrases):\n"
            f"{asked_list}\n"
            "If a gap was already addressed by one of those previous questions, "
            "acknowledge it instead of asking again and do not generate a new "
            "question for it.\n"
        )
    return f"""\
You are a senior technical recruiter writing follow-up questions to give a
candidate a fair chance to address the gaps found in their fit analysis.

You are given the GAPS (each with a severity). Generate follow-up questions
ONLY for gaps whose severity is "critical" or "moderate". Generate AT MOST
{_MAX_QUESTIONS} questions total (pick the most important gaps if there are more).
{asked_block}

Question rules:
- Be SPECIFIC, never generic.
  BAD:  "Do you have ML experience?"
  GOOD: "The role requires deploying models to production. Can you describe a
         specific project where you took a model from notebook to a live serving
         endpoint with monitoring?"
- Frame each question constructively (an invitation to share evidence), not as
  an interrogation.
- "what_good_answer_looks_like" describes the concrete signals a strong answer
  would contain. It is HIDDEN from the candidate and used only for later
  re-evaluation.

Output ONLY a single JSON object with EXACTLY this structure:
{{
  "questions": [
    {{
      "gap_addressed": string,
      "question": string,
      "what_good_answer_looks_like": string
    }}
  ]
}}

GAPS:
{gaps_json}
"""


def generate_followups(gaps: list, previously_asked: Optional[list] = None) -> dict:
    """
    Generate up to 3 specific follow-up questions for critical/moderate gaps.

    previously_asked: questions this candidate already answered in past sessions
    — they will not be asked again.
    """
    relevant = [g for g in (gaps or [])
                if isinstance(g, dict)
                and g.get("severity") in ("critical", "moderate")]
    if not relevant:
        return {"questions": []}

    previously_asked = previously_asked or []
    llm = _get_llm()
    raw = str(llm.complete(_build_prompt(relevant, previously_asked)))
    data = _extract_json(raw)
    if data is None:
        raw = str(llm.complete(
            _build_prompt(relevant, previously_asked) +
            "\n\nIMPORTANT: respond with ONLY raw JSON, no markdown."
        ))
        data = _extract_json(raw)

    # Defensive: drop any question that matches a previously asked one.
    asked_norm = {q.strip().lower() for q in previously_asked}

    questions = []
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        for q in data["questions"]:
            if not isinstance(q, dict):
                continue
            entry = {
                "gap_addressed": str(q.get("gap_addressed", "")).strip(),
                "question": str(q.get("question", "")).strip(),
                "what_good_answer_looks_like":
                    str(q.get("what_good_answer_looks_like", "")).strip(),
            }
            if entry["question"] and entry["question"].lower() not in asked_norm:
                questions.append(entry)

    return {"questions": questions[:_MAX_QUESTIONS]}
