"""
Planner agent.

Takes a parsed resume + a job description and produces an investigation PLAN
(which queries to run, what to evaluate, hypothesised gaps) BEFORE any tools are
called. Planning-first is what makes the system agentic.

Usage:
    from agents.planner import plan_investigation
    plan = plan_investigation(resume_dict, job_description)
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

_VALID_PRIORITIES = {"sql_first", "rag_first", "parallel"}


def _get_planner_llm() -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(
        api_key=api_key,
        model=_MODEL,
        max_tokens=1024,
        temperature=0.2,
    )


_PLAN_SCHEMA = """\
{
  "sql_queries": ["2-3 specific natural-language queries for the SQL job database"],
  "rag_queries": ["2-3 specific queries for culture/context documents"],
  "key_skills_to_evaluate": ["specific skills from the JD to check against the resume"],
  "initial_gap_hypothesis": ["preliminary gaps spotted before tool results"],
  "hybrid_questions": ["questions needing BOTH SQL data AND culture context"],
  "investigation_priority": "sql_first" | "rag_first" | "parallel"
}"""


def _build_prompt(resume_json: str, job_description: str,
                  classification: Optional[dict] = None) -> str:
    clf_block = ""
    if classification:
        clf_type = classification.get("type", "HYBRID")
        sql_w = classification.get("sql_weight", 0.5)
        rag_w = classification.get("rag_weight", 0.5)
        reasoning = classification.get("reasoning", "")
        if clf_type == "SQL_ONLY":
            priority_hint = "sql_first"
            query_hint = "Focus almost entirely on SQL queries (3 SQL, 1 RAG maximum)."
        elif clf_type == "RAG_ONLY":
            priority_hint = "rag_first"
            query_hint = "Focus almost entirely on RAG queries (1 SQL, 3 RAG maximum)."
        else:
            priority_hint = "parallel"
            if sql_w >= 0.6:
                query_hint = f"Lean toward SQL queries (sql_weight={sql_w}). Aim for 3 SQL and 2 RAG queries."
            elif rag_w >= 0.6:
                query_hint = f"Lean toward RAG queries (rag_weight={rag_w}). Aim for 2 SQL and 3 RAG queries."
            else:
                query_hint = "Balance SQL and RAG equally (2-3 each)."
        clf_block = (
            f"\nCLASSIFICATION GUIDANCE (from QueryClassifier):\n"
            f"  Type              : {clf_type}\n"
            f"  SQL weight        : {sql_w}  |  RAG weight: {rag_w}\n"
            f"  Classifier reason : {reasoning}\n"
            f"  Query guidance    : {query_hint}\n"
            f"  Set investigation_priority to: '{priority_hint}'\n"
        )
    return f"""\
You are a senior technical recruiter planning how to investigate whether a
candidate fits a role. Think before acting.

First, read the candidate resume and the job description FULLY. Then produce an
investigation plan. Do NOT evaluate fit yet — only plan what to investigate.
{clf_block}
When planning:
- Identify the key skills required by the JD and which appear MISSING from the
  resume.
- Note any experience-level or years-of-experience mismatch between JD and resume.
- Design SQL queries that surface comparable roles, salary ranges, skill demand,
  and market context relevant to this candidate and JD.
- Design RAG queries about company culture, what skills really mean in practice,
  interview expectations, and implicit JD requirements.
- Plan queries that surface BOTH the candidate's strengths AND their gaps.
- hybrid_questions are questions that need both the job database AND culture
  context to answer well.
- Think about what a senior recruiter would want independently verified.

About the SQL job database (for writing sql_queries):
- It contains JOB LISTINGS, not candidates or salary tables. Columns available:
  title, company, location, salary range (in LPA), experience required, skills
  required, industry, company funding stage (seed/series-a/series-b/mnc), remote
  availability; plus a companies table (industry, stage, size, tech stack,
  culture summary, founded year).
- Write each sql_query as a PLAIN-ENGLISH question (NOT SQL code, no SELECT
  statements). Example: "What ML Engineer roles in Bangalore at Series A
  companies require PyTorch and Docker?"

Output ONLY a single JSON object with EXACTLY this structure (no markdown, no
explanation):
{_PLAN_SCHEMA}

CANDIDATE RESUME (parsed JSON):
{resume_json}

JOB DESCRIPTION:
\"\"\"
{job_description}
\"\"\"
"""


def _extract_json(raw: str) -> Optional[dict]:
    """Parse a JSON object from an LLM response, tolerating fences/prose."""
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


def _empty_plan() -> dict:
    return {
        "sql_queries": [],
        "rag_queries": [],
        "key_skills_to_evaluate": [],
        "initial_gap_hypothesis": [],
        "hybrid_questions": [],
        "investigation_priority": "parallel",
    }


def _validate_plan(data: Optional[dict]) -> dict:
    """Ensure every plan field exists with the right type; fill gaps safely."""
    plan = _empty_plan()
    if not isinstance(data, dict):
        return plan
    for key in ("sql_queries", "rag_queries", "key_skills_to_evaluate",
                "initial_gap_hypothesis", "hybrid_questions"):
        val = data.get(key)
        if isinstance(val, list):
            plan[key] = [str(x) for x in val]
    priority = data.get("investigation_priority")
    if priority in _VALID_PRIORITIES:
        plan["investigation_priority"] = priority
    return plan


def plan_investigation(
    resume_dict: dict,
    job_description: str,
    classification: Optional[dict] = None,
) -> dict:
    """
    Produce an investigation plan (dict) from a parsed resume + a JD.

    classification: optional output from QueryClassifier.classify_hiring_context().
    When provided, the planner uses the type and weights to generate the right
    balance of SQL vs RAG queries and set investigation_priority correctly.

    Asks the LLM for strict JSON and retries once with a stricter prompt if the
    first parse fails. Always returns a fully-formed plan dict (never raises on
    bad LLM output).
    """
    resume_json = json.dumps(resume_dict, indent=2, ensure_ascii=False)
    llm = _get_planner_llm()

    raw = str(llm.complete(_build_prompt(resume_json, job_description, classification)))
    data = _extract_json(raw)

    if data is None:
        # Retry once with a stricter instruction.
        strict = _build_prompt(resume_json, job_description, classification) + (
            "\n\nIMPORTANT: respond with ONLY raw JSON, no markdown, no "
            "explanation."
        )
        raw = str(llm.complete(strict))
        data = _extract_json(raw)

    if data is None:
        print("[planner] WARNING: could not parse a valid plan from the LLM; "
              "returning an empty plan.")

    return _validate_plan(data)


if __name__ == "__main__":
    # Quick standalone smoke test with a tiny resume stub.
    demo_resume = {
        "name": "Demo Candidate",
        "skills": {"programming_languages": ["Python"],
                   "frameworks_and_libraries": ["PyTorch"]},
        "total_experience_years": 0.42,
        "experience_level": "fresher",
        "target_roles": ["ML Engineer"],
    }
    demo_jd = (
        "ML Engineer at a Series A fintech startup in Bangalore. 3-5 years "
        "experience. Required: Python, PyTorch, MLflow, Docker, REST APIs."
    )
    print(json.dumps(plan_investigation(demo_resume, demo_jd), indent=2))
