"""
Reflector / re-evaluation agent.

Two public functions:

  reflect()     — in-workflow quality check. Reads the synthesis output,
                  checks every claim against available SQL/RAG evidence,
                  and decides whether to replan (cycle < 2 AND confidence < 70).

  re_evaluate() — post-workflow answer scorer. Called after the candidate
                  answers follow-up questions; adjusts fit score based on
                  answer quality vs hidden criteria.

Usage:
    from agents.reflector import reflect, re_evaluate
    # In-workflow quality check:
    reflection = reflect(synthesis, sql_results, rag_results, plan, cycle_number=1)
    # Post-answer re-evaluation:
    updated = re_evaluate(original_synthesis, followup_qa, followup_questions)
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

# Reuse the shared score helpers from the synthesizer.
from agents.synthesizer import clamp_score, recommendation_from_score  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

_MODEL = "openai/gpt-oss-120b:free"

# ── In-workflow quality reflection ────────────────────────────────────────────

_REFLECT_SCHEMA = """{
  "confidence": <int 0-100>,
  "weak_points": [
    {
      "claim": "<the exact claim in the synthesis that is questionable>",
      "issue": "no_evidence" | "contradictory_evidence" | "low_confidence",
      "suggested_query": "<a specific SQL or RAG query to fill this gap>"
    }
  ],
  "replan_queries": {
    "sql": ["<query 1>", "<query 2>"],
    "rag": ["<query 1>", "<query 2>"]
  },
  "reflection_notes": "<1-2 sentence summary of your quality assessment>"
}"""

_REFLECT_PROMPT = """\
You are a quality control agent checking an AI hiring analysis before it is
shown to a candidate. Your job: verify that every factual claim in the
synthesis is actually supported by the evidence in the tool results.

CHECK THESE FIVE THINGS:
1. Does every strength that cites source="sql" have matching content in SQL RESULTS?
2. Does every strength or gap that cites source="rag" have matching content in RAG RESULTS?
3. Is every gap backed by concrete evidence (not just assumed from the JD)?
4. Is the fit_score numerically consistent with the evidence quality?
5. Are there any contradictions between SQL results and RAG results that the
   synthesis failed to address?

CONFIDENCE SCORING GUIDE:
  90-100 : All claims strongly evidenced and internally consistent.
  70-89  : Most claims evidenced; a few minor gaps are acceptable.
  50-69  : Some claims lack clear evidence — replan RECOMMENDED.
  0-49   : Significant evidence gaps or contradictions — replan REQUIRED.

NOTE: A confidence below 70 on cycle 1 means you should add targeted
replan_queries (2-3 specific SQL and/or RAG queries that would fill the gaps).
Leave replan_queries empty lists if confidence >= 70.

Output ONLY a JSON object matching this schema (no markdown, no prose):
{schema}

SYNTHESIS TO VERIFY:
{synthesis_json}

AVAILABLE SQL RESULTS (evidence from job database):
{sql_block}

AVAILABLE RAG RESULTS (evidence from culture docs):
{rag_block}

INVESTIGATION PLAN (what was originally asked):
  SQL queries: {sql_queries}
  RAG queries: {rag_queries}
"""


def reflect(
    synthesis: dict,
    sql_results: list,
    rag_results: list,
    investigation_plan: dict,
    cycle_number: int = 1,
) -> dict:
    """
    Quality-check the synthesis against the available evidence.

    Parameters
    ----------
    synthesis         : The synthesize_fit() output dict.
    sql_results       : List of SQL tool result strings.
    rag_results       : List of RAG tool result strings.
    investigation_plan: The planner output (sql_queries, rag_queries, …).
    cycle_number      : 1 on the first call; 2 on the replan call.

    Returns
    -------
    dict:
        confidence    : int 0-100
        should_replan : bool  (True only when confidence < 70 AND cycle < 2)
        weak_points   : list of {claim, issue, suggested_query}
        replan_queries: {sql: [...], rag: [...]}
        reflection_notes: str
    """

    def _fmt_results(results: list) -> str:
        if not results:
            return "(none)"
        lines = []
        for i, r in enumerate(results, 1):
            snippet = str(r).strip().replace("\n", " ")[:300]
            lines.append(f"  [{i}] {snippet}")
        return "\n".join(lines)

    prompt = _REFLECT_PROMPT.format(
        schema=_REFLECT_SCHEMA,
        synthesis_json=json.dumps({
            "fit_score": synthesis.get("fit_score"),
            "recommendation": synthesis.get("recommendation"),
            "strengths": synthesis.get("strengths", []),
            "gaps": synthesis.get("gaps", []),
            "summary": synthesis.get("summary", "")[:400],
            "confidence": synthesis.get("confidence"),
        }, indent=2),
        sql_block=_fmt_results(sql_results),
        rag_block=_fmt_results(rag_results),
        sql_queries=json.dumps(investigation_plan.get("sql_queries", [])),
        rag_queries=json.dumps(investigation_plan.get("rag_queries", [])),
    )

    llm = OpenRouter(
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model=_MODEL,
        max_tokens=800,
        temperature=0.0,
    )

    raw = str(llm.complete(prompt)).strip()

    # Parse JSON
    try:
        text = re.sub(r"^```(?:json)?", "", raw).strip()
        text = re.sub(r"```$", "", text).strip()
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        try:
            data = json.loads(raw[start:end + 1]) if start != -1 else {}
        except Exception:
            data = {}

    confidence = max(0, min(100, int(data.get("confidence", 85))))
    weak_points = data.get("weak_points", []) or []
    replan_q = data.get("replan_queries", {}) or {}
    notes = str(data.get("reflection_notes", "")).strip()

    # Programmatic rule: should_replan = confidence < 70 AND cycle_number < 2
    should_replan = confidence < 70 and cycle_number < 2

    return {
        "confidence": confidence,
        "should_replan": should_replan,
        "weak_points": weak_points,
        "replan_queries": {
            "sql": replan_q.get("sql", []) if should_replan else [],
            "rag": replan_q.get("rag", []) if should_replan else [],
        },
        "reflection_notes": notes,
    }

# Score increments by (rating, gap severity). Weak answers earn nothing.
_DELTA = {
    ("strong", "critical"): 10,   # range 8-12
    ("adequate", "critical"): 4,  # range 3-5
    ("strong", "moderate"): 4,    # range 3-5
    ("adequate", "moderate"): 2,
    ("strong", "minor"): 2,
    ("adequate", "minor"): 1,
}


def _get_llm() -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(api_key=api_key, model=_MODEL,
                      max_tokens=700, temperature=0.0)


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


def _severity_for_gap(gap_text: str, gaps: list) -> str:
    """Find the severity of the gap a follow-up question addresses."""
    gap_text_low = (gap_text or "").lower()
    for g in gaps or []:
        point = str(g.get("point", "")).lower()
        if point and (point in gap_text_low or gap_text_low in point
                      or point[:30] == gap_text_low[:30]):
            return g.get("severity", "moderate")
    return "moderate"


def _build_scoring_prompt(scored_items: list) -> str:
    """scored_items: list of {question, criteria, answer}."""
    items_json = json.dumps(
        [{"index": i, "question": it["question"],
          "what_good_answer_looks_like": it["criteria"],
          "candidate_answer": it["answer"]}
         for i, it in enumerate(scored_items)],
        indent=2, ensure_ascii=False)
    return f"""\
You are evaluating a candidate's answers to follow-up interview questions.

For EACH item, compare the candidate_answer against
what_good_answer_looks_like and rate how well it meets that bar:
- "strong"   : clearly meets the bar with specific, credible evidence.
- "adequate" : partially meets it; some relevant signal but shallow or incomplete.
- "weak"     : does not meet it; vague, generic, or off-target.

Output ONLY a single JSON object with EXACTLY this structure:
{{
  "ratings": [
    {{"index": int, "rating": "strong"|"adequate"|"weak"}}
  ]
}}

ITEMS:
{items_json}
"""


def re_evaluate(
    original_synthesis: dict,
    followup_qa: list,
    followup_questions_with_criteria: dict,
) -> dict:
    """
    Re-evaluate the fit analysis given the candidate's follow-up answers.

    Returns a complete updated synthesis dict with updated fit_score,
    score_delta, resolved_gaps, remaining_gaps, updated recommendation, and
    per-answer assessments.
    """
    updated = json.loads(json.dumps(original_synthesis))  # deep copy
    criteria_questions = (followup_questions_with_criteria or {}).get("questions", [])
    gaps = updated.get("gaps", [])

    # Match each answered question to its criteria + the gap it addresses.
    def _match(question: str):
        for cq in criteria_questions:
            if cq.get("question", "").strip() == (question or "").strip():
                return cq
        return None

    scored_items = []
    for i, qa in enumerate(followup_qa or []):
        question = qa.get("question", "")
        cq = _match(question)
        if cq is None and i < len(criteria_questions):
            cq = criteria_questions[i]  # fall back to positional match
        if cq is None:
            continue
        gap_text = cq.get("gap_addressed", "")
        scored_items.append({
            "question": question,
            "answer": qa.get("answer", ""),
            "criteria": cq.get("what_good_answer_looks_like", ""),
            "gap_addressed": gap_text,
            "severity": _severity_for_gap(gap_text, gaps),
        })

    # Ask the LLM to rate each answer.
    ratings = {}
    if scored_items:
        llm = _get_llm()
        raw = str(llm.complete(_build_scoring_prompt(scored_items)))
        data = _extract_json(raw)
        if isinstance(data, dict):
            for r in data.get("ratings", []) or []:
                if isinstance(r, dict) and "index" in r:
                    ratings[int(r["index"])] = r.get("rating", "weak")

    # Apply deltas, collect assessments and resolved gaps.
    total_delta = 0
    assessments = []
    resolved_points = set()
    for i, it in enumerate(scored_items):
        rating = ratings.get(i, "weak")
        delta = _DELTA.get((rating, it["severity"]), 0)
        total_delta += delta
        if rating == "strong":
            resolved_points.add(it["gap_addressed"])
        assessments.append({
            "question": it["question"],
            "gap_addressed": it["gap_addressed"],
            "severity": it["severity"],
            "rating": rating,
            "score_delta": delta,
        })

    # Update scores and recommendation.
    original_score = clamp_score(updated.get("fit_score", 0))
    new_score = clamp_score(original_score + total_delta)

    resolved_gaps, remaining_gaps = [], []
    for g in gaps:
        point = g.get("point", "")
        is_resolved = any(point and (point in rp or rp in point or point == rp)
                          for rp in resolved_points)
        g["resolved"] = bool(is_resolved)
        (resolved_gaps if is_resolved else remaining_gaps).append(g)

    updated["fit_score"] = new_score
    updated["score_delta"] = new_score - original_score
    updated["recommendation"] = recommendation_from_score(new_score)
    updated["resolved_gaps"] = resolved_gaps
    updated["remaining_gaps"] = remaining_gaps
    updated["answer_assessments"] = assessments
    return updated
