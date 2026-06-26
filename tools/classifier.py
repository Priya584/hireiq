"""
QueryClassifier — replaces LlamaIndex's default router with a project-
controlled prompt-driven classifier.

Each hiring-analysis query is classified into one of three categories:
  SQL_ONLY  — structured job data (salary, skills, experience, location …)
  RAG_ONLY  — culture/context documents (expectations, interview style …)
  HYBRID    — needs both tools (full fit assessment, salary-in-culture context …)

The workflow calls classify_step (before planning) to get tool weights that
guide the planner: a sql_weight=0.7 means "generate more SQL queries than RAG".

Usage:
    from tools.classifier import QueryClassifier
    clf = QueryClassifier()
    result = clf.classify(query_string)
    # → {"type": "HYBRID", "confidence": 82,
    #    "reasoning": "...", "sql_weight": 0.6, "rag_weight": 0.4}
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama_index.llms.openrouter import OpenRouter  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

_MODEL = "openai/gpt-oss-120b:free"

_OUTPUT_SCHEMA = """{
  "type": "SQL_ONLY" | "RAG_ONLY" | "HYBRID",
  "confidence": <int 0-100>,
  "reasoning": "<1-2 sentence explanation>",
  "sql_weight": <float 0.0-1.0>,
  "rag_weight": <float 0.0-1.0>
}"""

_PROMPT_TEMPLATE = """\
You are a query router for an AI hiring assistant with two tools:

TOOL 1 — SQL JOB DATABASE (structured data):
  Columns: job title, company name, location, salary_min/max (LPA),
  experience_min/max (years), skills_required, industry, company funding
  stage (seed/series-a/series-b/mnc), remote_friendly, company size,
  tech_stack, founding year.
  Best for: salary figures, years of experience required, specific skill
  lookups, location/stage filtering, job counts, direct comparisons of
  structured attributes, market benchmarks.

TOOL 2 — RAG CULTURE DOCUMENTS (35 unstructured docs):
  Covers: company culture by funding stage, what skills mean in daily
  practice, interview processes & style, implicit JD expectations, team
  dynamics, growth & mentorship opportunities, work environment, soft
  skills in technical roles, onboarding expectations.
  Best for: culture fit, what companies truly value beyond the spec,
  interview prep, implicit requirements, growth paths, work style.

CLASSIFY the query below into exactly one category:

  SQL_ONLY  — Query answered entirely by structured job data.
    Signals: specific numbers (salary, years, headcount), location/stage
    filters, skill requirement lookup, counting/ranking structured fields,
    direct factual lookups about listed attributes.
    Examples:
      "What salary does this role pay?"
      "How many ML Engineer roles require Kubernetes?"
      "Which companies hire remote data scientists?"

  RAG_ONLY  — Query answered entirely by culture/context docs.
    Signals: company culture, what a role/skill means in practice,
    interview style, team dynamics, implicit expectations, growth paths,
    soft skills context, work environment.
    Examples:
      "What does a Series A startup really expect beyond the tech stack?"
      "How is the interview process different at MNCs vs startups?"
      "What does production ML experience actually mean day-to-day?"

  HYBRID    — Query needs BOTH structured data AND culture context.
    Signals: holistic fit assessment, salary evaluated in culture context,
    comparing candidate suitability across technical AND cultural dimensions,
    questions combining market data with expectations.
    Examples:
      "Is this candidate's background a good fit for this role's culture?"
      "Is the salary fair given the startup stage and pace?"
      "Does this candidate meet BOTH the technical AND cultural bar?"

RULES:
- If confidence in SQL_ONLY or RAG_ONLY is below 70 → use HYBRID instead.
- sql_weight + rag_weight MUST equal exactly 1.0.
- SQL_ONLY  → sql_weight=1.0, rag_weight=0.0
- RAG_ONLY  → sql_weight=0.0, rag_weight=1.0
- HYBRID    → set weights reflecting which tool contributes more to the answer.
  Example: salary-heavy + slight culture context → sql_weight=0.7, rag_weight=0.3

Output ONLY a JSON object matching this schema (no markdown, no prose):
{schema}

QUERY:
\"\"\"{query}\"\"\"
"""


class QueryClassifier:
    """
    Classifies a hiring-analysis query into SQL_ONLY, RAG_ONLY, or HYBRID
    and returns confidence + tool weights for the planner.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = _MODEL):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model

    def _get_llm(self) -> OpenRouter:
        if not self._api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Add it to your .env file."
            )
        return OpenRouter(
            api_key=self._api_key,
            model=self._model,
            max_tokens=300,
            temperature=0.0,
        )

    @staticmethod
    def _parse(raw: str) -> dict:
        """Extract and parse the JSON object from LLM output."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start : end + 1])
        raise ValueError(f"Could not parse JSON from classifier output:\n{raw}")

    @staticmethod
    def _validate_and_fix(data: dict) -> dict:
        """Apply classification rules and normalise weights."""
        valid_types = {"SQL_ONLY", "RAG_ONLY", "HYBRID"}
        q_type = str(data.get("type", "HYBRID")).upper()
        if q_type not in valid_types:
            q_type = "HYBRID"

        confidence = max(0, min(100, int(data.get("confidence", 50))))

        # Rule: low-confidence SQL_ONLY or RAG_ONLY → HYBRID
        if q_type != "HYBRID" and confidence < 70:
            q_type = "HYBRID"
            confidence = confidence  # keep reported confidence

        sql_w = float(data.get("sql_weight", 0.5))
        rag_w = float(data.get("rag_weight", 0.5))

        # Force correct weights for single-tool types
        if q_type == "SQL_ONLY":
            sql_w, rag_w = 1.0, 0.0
        elif q_type == "RAG_ONLY":
            sql_w, rag_w = 0.0, 1.0
        else:
            # Normalise so they sum to 1.0
            total = sql_w + rag_w
            if total > 0:
                sql_w = round(sql_w / total, 2)
                rag_w = round(1.0 - sql_w, 2)
            else:
                sql_w, rag_w = 0.5, 0.5

        return {
            "type": q_type,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")).strip(),
            "sql_weight": sql_w,
            "rag_weight": rag_w,
        }

    def classify(self, query: str) -> dict:
        """
        Classify a query and return the routing decision dict.

        Returns
        -------
        dict with keys:
            type        : "SQL_ONLY" | "RAG_ONLY" | "HYBRID"
            confidence  : int 0-100
            reasoning   : str
            sql_weight  : float (0.0-1.0)
            rag_weight  : float (0.0-1.0)
        """
        prompt = _PROMPT_TEMPLATE.format(schema=_OUTPUT_SCHEMA, query=query)
        llm = self._get_llm()
        raw = str(llm.complete(prompt)).strip()
        try:
            data = self._parse(raw)
            return self._validate_and_fix(data)
        except Exception as exc:
            # Fallback: return safe HYBRID with equal weights
            return {
                "type": "HYBRID",
                "confidence": 50,
                "reasoning": f"Classifier error — defaulting to HYBRID. ({exc})",
                "sql_weight": 0.5,
                "rag_weight": 0.5,
            }

    def classify_hiring_context(self, resume_dict: dict, job_description: str) -> dict:
        """
        Classify the overall hiring investigation context (resume + JD) to
        determine how the planner should weight SQL vs RAG queries.
        """
        skills = resume_dict.get("skills", {}) or {}
        flat_skills = []
        for group in ("programming_languages", "frameworks_and_libraries",
                      "tools_and_platforms"):
            flat_skills.extend(skills.get(group, []) or [])
        exp_level = resume_dict.get("experience_level", "unknown")
        exp_years = resume_dict.get("total_experience_years", 0)

        # Build a concise meta-query describing the investigation
        meta_query = (
            f"Assess the overall hiring fit: a {exp_level} candidate "
            f"({exp_years} years, skills: {', '.join(flat_skills[:8])}) "
            f"applying to this role — {str(job_description).strip()[:300]}"
        )
        return self.classify(meta_query)


# ── Standalone test ───────────────────────────────────────────────────────────

_TEST_QUERIES = [
    ("SQL_ONLY",  "What is the salary range for ML Engineer roles at Series A companies in Bangalore?"),
    ("SQL_ONLY",  "How many job listings in the database require Kubernetes experience?"),
    ("SQL_ONLY",  "Which companies offer remote-friendly ML Engineer positions with 3-5 years experience?"),
    ("SQL_ONLY",  "What is the average minimum experience required for Data Scientist roles?"),
    ("RAG_ONLY",  "What does a Series A fintech startup really expect from an ML Engineer beyond the listed technical requirements?"),
    ("RAG_ONLY",  "How does the interview process at MNC companies differ from early-stage startups?"),
    ("RAG_ONLY",  "What does production ML deployment experience actually mean in day-to-day work?"),
    ("HYBRID",   "Is this candidate's background a strong cultural and technical fit for this role at a Series A company?"),
    ("HYBRID",   "Is the salary range for this role competitive given the pace and culture of the startup?"),
    ("HYBRID",   "Would someone with strong ML research experience but no startup exposure fit both the technical and cultural requirements of this role?"),
]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    clf = QueryClassifier()
    print("\n" + "=" * 70)
    print("  QUERY CLASSIFIER — 10-QUERY TEST")
    print("=" * 70)

    correct = 0
    for i, (expected, query) in enumerate(_TEST_QUERIES, 1):
        print(f"\n[{i:2d}] Expected: {expected}")
        print(f"     Query  : {query[:80]}...")
        result = clf.classify(query)
        match = "OK" if result["type"] == expected else "!!"
        print(f"     Result : [{match}] {result['type']} "
              f"(conf={result['confidence']}%, "
              f"sql={result['sql_weight']}, rag={result['rag_weight']})")
        print(f"     Reason : {result['reasoning'][:100]}")
        if result["type"] == expected:
            correct += 1

    print(f"\n{'=' * 70}")
    print(f"  Classification Accuracy: {correct}/{len(_TEST_QUERIES)}")
    print(f"{'=' * 70}\n")
