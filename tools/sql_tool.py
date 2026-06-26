"""
SQL tool wrapping jobs.db as a LlamaIndex NLSQLTableQueryEngine.

Usage:
    from tools.sql_tool import setup_sql_tool
    tool = setup_sql_tool()
"""

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine

from llama_index.core import SQLDatabase
from llama_index.core.base.response.schema import Response
from llama_index.core.indices.struct_store.sql_retriever import DefaultSQLParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine, NLSQLTableQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _PROJECT_ROOT / "database" / "jobs.db"

# Load .env from the project root so the key resolves regardless of the cwd
# the importing agent runs from.
load_dotenv(_PROJECT_ROOT / ".env")

# ── Custom text-to-SQL prompt ────────────────────────────────────────────────
# Must contain {dialect}, {schema}, {query_str} — same vars as the default.
# CRITICAL: this prompt governs ONLY the SQL-generation step, so it must ask for
# raw SQL and nothing else. Any "human readable / bullet points / never show
# SQL" guidance belongs in the response-synthesis prompt below — putting it here
# makes the model answer conversationally (and hallucinate) instead of emitting
# a query.

_SQL_PROMPT_TMPL = """\
You are an expert {dialect} SQL generator for a job-listings database.
Given a question, output ONE valid {dialect} SQL query that answers it.

SCHEMA NOTES
============
TABLE jobs:
  id, title, company, location (city name or 'Remote'),
  salary_min, salary_max         -- numbers in LPA (Lakhs Per Annum, INR)
  experience_min, experience_max -- years of experience
  skills_required                -- comma-separated skills, search with LIKE '%Skill%'
  industry,
  company_stage                  -- one of: 'seed','series-a','series-b','mnc'
  remote_friendly                -- 1 = remote allowed, 0 = not
  full_description

TABLE companies:
  id, name (matches jobs.company), industry,
  stage  -- 'seed','series-a','series-b','mnc'
  size, tech_stack, culture_summary, founded_year

QUERY-WRITING RULES
===================
- Salary columns are already in LPA; compare directly.
- A role's pay band is [salary_min, salary_max]. Treat a salary filter as
  "the band reaches that region", i.e.:
    "under N LPA"  -> salary_min <= N
    "above N LPA" / "at least N LPA" -> salary_max >= N
    "between A and B LPA" -> salary_min <= B AND salary_max >= A
- Join companies on jobs.company = companies.name when company details
  (size, tech_stack, culture, founded_year) are asked for.
- Match a role family with title LIKE '%Keyword%' (e.g. title LIKE '%ML Engineer%').
- Match a skill with skills_required LIKE '%Skill%'; for "both X and Y",
  use skills_required LIKE '%X%' AND skills_required LIKE '%Y%'.
- For remote roles filter remote_friendly = 1.
- For funding stage filter company_stage = 'series-a' (etc.), lowercase.
- Select only the columns relevant to the question, never SELECT *.

OUTPUT FORMAT — STRICT
======================
Return ONLY the SQL query text on a single line.
Do NOT add explanations, comments, markdown, code fences, or the word "sql".

Only use tables and columns shown below.
{schema}

Question: {query_str}
SQLQuery: \
"""

_SQL_PROMPT = PromptTemplate(_SQL_PROMPT_TMPL)

# ── Custom response-synthesis prompt ─────────────────────────────────────────
# Runs AFTER the SQL executes. Vars: {query_str}, {sql_query}, {context_str}.
# This is where the user-facing formatting rules live.

_RESPONSE_SYNTHESIS_TMPL = """\
You are a hiring assistant. Using ONLY the SQL results below, answer the
user's question in clear natural language.

Rules:
- Base your answer strictly on the SQL Response. Never invent companies,
  roles, salaries, or any data that is not present in the SQL Response.
- Salary numbers are in LPA (Lakhs Per Annum) — always say "LPA".
- If the SQL Response is empty, contains no rows, or is an error, reply
  exactly: "No results found for that query."
- Never show the raw SQL query to the user.
- When there are multiple rows, present them as a concise bullet list.

Question: {query_str}
SQL: {sql_query}
SQL Response: {context_str}
Answer: \
"""

_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(_RESPONSE_SYNTHESIS_TMPL)


def _get_llm() -> OpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Add it to your .env file."
        )
    return OpenRouter(
        api_key=api_key,
        # NOTE: OpenRouter retired the llama-3.1-70b ":free" tier, and the free
        # llama-3.3-70b / qwen3-next endpoints are currently saturated and time
        # out. gpt-oss-120b is the most capable free model that responds
        # reliably (~4s) — a strong open-weight 120B MoE, well-suited to NL->SQL.
        model="openai/gpt-oss-120b:free",
        max_tokens=512,
        temperature=0.1,
    )


# Cache the embedding model so we only load it from disk once per process.
_EMBED_MODEL = None


def _get_embed_model() -> HuggingFaceEmbedding:
    """Local BAAI embedding model — avoids the OpenAI embeddings default."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL


# ── Robustness layer ─────────────────────────────────────────────────────────
# The free gpt-oss model is non-deterministic and occasionally answers in prose
# instead of emitting SQL. Two safeguards keep the tool reliable when that
# happens: (1) a parser that extracts the real SELECT/WITH statement even if the
# model wraps it in prose/markdown, and (2) a retry wrapper that regenerates only
# when no valid SQL was produced — a valid query returning 0 rows is a genuine
# "no results" and is NOT retried.

_SQL_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def _extract_sql(text: str) -> str | None:
    """Pull the first SELECT/WITH statement out of an LLM response, or None."""
    if not text:
        return None
    match = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if not match:
        return None
    sql = text[match.start():].strip()
    # Stop at the first statement terminator if present...
    semi = sql.find(";")
    if semi != -1:
        return sql[:semi].strip()
    # ...otherwise drop any trailing prose separated by a blank line.
    return sql.split("\n\n")[0].strip()


class RobustSQLParser(DefaultSQLParser):
    """SQL parser that recovers a query even when the model adds prose."""

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        cleaned = super().parse_response_to_sql(response, query_bundle)
        if _SQL_START_RE.match(cleaned):
            return _extract_sql(cleaned) or cleaned
        # Default parser left prose in place — try to dig SQL out of the raw text.
        return _extract_sql(response) or cleaned


class RobustSQLQueryEngine(CustomQueryEngine):
    """Wraps NLSQLTableQueryEngine, retrying when SQL generation fails."""

    inner: Any
    max_retries: int = 2

    def custom_query(self, query_str: str) -> Response:
        last: Response | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self.inner.query(query_str)
            except Exception as exc:  # SQL execution / generation error → retry
                last = Response(response=f"No results found for that query. ({exc})")
                continue
            sql = (resp.metadata or {}).get("sql_query", "") or ""
            if _SQL_START_RE.match(sql.strip()):
                # Valid SQL was generated (even if it returned 0 rows) → trust it.
                return resp
            last = resp  # prose / invalid SQL → try again
        # All attempts produced no valid SQL.
        return Response(
            response="No results found for that query.",
            metadata=(last.metadata if last is not None else None),
        )


def setup_sql_tool() -> QueryEngineTool:
    """Return a QueryEngineTool backed by the jobs SQLite database."""
    if not _DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {_DB_PATH}. "
            "Run data/processed/create_database.py first."
        )

    engine = create_engine(f"sqlite:///{_DB_PATH}")
    sql_db = SQLDatabase(engine, include_tables=["jobs", "companies"])

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_db,
        tables=["jobs", "companies"],
        llm=_get_llm(),
        embed_model=_get_embed_model(),
        text_to_sql_prompt=_SQL_PROMPT,
        response_synthesis_prompt=_RESPONSE_SYNTHESIS_PROMPT,
        synthesize_response=True,
        verbose=False,
    )
    # Swap in the robust SQL extractor on the underlying retriever.
    query_engine._sql_retriever._sql_parser = RobustSQLParser()

    robust_engine = RobustSQLQueryEngine(inner=query_engine)

    return QueryEngineTool(
        query_engine=robust_engine,
        metadata=ToolMetadata(
            name="job_database_tool",
            description=(
                "Use this tool for queries about job listings, salary ranges, "
                "required skills, company funding stages, locations, experience "
                "requirements, remote work availability. Contains 50 real job "
                "records across ML, backend, devops, and product roles in "
                "Indian companies."
            ),
        ),
    )


def test_sql_tool() -> None:
    """Run 5 test queries and print results."""
    queries = [
        "Show all ML Engineer roles in Bangalore under 20 LPA",
        "Which companies offer remote work for backend engineers?",
        "What is average salary for Data Scientists with 2-4 years experience?",
        "List all Series A startups hiring in AI/ML",
        "Which roles require both Docker and Kubernetes?",
    ]

    print("\n" + "=" * 65)
    print("  SQL TOOL — TEST RESULTS")
    print("=" * 65)

    try:
        tool = setup_sql_tool()
        engine = tool.query_engine
    except (FileNotFoundError, EnvironmentError) as exc:
        print(f"[SETUP ERROR] {exc}")
        return

    for i, q in enumerate(queries, 1):
        print(f"\n[Query {i}] {q}")
        print("-" * 55)
        try:
            response = engine.query(q)
            print(str(response).strip())
        except Exception as exc:
            print(f"[ERROR] {exc}")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    test_sql_tool()
