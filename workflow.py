"""
RouterOutputAgentWorkflow — the main orchestration for the Hiring Co-pilot.

Pipeline:
    plan  ->  execute tools (SQL + RAG, in parallel)  ->  synthesize
          ->  reflect (optional replan loop, max 2)    ->  follow-up  ->  done

Synthesizer, reflector and follow-up generator are PLACEHOLDERS here; they are
replaced by their dedicated agents in later prompts. The orchestration, planning,
parallel tool execution, reflection loop and context bookkeeping are real.

Usage:
    from workflow import run_copilot
    result = run_copilot(resume_dict, job_description)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Union

# Make the project root importable whether run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.memory import ChatMemoryBuffer  # noqa: E402
from llama_index.core.workflow import (  # noqa: E402
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from agents.followup import generate_followups  # noqa: E402
from agents.planner import plan_investigation  # noqa: E402
from agents.reflector import re_evaluate, reflect  # noqa: E402
from agents.synthesizer import synthesize_fit  # noqa: E402
from tools.classifier import QueryClassifier  # noqa: E402
from tools.rag_tool import setup_rag_tool  # noqa: E402
from tools.sql_tool import setup_sql_tool  # noqa: E402

# Tool engines are expensive to build (they load the embedding model), so cache
# them across runs at module level.
_SQL_ENGINE = None
_RAG_ENGINE = None
_CLASSIFIER = None


def _get_sql_engine():
    global _SQL_ENGINE
    if _SQL_ENGINE is None:
        _SQL_ENGINE = setup_sql_tool().query_engine
    return _SQL_ENGINE


def _get_rag_engine():
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        _RAG_ENGINE = setup_rag_tool().query_engine
    return _RAG_ENGINE


def _get_classifier() -> QueryClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = QueryClassifier()
    return _CLASSIFIER


# ── Conversational memory ────────────────────────────────────────────────────

class WorkflowMemory:
    """
    Session memory for the co-pilot, backed by a LlamaIndex ChatMemoryBuffer
    (token_limit=4000). The workflow records a compact summary of each step here
    so later agents — and the chat handler — can recall the whole analysis.
    """

    def __init__(self, token_limit: int = 4000):
        self.buffer = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    def add(self, label: str, content: str) -> None:
        """Store one labelled memory item as an assistant message."""
        text = (content or "").strip()
        if not text:
            return
        self.buffer.put(
            ChatMessage(role=MessageRole.ASSISTANT, content=f"[{label}]\n{text}")
        )

    def add_qa(self, question: str, answer: str) -> None:
        """Store a follow-up question/answer pair."""
        self.buffer.put(ChatMessage(role=MessageRole.USER, content=question))
        self.buffer.put(ChatMessage(role=MessageRole.ASSISTANT, content=answer))

    def get_memory_context(self) -> str:
        """Return the full memory as a formatted string to prepend to a prompt."""
        items = [m.content for m in self.buffer.get_all() if m.content]
        if not items:
            return "(no memory recorded yet)"
        return (
            "=== SESSION MEMORY (what we know so far) ===\n\n"
            + "\n\n".join(items)
        )


# Most-recent session memory, so the no-arg get_memory_context() (per spec) and
# the chat handler can reach it after a run. The UI (Prompt 7) will hold its own
# WorkflowMemory per Streamlit session instead of relying on this global.
_CURRENT_MEMORY: Optional[WorkflowMemory] = None


def get_memory_context() -> str:
    """Formatted memory string for the current session (empty if none)."""
    if _CURRENT_MEMORY is None:
        return "(no memory recorded yet)"
    return _CURRENT_MEMORY.get_memory_context()


# ── Memory summarisers (compact, grounded — no extra LLM calls) ──────────────

def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _cap_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + "…"


def _summarize_resume(resume: dict) -> str:
    """Build a ~200-word plain-text summary from the parsed resume dict."""
    skills = resume.get("skills", {}) or {}
    flat_skills = []
    for group in ("programming_languages", "frameworks_and_libraries",
                  "tools_and_platforms", "databases"):
        flat_skills.extend(skills.get(group, []) or [])
    edu = resume.get("education", []) or []
    edu_str = ""
    if edu:
        e = edu[0]
        edu_str = (f"{e.get('degree','')} in {e.get('field','')} from "
                   f"{e.get('institution','')} ({e.get('year','')}, "
                   f"CGPA {e.get('cgpa')})")
    work = resume.get("work_experience", []) or []
    work_str = "; ".join(
        f"{w.get('role','')} at {w.get('company','')} "
        f"({w.get('duration_months','?')} mo)" for w in work
    )
    parts = [
        f"Candidate: {resume.get('name','Unknown')}.",
        f"Experience level: {resume.get('experience_level','unknown')}, "
        f"~{resume.get('total_experience_years','?')} years total.",
        f"Education: {edu_str}." if edu_str else "",
        f"Key skills: {', '.join(flat_skills)}." if flat_skills else "",
        f"Work experience: {work_str}." if work_str else "",
        f"Projects: {len(resume.get('projects', []) or [])} listed.",
        f"Certifications: {len(resume.get('certifications', []) or [])}.",
        f"Target roles: {', '.join(resume.get('target_roles', []) or [])}.",
    ]
    return _cap_words(" ".join(p for p in parts if p), 200)


def _summarize_plan(plan: dict) -> str:
    skills = ", ".join(plan.get("key_skills_to_evaluate", []) or [])
    gaps = "; ".join(plan.get("initial_gap_hypothesis", []) or [])
    return (
        f"Investigation priority: {plan.get('investigation_priority','?')}. "
        f"{len(plan.get('sql_queries', []) or [])} SQL queries, "
        f"{len(plan.get('rag_queries', []) or [])} RAG queries planned.\n"
        f"Key skills to evaluate: {skills}.\n"
        f"Initial gap hypotheses: {gaps}."
    )


def _bulletize(results: list, max_each: int = 280) -> str:
    """Turn a list of tool result strings into a bullet list."""
    lines = []
    for r in results or []:
        first = _truncate(str(r).strip().replace("\n", " "), max_each)
        if first:
            lines.append(f"- {first}")
    return "\n".join(lines) if lines else "(no findings)"


def _summarize_synthesis(synthesis: dict) -> str:
    if not isinstance(synthesis, dict):
        return str(synthesis)
    bd = synthesis.get("score_breakdown", {}) or {}
    return _truncate(
        f"Fit score: {synthesis.get('fit_score')}/100 "
        f"(skills {bd.get('skills_match')}, experience {bd.get('experience_match')}, "
        f"culture {bd.get('culture_fit')}). "
        f"Recommendation: {synthesis.get('recommendation')}. "
        f"Strengths: {len(synthesis.get('strengths', []))}, "
        f"Gaps: {len(synthesis.get('gaps', []))}. "
        f"Summary: {synthesis.get('summary', '')}",
        1200,
    )


# ── Events ───────────────────────────────────────────────────────────────────

class ClassifyEvent(Event):
    """Carries query classification result + original inputs to plan_step."""
    classification: dict
    resume_dict: dict
    job_description: str


class PlanEvent(Event):
    investigation_plan: dict


class ToolResultsEvent(Event):
    sql_results: list
    rag_results: list
    investigation_plan: dict


class SynthesisEvent(Event):
    synthesis: dict
    tool_results: dict


class ReflectionEvent(Event):
    reflection: dict
    synthesis: dict
    tool_results: dict


class ReplanEvent(Event):
    """Loop-back event: carries new targeted queries for another tool pass."""
    investigation_plan: dict


class FollowUpEvent(Event):
    followup: dict
    synthesis: dict


# ── Workflow ─────────────────────────────────────────────────────────────────

_MAX_REFLECTION_CYCLES = 2


class RouterOutputAgentWorkflow(Workflow):
    """Plan -> tools -> synthesize -> reflect (loop) -> follow-up."""

    # The live session memory is held as an instance attribute (set by
    # run_copilot) rather than round-tripped through ctx.store: the context may
    # serialize/copy stored objects between steps, which would silently drop
    # mutations to a shared mutable object. All steps mutate this one instance.
    _memory: Optional["WorkflowMemory"] = None
    # Questions this candidate answered in previous sessions (skipped this run).
    _previously_asked: Optional[list] = None

    @step
    async def classify_step(self, ctx: Context, ev: StartEvent) -> ClassifyEvent:
        print("[Classifier] Classifying investigation type...")
        if self._memory is None:
            self._memory = getattr(ev, "memory", None) or WorkflowMemory()
        await ctx.store.set("has_memory", True)

        try:
            clf = _get_classifier()
            classification = clf.classify_hiring_context(
                ev.resume_dict, ev.job_description
            )
        except Exception as exc:
            print(f"   [warn] Classifier failed, defaulting to HYBRID: {exc}")
            classification = {
                "type": "HYBRID", "confidence": 50,
                "reasoning": f"Fallback (error: {exc})",
                "sql_weight": 0.5, "rag_weight": 0.5,
            }

        await ctx.store.set("classification", classification)
        print(
            f"   -> Type: {classification['type']} "
            f"(conf={classification['confidence']}%, "
            f"sql={classification['sql_weight']}, rag={classification['rag_weight']})"
        )
        return ClassifyEvent(
            classification=classification,
            resume_dict=ev.resume_dict,
            job_description=ev.job_description,
        )

    @step
    async def plan_step(self, ctx: Context, ev: ClassifyEvent) -> PlanEvent:
        print("[Planner] Planning investigation...")
        memory = self._memory

        plan = plan_investigation(
            ev.resume_dict, ev.job_description, classification=ev.classification
        )
        await ctx.store.set("resume_dict", ev.resume_dict)
        await ctx.store.set("job_description", ev.job_description)
        await ctx.store.set("investigation_plan", plan)

        # Record resume summary, JD, and plan summary into memory.
        memory.add("RESUME SUMMARY", _summarize_resume(ev.resume_dict))
        memory.add("JOB DESCRIPTION", _truncate(ev.job_description, 500))
        memory.add("INVESTIGATION PLAN", _summarize_plan(plan))
        return PlanEvent(investigation_plan=plan)

    @step
    async def execute_tools_step(
        self, ctx: Context, ev: Union[PlanEvent, ReplanEvent]
    ) -> ToolResultsEvent:
        is_replan = isinstance(ev, ReplanEvent)
        label = "[Tools] Re-running targeted queries (replan)..." if is_replan \
                else "[Tools] Querying job database and culture docs..."
        print(label)
        plan = ev.investigation_plan
        sql_queries = plan.get("sql_queries", [])
        rag_queries = plan.get("rag_queries", [])

        async def run_group(engine_getter, queries, tool_label):
            """Run a group of queries; isolate failures so the other tool survives."""
            if not queries:
                return []
            try:
                engine = engine_getter()
            except Exception as exc:
                print(f"   [warn] {tool_label} tool unavailable: {exc}")
                return [f"[{tool_label} ERROR] tool unavailable: {exc}"]
            tasks = [
                asyncio.to_thread(lambda q=q: str(engine.query(q)))
                for q in queries
            ]
            done = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for q, r in zip(queries, done):
                if isinstance(r, Exception):
                    print(f"   [warn] {tool_label} query failed: {q!r} -> {r}")
                    results.append(f"[{tool_label} ERROR] {q}: {r}")
                else:
                    results.append(r)
            return results

        # Run SQL and RAG groups concurrently.
        new_sql, new_rag = await asyncio.gather(
            run_group(_get_sql_engine, sql_queries, "SQL"),
            run_group(_get_rag_engine, rag_queries, "RAG"),
        )

        # On replan: MERGE new results with previous ones so the synthesizer
        # has the full evidence base (not just the new queries).
        if is_replan:
            prev_sql = await ctx.store.get("sql_results", default=[])
            prev_rag = await ctx.store.get("rag_results", default=[])
            sql_results = (prev_sql or []) + new_sql
            rag_results = (prev_rag or []) + new_rag
            print(f"   Merged: {len(prev_sql)} prev SQL + {len(new_sql)} new = {len(sql_results)} total")
            print(f"   Merged: {len(prev_rag)} prev RAG + {len(new_rag)} new = {len(rag_results)} total")
        else:
            sql_results = new_sql
            rag_results = new_rag

        await ctx.store.set("sql_results", sql_results)
        await ctx.store.set("rag_results", rag_results)

        memory = self._memory
        memory.add("KEY SQL FINDINGS", _bulletize(sql_results))
        memory.add("KEY RAG FINDINGS", _bulletize(rag_results))

        return ToolResultsEvent(
            sql_results=sql_results,
            rag_results=rag_results,
            investigation_plan=plan,
        )

    @step
    async def synthesize_step(
        self, ctx: Context, ev: ToolResultsEvent
    ) -> SynthesisEvent:
        print("⚖️  Synthesizing fit analysis...")
        tool_results = {
            "sql_results": ev.sql_results,
            "rag_results": ev.rag_results,
            "investigation_plan": ev.investigation_plan,
        }
        memory = self._memory
        inputs = {
            "resume": await ctx.store.get("resume_dict", default={}),
            "job_description": await ctx.store.get("job_description", default=""),
            "sql_results": ev.sql_results,
            "rag_results": ev.rag_results,
            "investigation_plan": ev.investigation_plan,
            "memory_context": memory.get_memory_context(),
        }
        synthesis = synthesize_fit(inputs)
        await ctx.store.set("synthesis", synthesis)

        # Record synthesis summary into memory.
        memory.add("SYNTHESIS", _summarize_synthesis(synthesis))
        return SynthesisEvent(synthesis=synthesis, tool_results=tool_results)

    @step
    async def reflect_step(
        self, ctx: Context, ev: SynthesisEvent
    ) -> Union[FollowUpEvent, ReplanEvent]:
        cycles = await ctx.store.get("reflection_cycles", default=0)
        cycle_num = cycles + 1
        print(f"[Reflector] Cycle {cycle_num} — checking analysis quality...")

        sql_results = ev.tool_results.get("sql_results", [])
        rag_results = ev.tool_results.get("rag_results", [])
        plan = ev.tool_results.get("investigation_plan", {})

        try:
            reflection = reflect(
                synthesis=ev.synthesis,
                sql_results=sql_results,
                rag_results=rag_results,
                investigation_plan=plan,
                cycle_number=cycle_num,
            )
        except Exception as exc:
            print(f"   [warn] Reflector failed, proceeding: {exc}")
            reflection = {
                "confidence": 80, "should_replan": False,
                "weak_points": [], "replan_queries": {"sql": [], "rag": []},
                "reflection_notes": f"Reflector error: {exc}",
            }

        confidence = reflection.get("confidence", 80)
        should_replan = reflection.get("should_replan", False)
        status = "Replanning" if should_replan else "Proceeding"
        print(f"   [Reflector] Cycle {cycle_num} — Confidence: {confidence}% — {status}")

        await ctx.store.set("reflection", reflection)
        await ctx.store.set("reflection_cycles", cycle_num)

        if should_replan:
            replan_q = reflection.get("replan_queries", {})
            new_sql_q = replan_q.get("sql", [])
            new_rag_q = replan_q.get("rag", [])
            if new_sql_q or new_rag_q:
                print(f"   Replan queries: {len(new_sql_q)} SQL, {len(new_rag_q)} RAG")
                new_plan = {
                    **plan,
                    "sql_queries": new_sql_q,
                    "rag_queries": new_rag_q,
                }
                return ReplanEvent(investigation_plan=new_plan)
            else:
                print("   No replan queries provided — proceeding anyway.")

        return FollowUpEvent(followup={}, synthesis=ev.synthesis)

    @step
    async def followup_step(
        self, ctx: Context, ev: FollowUpEvent
    ) -> StopEvent:
        print("[Follow-up] Generating follow-up questions...")
        memory = self._memory
        followup = generate_followups(ev.synthesis.get("gaps", []),
                                      previously_asked=self._previously_asked)
        await ctx.store.set("followup", followup)

        q_lines = "\n".join(
            f"- ({q.get('gap_addressed','')}) {q.get('question','')}"
            for q in followup.get("questions", []) or []
        )
        memory.add("FOLLOW-UP QUESTIONS", q_lines)

        reflection_cycles = await ctx.store.get("reflection_cycles", default=0)
        classification = await ctx.store.get("classification", default={})

        final = {
            "investigation_plan": await ctx.store.get("investigation_plan"),
            "classification": classification,
            "tool_results": {
                "sql_results": await ctx.store.get("sql_results", default=[]),
                "rag_results": await ctx.store.get("rag_results", default=[]),
            },
            "synthesis": ev.synthesis,
            "reflection": await ctx.store.get("reflection"),
            "reflection_cycles": reflection_cycles,
            "followup_questions": followup,
        }
        return StopEvent(result=final)


def run_copilot(
    resume_dict: dict,
    job_description: str,
    memory: Optional[WorkflowMemory] = None,
    previously_asked: Optional[list] = None,
) -> dict:
    """
    Run the full workflow synchronously and return the final output dict.

    A WorkflowMemory is created if one isn't supplied (the UI passes its own
    per-session instance). The populated memory is exposed on the result as
    "memory_context" (string) and "_session_memory" (the live object), and also
    registered as the module-level current session memory.

    previously_asked: follow-up questions answered in earlier sessions (from the
    profile store) — they won't be asked again.
    """
    global _CURRENT_MEMORY
    memory = memory or WorkflowMemory()
    _CURRENT_MEMORY = memory
    workflow = RouterOutputAgentWorkflow(timeout=360, verbose=True)
    # Share the live memory object + cross-session question history with steps.
    workflow._memory = memory
    workflow._previously_asked = previously_asked or []

    async def _arun() -> dict:
        # workflow.run() must be called inside a running loop (it schedules a
        # task at call-time), so build and await the handler here.
        handler = workflow.run(
            resume_dict=resume_dict,
            job_description=job_description,
            memory=memory,
        )
        return await handler

    result = asyncio.run(_arun())
    result["memory_context"] = memory.get_memory_context()
    result["_session_memory"] = memory
    return result


# ── Hardcoded test ───────────────────────────────────────────────────────────

# Parsed resume from Prompt 4 (agents/resume_parser.py output for the sample).
_TEST_RESUME = {
    "name": "ARJUN MEHTA",
    "email": "arjun.mehta2025@gmail.com",
    "phone": "+91 98765 43210",
    "education": [
        {"degree": "Bachelor of Technology (BTech)",
         "field": "Computer Science and Engineering",
         "institution": "Pune Institute of Engineering and Technology, Pune (Tier-2)",
         "year": "2025", "cgpa": "8.2 / 10"},
        {"degree": "Higher Secondary (Class XII)", "field": "Science (PCM)",
         "institution": "Maharashtra State Board", "year": "2021", "cgpa": None},
    ],
    "total_experience_years": 0.42,
    "skills": {
        "programming_languages": ["Python", "SQL", "JavaScript"],
        "frameworks_and_libraries": ["PyTorch", "FastAPI", "pandas",
                                     "scikit-learn", "React"],
        "tools_and_platforms": ["Docker", "Git", "MLflow", "Linux", "VS Code"],
        "databases": ["PostgreSQL", "SQLite", "ChromaDB"],
        "soft_skills": ["Problem-solving", "Teamwork", "Communication",
                        "Fast learner"],
    },
    "work_experience": [
        {"company": "NeuralNest AI", "role": "Machine Learning Intern",
         "duration_months": 3,
         "key_responsibilities": [
             "Built an image classification pipeline using PyTorch (91% accuracy).",
             "Containerized the model with Docker and served it via FastAPI.",
             "Set up experiment tracking with MLflow."],
         "technologies_used": ["Python", "PyTorch", "FastAPI", "Docker",
                               "MLflow", "pandas"]},
        {"company": "Qualex Systems", "role": "Backend Engineering Intern",
         "duration_months": 2,
         "key_responsibilities": [
             "Developed 8 REST API endpoints in FastAPI backed by PostgreSQL.",
             "Wrote unit tests and added Git-based CI checks."],
         "technologies_used": ["Python", "FastAPI", "PostgreSQL", "Git",
                               "Docker"]},
    ],
    "projects": [
        {"name": "RAG Chatbot for College FAQs",
         "description": "Retrieval-augmented chatbot over college documents.",
         "technologies": ["Python", "LlamaIndex", "ChromaDB", "Hugging Face",
                          "Streamlit"],
         "impact": "Reduced repetitive queries to the admin office."},
        {"name": "Image Classifier for Plant Disease Detection",
         "description": "CNN in PyTorch classifying 5 plant diseases (88% acc).",
         "technologies": ["Python", "PyTorch", "OpenCV", "Flask"],
         "impact": None},
        {"name": "REST API for Expense Tracker",
         "description": "Secure REST API with JWT auth and CRUD operations.",
         "technologies": ["Python", "FastAPI", "PostgreSQL", "Docker"],
         "impact": None},
        {"name": "Personal Portfolio Website",
         "description": "Responsive portfolio website.",
         "technologies": ["React", "JavaScript", "HTML", "CSS"], "impact": None},
        {"name": "GenAI Resume Analyzer (Ongoing)",
         "description": "Analyzes resumes against JDs using LLMs and embeddings.",
         "technologies": ["Python", "LangChain", "OpenAI API", "FAISS"],
         "impact": None},
    ],
    "certifications": [
        "Machine Learning Specialization - Coursera (Andrew Ng / DeepLearning.AI), 2023"],
    "target_roles": ["ML Engineer", "AI Engineer"],
    "experience_level": "fresher",
}

_TEST_JD = (
    "ML Engineer at a Series A fintech startup in Bangalore. 3-5 years "
    "experience. Required: Python, PyTorch, MLflow, Docker, REST APIs. "
    "Nice to have: Kubernetes, AWS, system design experience. You will own the "
    "full ML lifecycle from experimentation to production deployment. "
    "Fast-paced environment, strong ownership culture."
)


def _print_test_output(result: dict) -> None:
    import json

    plan = result["investigation_plan"]
    tr = result["tool_results"]

    print("\n" + "=" * 70)
    print("  INVESTIGATION PLAN (generated by the planner)")
    print("=" * 70)
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("  TOOL RESULTS COLLECTED")
    print("=" * 70)
    print(f"\n--- SQL results ({len(tr['sql_results'])}) ---")
    for i, r in enumerate(tr["sql_results"], 1):
        print(f"\n[SQL {i}] {plan['sql_queries'][i-1] if i-1 < len(plan['sql_queries']) else ''}")
        print(str(r).strip())
    print(f"\n--- RAG results ({len(tr['rag_results'])}) ---")
    for i, r in enumerate(tr["rag_results"], 1):
        print(f"\n[RAG {i}] {plan['rag_queries'][i-1] if i-1 < len(plan['rag_queries']) else ''}")
        print(str(r).strip())

    print("\n" + "=" * 70)
    print("  STAGE 1 — SYNTHESIS (fit analysis)")
    print("=" * 70)
    print(json.dumps(result["synthesis"], indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("  STAGE 2 — FOLLOW-UP QUESTIONS")
    print("=" * 70)
    fq = result["followup_questions"].get("questions", [])
    if not fq:
        print("  (no follow-up questions generated)")
    for i, q in enumerate(fq, 1):
        print(f"\n[Q{i}] (addresses gap: {q.get('gap_addressed','')})")
        print(f"  {q.get('question','')}")
        print(f"  [hidden criteria] {q.get('what_good_answer_looks_like','')}")


def _test_memory(output: dict) -> None:
    """Part C: verify memory is populated and the chat handler can use it."""
    from memory.chat_handler import ConversationHandler

    memory_context = output["memory_context"]

    print("\n" + "=" * 70)
    print("  SESSION MEMORY CONTEXT (after workflow run)")
    print("=" * 70)
    print(memory_context)

    print("\n" + "=" * 70)
    print("  CONVERSATION HANDLER — sample chat")
    print("=" * 70)
    handler = ConversationHandler()

    q1 = "Why did I get this score?"
    print(f"\n[User] {q1}")
    print(f"[Co-pilot] {handler.answer_question(q1, memory_context)}")

    # Second question demonstrates memory persistence within the session.
    q2 = "What should I improve first?"
    print(f"\n[User] {q2}")
    print(f"[Co-pilot] {handler.answer_question(q2, memory_context)}")

    print("\n" + "=" * 70)
    print("  MEMORY PERSISTENCE CHECK")
    print("=" * 70)
    print(f"  conversation turns stored : {len(handler.get_history())}")
    print(f"  memory_context length     : {len(memory_context)} chars")
    print(f"  get_memory_context() works: {len(get_memory_context()) > 0}")


def _test_reevaluation(output: dict) -> None:
    """Part E: re-evaluate after simulated (hardcoded strong) follow-up answers."""
    import json

    from agents.reflector import re_evaluate

    synthesis = output["synthesis"]
    followup_questions = output["followup_questions"]
    questions = followup_questions.get("questions", [])

    print("\n" + "=" * 70)
    print("  STAGE 3 — RE-EVALUATION (after simulated answers)")
    print("=" * 70)
    if not questions:
        print("  (no follow-up questions to answer — skipping re-evaluation)")
        return

    # Hardcode 2 strong answers for testing. Follow-ups exist precisely to let a
    # candidate surface relevant experience NOT captured on the resume, so these
    # answers reveal additional, concrete production-ML work.
    simulated_answers = [
        ("Beyond the two internships, I spent about 10 months as a contract ML "
         "engineer for a logistics startup where I owned a demand-forecasting "
         "service end to end: built the ingestion pipeline, trained PyTorch "
         "models, served them via FastAPI in Docker, deployed on AWS EKS "
         "(Kubernetes) with horizontal autoscaling, and set up Prometheus/Grafana "
         "monitoring with automated data-drift alerts. I handled a real "
         "production incident when input distribution shifted and added a "
         "retraining trigger. That adds close to a year of continuous, "
         "production-grade ML ownership."),
        ("For system design: I designed and ran a production ML service at "
         "~50 req/s — queue-based data ingestion, a PostgreSQL feature store, "
         "model serving behind a load-balanced FastAPI/Docker deployment on AWS "
         "EKS with autoscaling, CloudWatch + Prometheus monitoring, automated "
         "drift detection that triggers retraining, and blue-green deploys for "
         "safe rollbacks. I made explicit latency-vs-cost trade-offs and "
         "documented the architecture and failure modes."),
    ]
    followup_qa = []
    for q, ans in zip(questions, simulated_answers):
        followup_qa.append({"question": q["question"], "answer": ans})
    print(f"  Simulated {len(followup_qa)} strong answers to the "
          f"{len(followup_qa)} top question(s).\n")

    updated = re_evaluate(synthesis, followup_qa, followup_questions)

    print(f"  Original fit_score : {synthesis['fit_score']}")
    print(f"  Updated  fit_score : {updated['fit_score']}  "
          f"(delta {updated['score_delta']:+d})")
    print(f"  Recommendation     : {synthesis['recommendation']} -> "
          f"{updated['recommendation']}")
    print("\n  Per-answer assessments:")
    for a in updated["answer_assessments"]:
        print(f"   - [{a['rating']}] {a['severity']} gap "
              f"(+{a['score_delta']}): {a['gap_addressed']}")
    print(f"\n  Resolved gaps  : {[g.get('point') for g in updated['resolved_gaps']]}")
    print(f"  Remaining gaps : {[g.get('point') for g in updated['remaining_gaps']]}")
    print("\n  Full updated synthesis:")
    print(json.dumps(updated, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    output = run_copilot(_TEST_RESUME, _TEST_JD)
    _print_test_output(output)
    _test_reevaluation(output)
    _test_memory(output)
