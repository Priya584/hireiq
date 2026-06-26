"""
Benchmark evaluation script for AI Hiring Co-pilot.

Runs 30 test cases against the live pipeline and measures:
  - Recommendation accuracy (overall + per category)
  - Score calibration (% within expected range)
  - Gap detection quality
  - Per-analysis runtime

Usage:
    python benchmark/evaluate.py               # Full live run (~30-90 min)
    python benchmark/evaluate.py --dry-run     # Logic-only, no LLM calls (~5 sec)
    python benchmark/evaluate.py --limit 5     # Run first 5 cases only
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to avoid cp1252 emoji errors.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Make project root importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

_BENCH_DIR = Path(__file__).resolve().parent
_TEST_CASES_PATH = _BENCH_DIR / "test_cases.json"
_RESULTS_PATH = _BENCH_DIR / "results.json"

_MODEL = "openai/gpt-oss-120b:free"


# ── Resume synthesis from summary ─────────────────────────────────────────────

def _summary_to_resume_dict_stub(summary: str) -> dict:
    """
    Heuristic minimal resume dict for --dry-run mode.
    No LLM call — constructs a plausible stub from keywords in the summary.
    """
    summary_lower = summary.lower()

    # Infer experience level
    level = "fresher"
    if "senior" in summary_lower or "6 year" in summary_lower or "7 year" in summary_lower or "8 year" in summary_lower:
        level = "senior"
    elif "3 year" in summary_lower or "4 year" in summary_lower or "5 year" in summary_lower:
        level = "mid"
    elif "1 year" in summary_lower or "2 year" in summary_lower:
        level = "junior"

    # Infer total experience years
    nums = re.findall(r"(\d+)\s+year", summary_lower)
    exp_years = float(nums[0]) if nums else 0.5

    # Infer skills from keywords
    all_skills = [
        "Python", "PyTorch", "TensorFlow", "scikit-learn", "pandas", "numpy",
        "FastAPI", "Flask", "Django", "React", "Node.js", "TypeScript",
        "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform",
        "MLflow", "Airflow", "Kafka", "Spark", "Redis", "PostgreSQL",
        "MongoDB", "ChromaDB", "FAISS", "LangChain", "LlamaIndex",
        "HuggingFace", "BERT", "GPT", "spaCy", "SQL", "Git", "Linux",
    ]
    found_skills = [s for s in all_skills if s.lower() in summary_lower]

    pl = [s for s in found_skills if s in ("Python", "SQL", "TypeScript", "Go", "Java", "Swift")]
    fw = [s for s in found_skills if s in ("PyTorch", "TensorFlow", "scikit-learn", "FastAPI",
                                            "Flask", "Django", "React", "LangChain", "LlamaIndex",
                                            "HuggingFace", "spaCy", "pandas", "numpy")]
    tools = [s for s in found_skills if s in ("Docker", "Kubernetes", "AWS", "GCP", "Azure",
                                               "Terraform", "MLflow", "Airflow", "Kafka", "Spark",
                                               "Redis", "Git", "Linux")]
    dbs = [s for s in found_skills if s in ("PostgreSQL", "MongoDB", "ChromaDB", "FAISS",
                                             "SQLite", "Redshift", "MySQL")]

    return {
        "name": "Benchmark Candidate",
        "email": "benchmark@example.com",
        "phone": "+91 99999 00000",
        "education": [{
            "degree": "B.Tech",
            "field": "Computer Science",
            "institution": "Test University",
            "year": str(2024 - int(exp_years)),
            "cgpa": "8.0",
        }],
        "total_experience_years": exp_years,
        "skills": {
            "programming_languages": pl or ["Python"],
            "frameworks_and_libraries": fw,
            "tools_and_platforms": tools,
            "databases": dbs,
            "soft_skills": ["Communication", "Teamwork"],
        },
        "work_experience": [{
            "company": "Previous Company",
            "role": "Software Engineer",
            "duration_months": int(exp_years * 12),
            "key_responsibilities": [summary[:200]],
            "technologies_used": (pl + fw + tools)[:6],
        }] if exp_years >= 0.5 else [],
        "projects": [],
        "certifications": [],
        "target_roles": ["Software Engineer"],
        "experience_level": level,
    }


def _summary_to_resume_dict(summary: str) -> dict:
    """
    Use the LLM to expand a one-paragraph resume summary into a full
    structured resume dict matching the parse_resume() output schema.
    Falls back to the stub on parse failure.
    """
    from llama_index.llms.openrouter import OpenRouter

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set. Add it to your .env file.")

    llm = OpenRouter(api_key=api_key, model=_MODEL, max_tokens=1200, temperature=0.0)

    schema = """{
  "name": "Candidate Name",
  "email": "email@example.com",
  "phone": "+91 99999 00000",
  "education": [{"degree": "string", "field": "string", "institution": "string", "year": "string", "cgpa": "string or null"}],
  "total_experience_years": float,
  "skills": {
    "programming_languages": [],
    "frameworks_and_libraries": [],
    "tools_and_platforms": [],
    "databases": [],
    "soft_skills": []
  },
  "work_experience": [{"company": "string", "role": "string", "duration_months": int, "key_responsibilities": [], "technologies_used": []}],
  "projects": [],
  "certifications": [],
  "target_roles": [],
  "experience_level": "fresher|junior|mid|senior"
}"""

    prompt = (
        f"Convert this candidate resume summary into a complete structured JSON resume profile.\n"
        f"Invent plausible but realistic values for name, email, company names.\n"
        f"Output ONLY the JSON object, nothing else.\n\n"
        f"TARGET SCHEMA:\n{schema}\n\n"
        f"RESUME SUMMARY:\n{summary}"
    )

    raw = str(llm.complete(prompt)).strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
        return data
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    print("    [warn] LLM resume expansion failed — using stub.")
    return _summary_to_resume_dict_stub(summary)


# ── Dry-run stub result ────────────────────────────────────────────────────────

def _stub_result(tc: dict) -> dict:
    """Return a predictable stub result for --dry-run (no LLM, no workflow)."""
    expected_rec = tc["expected_recommendation"]
    mid = sum(tc["expected_score_range"]) / 2
    rec_to_score = {
        "not_fit": 15,
        "weak_fit": 32,
        "possible_fit": 57,
        "strong_fit": 83,
    }
    stub_score = int(rec_to_score.get(expected_rec, mid))

    strengths = [
        {"point": s, "evidence": "stub evidence", "source": "resume"}
        for s in tc.get("key_strengths_that_should_appear", [])[:2]
    ]
    gaps = [
        {"point": g, "severity": "moderate", "evidence": "stub evidence", "source": "resume"}
        for g in tc.get("key_gaps_that_should_appear", [])[:2]
    ]
    return {
        "synthesis": {
            "fit_score": stub_score,
            "recommendation": expected_rec,
            "score_breakdown": {"skills_match": stub_score, "experience_match": stub_score, "culture_fit": stub_score},
            "strengths": strengths,
            "gaps": gaps,
            "summary": "Stub result for dry-run mode.",
            "alternative_roles": [],
            "citations": {"from_sql": [], "from_rag": []},
            "confidence": 80,
        },
        "followup_questions": {"questions": []},
    }


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _in_range(score: int, expected_range: list) -> bool:
    return expected_range[0] <= score <= expected_range[1]


def _rec_correct(actual: str, expected: str) -> bool:
    return actual == expected


def _gap_detection(gaps_found: list, required: list, forbidden: list) -> dict:
    """
    required  : gaps that SHOULD appear (mismatch/borderline cases)
    forbidden : skills that should NOT be flagged as gaps (fit cases)
    """
    found_points = [g.get("point", "").lower() for g in gaps_found]

    def _fuzzy_match(needle: str, haystack: list) -> bool:
        n = needle.lower()
        return any(n in h or h in n for h in haystack)

    correctly_flagged = sum(1 for g in required if _fuzzy_match(g, found_points))
    correctly_not_flagged = sum(1 for f in forbidden if not _fuzzy_match(f, found_points))

    return {
        "correctly_flagged": correctly_flagged,
        "total_required": len(required),
        "correctly_not_flagged": correctly_not_flagged,
        "total_forbidden": len(forbidden),
    }


# ── Core benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(dry_run: bool = False, limit: Optional[int] = None) -> dict:
    """
    Run the full benchmark and return the complete results dict.

    Parameters
    ----------
    dry_run : bool
        If True, skip all LLM/workflow calls and use stub results.
        Useful for testing the evaluation logic without API costs.
    limit : int or None
        If set, run only the first N test cases.
    """
    # ── Load test cases ───────────────────────────────────────────────────────
    if not _TEST_CASES_PATH.exists():
        raise FileNotFoundError(f"test_cases.json not found at {_TEST_CASES_PATH}")

    with open(_TEST_CASES_PATH, encoding="utf-8") as f:
        all_cases = json.load(f)

    test_cases = all_cases[:limit] if limit else all_cases
    n = len(test_cases)

    if not dry_run:
        from workflow import run_copilot  # noqa: PLC0415

    mode_label = "DRY RUN (no LLM calls)" if dry_run else f"LIVE — {n} cases, expect 30-90 min"
    print(f"\n{'='*62}")
    print(f"  AI HIRING CO-PILOT — BENCHMARK")
    print(f"  Mode : {mode_label}")
    print(f"  Cases: {n}")
    print(f"{'='*62}\n")

    # ── Per-case tracking ─────────────────────────────────────────────────────
    cat_stats: dict = {
        "fit":        {"total": 0, "correct": 0},
        "mismatch":   {"total": 0, "correct": 0},
        "borderline": {"total": 0, "correct": 0},
    }
    in_range_count = 0
    deviations: list[float] = []
    times: list[float] = []
    results: list[dict] = []

    for idx, tc in enumerate(test_cases, 1):
        tc_id = tc["id"]
        cat = tc["category"]
        expected_rec = tc["expected_recommendation"]
        expected_range = tc["expected_score_range"]

        print(f"[{idx:2d}/{n}] {tc_id:20s} ", end="", flush=True)
        t0 = time.time()

        try:
            if dry_run:
                result = _stub_result(tc)
                resume_dict = _summary_to_resume_dict_stub(tc["resume_summary"])
            else:
                resume_dict = _summary_to_resume_dict(tc["resume_summary"])
                result = run_copilot(resume_dict, tc["job_description"])

            elapsed = time.time() - t0
            times.append(elapsed)

            synth = result.get("synthesis", {})
            actual_score = int(synth.get("fit_score", 0))
            actual_rec = synth.get("recommendation", "not_fit")
            gaps_found = synth.get("gaps", [])
            strengths_found = synth.get("strengths", [])
            followup_qs = result.get("followup_questions", {}).get("questions", [])

            # ── Recommendation accuracy ───────────────────────────────────────
            rec_ok = _rec_correct(actual_rec, expected_rec)
            cat_stats[cat]["total"] += 1
            if rec_ok:
                cat_stats[cat]["correct"] += 1

            # ── Score calibration ─────────────────────────────────────────────
            score_ok = _in_range(actual_score, expected_range)
            if score_ok:
                in_range_count += 1
            midpoint = sum(expected_range) / 2
            deviations.append(abs(actual_score - midpoint))

            # ── Gap detection ─────────────────────────────────────────────────
            gap_det = _gap_detection(
                gaps_found,
                required=tc.get("key_gaps_that_should_appear", []),
                forbidden=tc.get("key_gaps_that_should_not_appear", []),
            )

            entry = {
                "id": tc_id,
                "category": cat,
                "resume_summary_snippet": tc["resume_summary"][:120] + "…",
                "expected_recommendation": expected_rec,
                "actual_recommendation": actual_rec,
                "recommendation_correct": rec_ok,
                "expected_score_range": expected_range,
                "actual_score": actual_score,
                "score_in_range": score_ok,
                "score_deviation_from_midpoint": round(abs(actual_score - midpoint), 1),
                "gap_detection": gap_det,
                "gaps_found": [g.get("point", "") for g in gaps_found],
                "strengths_found": [s.get("point", "") for s in strengths_found],
                "followup_questions_count": len(followup_qs),
                "time_seconds": round(elapsed, 2),
                "error": None,
            }

            status = "[OK]" if rec_ok else "[!!]"
            range_ok = "range=OK" if score_ok else "range=!!"
            print(f"{status} {range_ok}  score={actual_score:3d}  rec={actual_rec:12s}  ({elapsed:.1f}s)")

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            times.append(elapsed)
            cat_stats[cat]["total"] += 1  # count the failure
            print(f"[ERROR] ({elapsed:.1f}s): {exc}")
            entry = {
                "id": tc_id,
                "category": cat,
                "expected_recommendation": expected_rec,
                "actual_recommendation": None,
                "recommendation_correct": False,
                "score_in_range": False,
                "score_deviation_from_midpoint": None,
                "gap_detection": {},
                "time_seconds": round(elapsed, 2),
                "error": str(exc),
            }

        results.append(entry)

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    def pct(num: int, den: int) -> float:
        return round(100.0 * num / den, 1) if den else 0.0

    total_correct = sum(1 for r in results if r.get("recommendation_correct"))
    summary = {
        "total_cases": n,
        "overall_accuracy_pct": pct(total_correct, n),
        "by_category": {
            cat: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy_pct": pct(v["correct"], v["total"]),
            }
            for cat, v in cat_stats.items()
        },
        "score_calibration_pct": pct(in_range_count, n),
        "avg_score_deviation": round(sum(deviations) / len(deviations), 1) if deviations else None,
        "avg_time_seconds": round(sum(times) / len(times), 2) if times else None,
        "fastest_seconds": round(min(times), 2) if times else None,
        "slowest_seconds": round(max(times), 2) if times else None,
        "dry_run": dry_run,
        "limit": limit,
    }

    full_report = {"summary": summary, "results": results}

    # ── Save results.json ─────────────────────────────────────────────────────
    with open(_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # ── Print formatted summary ───────────────────────────────────────────────
    b = summary["by_category"]
    print(f"\n{'='*62}")
    print(f"  BENCHMARK RESULTS {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*62}")
    print(f"  Overall Accuracy        : {summary['overall_accuracy_pct']}%  ({total_correct}/{n} correct)")
    print(f"  Fit Cases Accuracy      : {b['fit']['accuracy_pct']}%  ({b['fit']['correct']}/{b['fit']['total']})")
    print(f"  Mismatch Cases Accuracy : {b['mismatch']['accuracy_pct']}%  ({b['mismatch']['correct']}/{b['mismatch']['total']})")
    print(f"  Borderline Accuracy     : {b['borderline']['accuracy_pct']}%  ({b['borderline']['correct']}/{b['borderline']['total']})")
    print(f"  Score Calibration       : {summary['score_calibration_pct']}% within expected range")
    print(f"  Avg Score Deviation     : {summary['avg_score_deviation']} pts from range midpoint")
    print(f"  Avg Time Per Analysis   : {summary['avg_time_seconds']}s")
    print(f"  Fastest / Slowest       : {summary['fastest_seconds']}s / {summary['slowest_seconds']}s")
    print(f"{'='*62}")
    print(f"  Full report saved to: {_RESULTS_PATH}")
    print(f"{'='*62}\n")

    return full_report


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the AI Hiring Co-pilot on 30 test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark/evaluate.py --dry-run        # Test logic, no LLM\n"
            "  python benchmark/evaluate.py --limit 5        # Run first 5 cases only\n"
            "  python benchmark/evaluate.py                  # Full run (30-90 min)\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip all LLM/workflow calls, use stub results. Tests evaluation logic only.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Run only the first N test cases (useful for quick checks).",
    )
    args = parser.parse_args()
    run_benchmark(dry_run=args.dry_run, limit=args.limit)
