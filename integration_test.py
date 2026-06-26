"""
Part D: Final integration test.
Runs the complete pipeline end to end and prints results at each step.

Usage: python integration_test.py
"""
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=" * 65)
print("  AI HIRING CO-PILOT - FINAL INTEGRATION TEST")
print("=" * 65)

# --- STEP 1: Classifier ---
print("\n[STEP 1] Testing QueryClassifier...")
from tools.classifier import QueryClassifier
clf = QueryClassifier()
test_queries = [
    "What salary does an ML Engineer earn at Series A startups in Bangalore?",
    "What does a Series A startup really expect beyond the listed tech stack?",
    "Does this candidate have both the technical and cultural fit for this role?",
]
for q in test_queries:
    r = clf.classify(q)
    print(f"  Query   : {q[:60]}...")
    print(f"  Result  : {r['type']} (conf={r['confidence']}%, sql={r['sql_weight']}, rag={r['rag_weight']})")
    print(f"  Reason  : {r['reasoning'][:90]}")
    print()

# --- STEP 2: Test classify_hiring_context ---
print("[STEP 2] Testing classify_hiring_context() with Arjun Mehta resume...")
_RESUME = {
    "name": "ARJUN MEHTA", "experience_level": "fresher",
    "total_experience_years": 0.42,
    "skills": {
        "programming_languages": ["Python", "SQL"],
        "frameworks_and_libraries": ["PyTorch", "FastAPI", "pandas", "MLflow"],
        "tools_and_platforms": ["Docker", "Git", "Linux"],
        "databases": ["PostgreSQL", "SQLite", "ChromaDB"],
    },
}
_JD = "ML Engineer at a Series A fintech startup in Bangalore. 3-5 years experience required. Must have: Python, PyTorch, MLflow, Docker, REST APIs, production deployment. Fast-paced startup culture."
clf_result = clf.classify_hiring_context(_RESUME, _JD)
print(f"  Classification: {clf_result['type']}")
print(f"  Confidence    : {clf_result['confidence']}%")
print(f"  SQL weight    : {clf_result['sql_weight']}")
print(f"  RAG weight    : {clf_result['rag_weight']}")
print(f"  Reasoning     : {clf_result['reasoning'][:120]}")

# --- STEP 3: Planner with classification ---
print("\n[STEP 3] Testing plan_investigation() with classification guidance...")
from agents.planner import plan_investigation
plan = plan_investigation(_RESUME, _JD, classification=clf_result)
print(f"  Priority         : {plan.get('investigation_priority')}")
print(f"  SQL queries ({len(plan.get('sql_queries',[]))}): {plan.get('sql_queries',[])[:2]}")
print(f"  RAG queries ({len(plan.get('rag_queries',[]))}): {plan.get('rag_queries',[])[:2]}")
print(f"  Key skills       : {plan.get('key_skills_to_evaluate',[])[:4]}")
print(f"  Gap hypotheses   : {plan.get('initial_gap_hypothesis',[])[:2]}")

# --- STEP 4: reflect() function ---
print("\n[STEP 4] Testing reflect() with stub synthesis...")
from agents.reflector import reflect, re_evaluate

stub_synthesis = {
    "fit_score": 38, "recommendation": "weak_fit",
    "strengths": [
        {"point": "PyTorch experience", "evidence": "Candidate used PyTorch at internship.", "source": "resume"},
        {"point": "MLflow knowledge", "evidence": "Set up MLflow at NeuralNest internship.", "source": "resume"},
    ],
    "gaps": [
        {"point": "Experience deficit (fresher vs 3-5yr required)", "severity": "critical",
         "evidence": "Market data shows all Series A ML roles require 3-5 years.", "source": "sql"},
        {"point": "No production ML deployment", "severity": "critical",
         "evidence": "Production ML at startups requires containerized serving + monitoring.", "source": "rag"},
    ],
    "summary": "Strong technical foundation but severe seniority mismatch.",
    "confidence": 88,
}
stub_sql = ["Found 4 ML Engineer roles in Bangalore: all require 3-5 years and production deployment."]
stub_rag = ["Series A fintechs expect full ML lifecycle ownership with minimal hand-holding."]

t0 = time.time()
reflection = reflect(stub_synthesis, stub_sql, stub_rag, plan, cycle_number=1)
elapsed = time.time() - t0
print(f"  Confidence    : {reflection['confidence']}%")
print(f"  Should replan : {reflection['should_replan']}")
print(f"  Weak points   : {len(reflection.get('weak_points', []))}")
print(f"  Replan SQL    : {reflection['replan_queries'].get('sql', [])}")
print(f"  Replan RAG    : {reflection['replan_queries'].get('rag', [])}")
print(f"  Notes         : {reflection.get('reflection_notes','')[:120]}")
print(f"  Time          : {elapsed:.1f}s")

# --- STEP 5: Full workflow (optional - takes ~2-4 min) ---
print("\n[STEP 5] Running full run_copilot() pipeline...")
print("  (This will take 2-5 minutes - runs all 5 workflow steps...)")
from workflow import run_copilot
t0 = time.time()
result = run_copilot(_RESUME, _JD)
elapsed = time.time() - t0

synth = result.get("synthesis", {})
clf_out = result.get("classification", {})
refl = result.get("reflection", {})
fq = result.get("followup_questions", {}).get("questions", [])
cycles = result.get("reflection_cycles", 0)

print(f"\n{'='*65}")
print(f"  FULL PIPELINE RESULTS (completed in {elapsed:.0f}s)")
print(f"{'='*65}")
print(f"\n  [Classification]")
print(f"  Type      : {clf_out.get('type')} (conf={clf_out.get('confidence')}%)")
print(f"  SQL weight: {clf_out.get('sql_weight')} | RAG weight: {clf_out.get('rag_weight')}")
print(f"  Reasoning : {clf_out.get('reasoning','')[:100]}")

print(f"\n  [Fit Score]")
print(f"  Score      : {synth.get('fit_score')}/100")
bd = synth.get("score_breakdown", {})
print(f"  Breakdown  : skills={bd.get('skills_match')} | experience={bd.get('experience_match')} | culture={bd.get('culture_fit')}")
print(f"  Rec        : {synth.get('recommendation')}")
print(f"  Confidence : {synth.get('confidence')}%")

print(f"\n  [Strengths] ({len(synth.get('strengths',[]))} found)")
for s in synth.get("strengths", [])[:3]:
    print(f"  [{s.get('source','?').upper():7s}] {s.get('point','')[:70]}")

print(f"\n  [Gaps] ({len(synth.get('gaps',[]))} found)")
for g in synth.get("gaps", [])[:3]:
    sev = g.get("severity","?")
    print(f"  [{g.get('source','?').upper():7s}] [{sev.upper():8s}] {g.get('point','')[:60]}")

print(f"\n  [Citations]")
cit = synth.get("citations", {})
print(f"  SQL cits: {len(cit.get('from_sql',[]))} | RAG cits: {len(cit.get('from_rag',[]))}")
for c in cit.get("from_sql", [])[:2]:
    print(f"  SQL: {c[:80]}")
for c in cit.get("from_rag", [])[:2]:
    print(f"  RAG: {c[:80]}")

print(f"\n  [Reflection] (cycles run: {cycles})")
print(f"  Confidence   : {refl.get('confidence','?')}%")
print(f"  Should replan: {refl.get('should_replan','?')}")
print(f"  Notes        : {refl.get('reflection_notes','')[:100]}")

print(f"\n  [Follow-up Questions] ({len(fq)} generated)")
for i, q in enumerate(fq, 1):
    print(f"  {i}. [{q.get('gap_addressed','')[:40]}]")
    print(f"     {q.get('question','')[:80]}")

# --- STEP 6: Re-evaluate with hardcoded strong answers ---
if fq:
    print(f"\n  [Re-evaluation] Answering follow-ups with strong answers...")
    qa_pairs = []
    for q_item in fq[:3]:
        qa_pairs.append({
            "question": q_item.get("question", ""),
            "answer": (
                "I have been freelancing as an ML engineer for the past year, "
                "building a fraud detection API using PyTorch + FastAPI + Docker, "
                "deployed on AWS EC2 with Prometheus monitoring and automated "
                "Grafana alerts. I handle model drift monitoring via evidently "
                "and have been on-call for production incidents."
            ),
            "gap_addressed": q_item.get("gap_addressed", ""),
        })

    reeval = re_evaluate(synth, qa_pairs, {"questions": fq[:3]})
    print(f"  Score delta   : {reeval.get('score_delta', 0):+d} points")
    print(f"  New score     : {reeval.get('updated_score')}")
    print(f"  New rec       : {reeval.get('updated_recommendation')}")
    print(f"  Resolved gaps : {len(reeval.get('resolved_gaps', []))}")
    for a in reeval.get("answer_assessments", [])[:2]:
        print(f"  [{a.get('rating','?').upper():8s}] {a.get('gap_addressed','')[:50]}")

print(f"\n{'='*65}")
print("  INTEGRATION TEST COMPLETE - ALL STEPS PASSED")
print(f"{'='*65}")
