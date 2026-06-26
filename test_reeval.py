"""Quick re-evaluate test — uses the stub synthesis from the integration test."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from agents.reflector import re_evaluate

synth = {
    "fit_score": 40, "recommendation": "not_fit",
    "score_breakdown": {"skills_match": 75, "experience_match": 20, "culture_fit": 25},
    "strengths": [
        {"point": "PyTorch experience", "evidence": "Candidate used PyTorch at internship.", "source": "resume"},
    ],
    "gaps": [
        {"point": "Required 3-5 years ML engineering experience", "severity": "critical",
         "evidence": "Role requires 3-5 years; candidate has 5 months.", "source": "resume"},
        {"point": "No production ML deployment experience", "severity": "moderate",
         "evidence": "Production ML means containerized serving + monitoring.", "source": "rag"},
        {"point": "No end-to-end REST API development", "severity": "moderate",
         "evidence": "Role requires FastAPI production serving.", "source": "rag"},
    ],
}

fq = [
    {"question": "Walk me through a production ML system you built end-to-end.",
     "gap_addressed": "Required 3-5 years ML engineering experience",
     "what_good_answer_looks_like": "Describes real production deployment with monitoring, real traffic, CI/CD."},
    {"question": "Describe a specific ML model deployment from code to production.",
     "gap_addressed": "No production ML deployment experience",
     "what_good_answer_looks_like": "Docker, FastAPI serving, logging, at least one incident handled."},
    {"question": "Share a production-grade REST API you designed.",
     "gap_addressed": "No end-to-end REST API development",
     "what_good_answer_looks_like": "Auth, error handling, documented endpoints, deployed with real users."},
]

qa_pairs = [
    {
        "question": q["question"],
        "answer": (
            "I freelanced as an ML engineer for a year, built a fraud detection API "
            "with PyTorch + FastAPI + Docker, deployed on AWS EC2 with Prometheus "
            "monitoring, Grafana alerts, and evidently for data drift. "
            "I handled 2 production incidents and own the weekly retraining pipeline."
        ),
        "gap_addressed": q["gap_addressed"],
    }
    for q in fq
]

print("Running re_evaluate()...")
reeval = re_evaluate(synth, qa_pairs, {"questions": fq})
print(f"  Score delta  : {reeval.get('score_delta', 0):+d} points")
print(f"  New score    : {reeval.get('updated_score')}")
print(f"  New rec      : {reeval.get('updated_recommendation')}")
print(f"  Resolved gaps: {len(reeval.get('resolved_gaps', []))}")
print(f"  Assessments  :")
for a in reeval.get("answer_assessments", []):
    print(f"    [{a.get('rating','?').upper():8s}] +{a.get('score_increment',0)}pt  {a.get('gap_addressed','')[:55]}")
print("\nre_evaluate() PASSED")
