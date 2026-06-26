---
title: HireIQ
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
---

# AI Hiring Co-pilot 🤖

> An evidence-based multi-agent system that analyzes candidate fit for a role by reasoning across structured job data (SQL) and company culture documents (RAG) — producing a cited fit score, identified gaps, follow-up questions, and a re-evaluation loop.

## What It Does

The AI Hiring Co-pilot takes a candidate's resume (PDF) and a job description, then runs a 5-step agentic pipeline to produce a deep fit analysis. Unlike generic resume scanners, it grounds every strength and gap in real evidence: market salary data from a SQL database and company culture context from a RAG corpus. Candidates can answer follow-up questions to refine their score, and a chat assistant lets them explore the analysis in natural language. A Recruiter mode supports batch analysis of multiple candidates against one role.

## Architecture

```
PDF Resume
    │
    ▼
┌─────────────────┐
│  Resume Parser  │  pdfplumber + LLM → structured JSON profile
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Planner      │  LLM generates SQL + RAG query plan before any tools run
└────────┬────────┘
         │
    ┌────┴────┐
    │         │   (parallel execution via asyncio.gather)
    ▼         ▼
┌───────┐  ┌───────┐
│  SQL  │  │  RAG  │
│ Tool  │  │ Tool  │
│       │  │       │
│SQLite │  │Chroma │
│50 jobs│  │35 docs│
└───┬───┘  └───┬───┘
    └────┬─────┘
         │
         ▼
┌─────────────────┐
│  Synthesizer    │  Fit score 0-100, strengths/gaps with citations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reflector     │  Quality check → optional replan (max 2 cycles)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Follow-up Gen   │  Up to 3 targeted questions for critical/moderate gaps
└────────┬────────┘
         │
    User answers follow-up questions
         │
         ▼
┌─────────────────┐
│  Re-evaluation  │  LLM scores answers → updates fit score + resolves gaps
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chat (Q&A)    │  Free-form chat grounded in full session memory
└─────────────────┘
```

## Tech Stack

| Component | Tool | Why This Choice |
|---|---|---|
| LLM | `openai/gpt-oss-120b:free` via OpenRouter | Free-tier capable 120B MoE model; reliable latency |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace local) | Zero API cost, runs locally, strong retrieval quality |
| Vector Store | ChromaDB (local persistent) | Simple, file-based, no server needed |
| SQL | SQLite + SQLAlchemy + LlamaIndex NLSQLTableQueryEngine | Natural-language queries over job listings DB |
| Orchestration | LlamaIndex Workflows (event-driven async) | Native parallel tool execution, typed events |
| UI | Streamlit | Rapid prototyping; native file upload + chat |
| Resume Parsing | pdfplumber + LLM | Handles multi-column layouts; structured JSON extraction |
| Cross-session Memory | SQLite (profiles.db keyed by PDF MD5 hash) | Recognizes returning candidates; deduplicates follow-ups |
| In-session Memory | LlamaIndex `ChatMemoryBuffer` (4k token limit) | Grounded context for chat without hallucination |
| Deployment | Hugging Face Spaces (free CPU tier) | Zero-cost public hosting with persistent `/data/` volume |

## Benchmark Results

> Run on 30 hand-crafted test cases: 10 fit (strong_fit, 75-95), 10 mismatch (not_fit/weak_fit, 20-45), 10 borderline (possible_fit, 45-70).
> Execute `python benchmark/evaluate.py` to reproduce.

| Metric | Result |
|---|---|
| Overall Accuracy | — |
| Fit Cases Accuracy | — |
| Mismatch Cases Accuracy | — |
| Borderline Accuracy | — |
| Score Calibration (% within range) | — |
| Avg Score Deviation | — |
| Avg Time Per Analysis | — |

*Run `python benchmark/evaluate.py` to populate these numbers.*

## How to Run Locally

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An [OpenRouter](https://openrouter.ai) API key (free tier works)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Priya584/Hiring_copilot.git
cd Hiring_copilot

# 2. Create virtual environment
uv venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and set: OPENROUTER_API_KEY=sk-or-...

# 5. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> **Demo mode**: If no API key is set, the app automatically shows a pre-computed example analysis with no API calls required.

### Run the Benchmark

```bash
# Test benchmark logic without LLM calls (~5 sec)
python benchmark/evaluate.py --dry-run

# Run 5 cases to verify end-to-end pipeline (~5-10 min)
python benchmark/evaluate.py --limit 5

# Full 30-case benchmark (~30-90 min)
python benchmark/evaluate.py
```

## Key Features

- **Planning-first agentic pipeline**: The Planner generates SQL + RAG queries *before* any tools are called — making tool usage targeted, not brute-force
- **Parallel tool execution**: SQL and RAG queries run concurrently via `asyncio.gather`, halving pipeline latency
- **Calibrated scoring**: Hard caps prevent overoptimistic scores for seniority mismatches (e.g., fresher → senior role is capped at 45/100)
- **Evidence-grounded output**: Every strength and gap cites its source (`sql`, `rag`, or `resume`) with a verbatim or paraphrased evidence quote
- **Re-evaluation loop**: Candidates answer up to 3 follow-up questions; the Reflector scores each answer against hidden criteria and adjusts the fit score (+2 to +10 per answer)
- **Cross-session memory**: Returning candidates are recognized by PDF MD5 hash — previously answered follow-up questions are never asked again
- **Demo mode**: No API key? No problem. A pre-computed realistic analysis loads instantly, demonstrating the full UI
- **Recruiter batch mode**: Upload multiple PDFs, analyze all against one role, download results as CSV
- **Robust SQL**: Custom `RobustSQLParser` + `RobustSQLQueryEngine` extract valid SELECT statements even when the LLM adds prose, with up to 2 auto-retries

## Project Structure

```
hiring-copilot/
├── app.py                    # Streamlit UI (Candidate + Recruiter modes)
├── workflow.py               # LlamaIndex Workflow orchestration (5 steps)
├── agents/
│   ├── planner.py            # Investigation plan generator
│   ├── resume_parser.py      # PDF → structured JSON profile
│   ├── synthesizer.py        # Fit score + strengths/gaps + citations
│   ├── reflector.py          # Answer scoring + score re-evaluation
│   └── followup.py           # Follow-up question generator
├── tools/
│   ├── sql_tool.py           # NL→SQL engine over jobs.db
│   ├── rag_tool.py           # ChromaDB RAG over culture/skill/interview docs
│   └── corpus_data.py        # 35 hardcoded corpus documents
├── memory/
│   ├── profile_store.py      # Cross-session SQLite persistence
│   └── chat_handler.py       # In-session chat Q&A
├── data/
│   ├── corpus/               # Culture/skill/interview .txt files
│   ├── processed/            # DB creation script (create_database.py)
│   └── demo_result.json      # Pre-computed demo analysis
├── database/
│   └── jobs.db               # SQLite — 50 jobs, 20 companies
├── benchmark/
│   ├── test_cases.json       # 30 labeled test cases
│   └── evaluate.py           # Benchmark runner with metrics
├── requirements.txt
└── .env.example
```

## Contributing

Pull requests welcome. For significant changes, please open an issue first.

