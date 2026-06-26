"""
RAG tool: indexes company-culture, skill-context, and interview-experience
documents in a persistent ChromaDB store and exposes them as a LlamaIndex
QueryEngineTool.

Usage:
    from tools.rag_tool import setup_rag_tool
    tool = setup_rag_tool()
"""

import sys
from pathlib import Path

import os

# Ensure the project root is importable whether this file is run directly
# (python tools/rag_tool.py) or imported as a package (from tools.rag_tool ...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore

from tools.corpus_data import CULTURE_DOCS, INTERVIEW_DOCS, SKILL_DOCS

# Reuse the LLM + embedding model from the SQL tool so the project has a single
# source of truth for model choice (gpt-oss-120b:free) and shares one loaded
# embedding model instance across tools.
from tools.sql_tool import _get_embed_model, _get_llm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_DIR = _PROJECT_ROOT / "data" / "corpus"

# On HuggingFace Spaces, SPACE_ID is set; use /data/ (persistent volume) for
# ChromaDB so the index survives container restarts.
_IS_HF_SPACE = bool(os.getenv("SPACE_ID"))
_CHROMA_DIR = (
    Path("/data/chroma_store") if _IS_HF_SPACE
    else _PROJECT_ROOT / "database" / "chroma_store"
)
_COLLECTION = "hiring_copilot_corpus"

# ── Custom QA prompt ─────────────────────────────────────────────────────────
# Vars: {context_str}, {query_str}.

_QA_PROMPT_TMPL = """\
You are a hiring-domain expert assistant. Answer the question using ONLY the
context documents provided below.

Rules:
- Base your answer strictly on the context. Do NOT invent companies, processes,
  numbers, or details that are not present in the context.
- When the answer depends on company type or stage (seed / Series A / Series B /
  MNC / product / deep tech, etc.), be specific about which type you are
  describing.
- If the context does not contain enough information to answer, say clearly:
  "The available documents don't cover this in enough detail to answer
  confidently." Do not fill the gap with assumptions.
- Be concise and practical. Use short paragraphs or bullets.

Context documents:
---------------------
{context_str}
---------------------

Question: {query_str}
Answer: \
"""

_QA_PROMPT = PromptTemplate(_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)


def build_corpus(overwrite: bool = False) -> int:
    """
    Write the 35 corpus documents (10 culture + 15 skill + 10 interview) to
    data/corpus/ as .txt files. Returns the number of files written.

    Skips files that already exist unless overwrite=True.
    """
    _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for docs in (CULTURE_DOCS, SKILL_DOCS, INTERVIEW_DOCS):
        for filename, text in docs.items():
            path = _CORPUS_DIR / filename
            if path.exists() and not overwrite:
                continue
            path.write_text(text, encoding="utf-8")
            written += 1
    return written


def _corpus_file_count() -> int:
    if not _CORPUS_DIR.exists():
        return 0
    return len(list(_CORPUS_DIR.glob("*.txt")))


def _load_or_build_index() -> VectorStoreIndex:
    """
    Load the existing ChromaDB-backed index if the collection is already
    populated; otherwise build the corpus, index it, and persist it.
    """
    # Ensure the corpus files exist on disk before any indexing.
    if _corpus_file_count() == 0:
        build_corpus()

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    collection = client.get_or_create_collection(_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = _get_embed_model()

    if collection.count() > 0:
        # Collection already populated → load, do not re-index.
        return VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )

    # Empty collection → build the index from the corpus files and persist.
    documents = SimpleDirectoryReader(
        input_dir=str(_CORPUS_DIR), required_exts=[".txt"]
    ).load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )


def setup_rag_tool() -> QueryEngineTool:
    """Return a QueryEngineTool over the culture/skill/interview corpus."""
    index = _load_or_build_index()
    query_engine = index.as_query_engine(
        llm=_get_llm(),
        text_qa_template=_QA_PROMPT,
        similarity_top_k=4,
    )

    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="culture_and_context_tool",
            description=(
                "Use this tool for queries about company culture, what skills "
                "really mean in practice, interview processes, implicit JD "
                "requirements, what companies actually value beyond listed "
                "skills. Do NOT use for specific job listings or salary data — "
                "use job_database_tool for that."
            ),
        ),
    )


def test_rag_tool() -> None:
    """Run 5 test queries and print results."""
    queries = [
        "What does a Series A startup look for in an ML engineer beyond "
        "technical skills?",
        "What does production ML experience really mean?",
        "How is the interview process different at startups vs MNCs?",
        "Is PyTorch experience transferable to TensorFlow roles?",
        "What do companies mean by strong communication skills in a technical JD?",
    ]

    print("\n" + "=" * 65)
    print("  RAG TOOL — TEST RESULTS")
    print("=" * 65)

    n = build_corpus()
    print(f"  Corpus: {_corpus_file_count()} files in data/corpus/ "
          f"({n} written this run)")

    tool = setup_rag_tool()
    engine = tool.query_engine

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
    test_rag_tool()
