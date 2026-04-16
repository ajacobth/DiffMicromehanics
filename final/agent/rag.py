"""
RAG retriever and LangChain tool for the DiffMicromechanics agent.

Uses a parent-document pattern:
  - Child chunks (300 chars) are matched in ChromaDB for sharp retrieval
  - The full parent page is returned to the LLM for complete context

Exposes:
    search_knowledge_base  — @tool the agent calls to search the knowledge base
    get_retriever()        — returns the raw ChromaDB retriever (for testing)
"""

import json
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR    = KNOWLEDGE_DIR / ".chroma"
PARENTS_DIR   = KNOWLEDGE_DIR / ".parents"

EMBED_MODEL   = "nomic-embed-text"
TOP_K         = 6   # child chunks to retrieve — deduped to unique parent pages

# ── Lazy singleton ─────────────────────────────────────────────────────────────

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Vector DB not found at {CHROMA_DIR}. "
                "Run `python agent/ingest.py` first."
            )
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        _retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever


def _load_parent(parent_id: str) -> dict:
    """Load full parent page from disk. Returns empty dict if not found."""
    parent_file = PARENTS_DIR / f"{parent_id}.json"
    if parent_file.exists():
        return json.loads(parent_file.read_text())
    return {}


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the material datasheets and reference documents for relevant information.

    Use this when you need specific material properties (moduli, CTE, conductivity),
    model assumptions, equipment descriptions, or theory from the uploaded PDFs.
    Returns the most relevant full pages from the knowledge base.
    """
    try:
        retriever = get_retriever()
        child_docs = retriever.invoke(query)
    except Exception as e:
        print(f"[DEBUG] tool exception: {type(e).__name__}: {e}")
        return f"Knowledge base error: {type(e).__name__}: {e}"

    if not child_docs:
        return "No relevant information found in the knowledge base."

    # Deduplicate to unique parent pages (preserving rank order)
    seen_parents = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen_parents:
            seen_parents.append(pid)

    # Load full parent pages
    parts = []
    for i, parent_id in enumerate(seen_parents, 1):
        parent = _load_parent(parent_id)
        if not parent:
            continue
        source = Path(parent["source"]).name
        page   = parent["page"]
        text   = parent["text"]
        parts.append(f"[{i}] {source} (page {page}):\n{text}")

    if not parts:
        return "No relevant information found in the knowledge base."

    result = "\n\n---\n\n".join(parts)
    pages  = [f"{Path(_load_parent(p).get('source','?')).name}:p{_load_parent(p).get('page','?')}"
              for p in seen_parents]
    print(f"[DEBUG] tool returned {len(parts)} parent pages: {pages}")
    return result
