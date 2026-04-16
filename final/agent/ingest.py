"""
Ingest PDFs from agent/knowledge/ into a persistent ChromaDB vector store.

Uses a parent-document pattern:
  - Full pages are stored as parents in agent/knowledge/.parents/
  - Each page is also split into small child chunks embedded in ChromaDB
  - At query time, child chunks are matched (sharp embeddings), but the full
    parent page is returned to the LLM (full context)

Run from final/ whenever you add new PDFs:
    python agent/ingest.py

Re-running is safe — existing IDs are skipped.
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CHROMA_DIR    = KNOWLEDGE_DIR / ".chroma"
PARENTS_DIR   = KNOWLEDGE_DIR / ".parents"   # stores full page text keyed by parent_id

EMBED_MODEL   = "nomic-embed-text"
CHILD_SIZE    = 300   # small chunks — sharp embeddings for retrieval
CHILD_OVERLAP = 50


def clean_text(text: str) -> str:
    """Rejoin words broken across lines by PDF layout hyphens."""
    return re.sub(r'-\s*\n\s*', '', text)


def ingest() -> None:
    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {KNOWLEDGE_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    PARENTS_DIR.mkdir(exist_ok=True)

    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
    )
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db         = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name}...")
        reader = PdfReader(str(pdf_path))

        # Extract title from first non-empty line of page 0
        first_page_text = reader.pages[0].extract_text() or ""
        title = next(
            (line.strip() for line in first_page_text.splitlines() if line.strip()),
            pdf_path.stem,
        )
        title_prefix = f"[Paper: {title}]\n\n"

        child_docs = []
        child_ids  = []
        child_idx  = 0

        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            text     = clean_text(raw_text)
            if not text.strip():
                continue

            # ── Parent: store full page text to disk ──────────────────────────
            parent_id   = f"{pdf_path.stem}__p{page_num}"
            parent_file = PARENTS_DIR / f"{parent_id}.json"

            if not parent_file.exists():
                parent_file.write_text(json.dumps({
                    "parent_id": parent_id,
                    "source":    str(pdf_path),
                    "page":      page_num,
                    "title":     title,
                    "text":      title_prefix + text,
                }))

            # ── Children: small chunks embedded into ChromaDB ─────────────────
            for chunk in splitter.split_text(text):
                if not chunk.strip():
                    continue
                child_docs.append(Document(
                    page_content=title_prefix + chunk,
                    metadata={
                        "source":    str(pdf_path),
                        "page":      page_num,
                        "parent_id": parent_id,
                    },
                ))
                child_ids.append(f"{parent_id}__c{child_idx}")
                child_idx += 1

        db.add_documents(documents=child_docs, ids=child_ids)
        print(f"    → {child_idx} child chunks, {len(reader.pages)} parent pages")

    print(f"\nDone. Vector DB at {CHROMA_DIR}, parents at {PARENTS_DIR}")


if __name__ == "__main__":
    ingest()
