#!/usr/bin/env python3
"""
Embeds all patient record documents and saves the vector store to disk.
Run once before querying.

Usage: python ingest.py
"""

import json
from pathlib import Path

from rich.console import Console
from rich.progress import track

from medrag.embedder import embed_batch
from medrag.vector_store import VectorStore, Document

console   = Console()
DATA_FILE = Path("data/patient_records.json")
STORE_OUT = Path("data/vector_store.pkl")


def main():
    records = json.loads(DATA_FILE.read_text())
    console.print(f"[dim]Loaded {len(records)} documents[/dim]")

    docs = [Document(id=r["id"], text=r["text"], metadata=r) for r in records]

    with console.status(f"Embedding {len(docs)} documents via OpenAI..."):
        embeddings = embed_batch([d.text for d in docs])

    store = VectorStore()
    for doc, emb in zip(docs, embeddings):
        doc.embedding = emb
        store.add(doc)

    store.save(STORE_OUT)
    console.print(f"[green]✓[/green] Indexed [bold]{len(store)}[/bold] documents → {STORE_OUT}")
    console.print("[dim]Next: python demo.py[/dim]")


if __name__ == "__main__":
    main()
