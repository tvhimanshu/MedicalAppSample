from pathlib import Path

from medrag import embedder, generator
from medrag.vector_store import VectorStore


class RAGPipeline:
    """retrieve → augment → generate."""

    def __init__(self, store: VectorStore):
        self._store = store

    @classmethod
    def load(cls, store_path: str | Path) -> "RAGPipeline":
        return cls(VectorStore.load(store_path))

    def ask(self, question: str, patient_id: str | None = None, top_k: int = 5) -> str:
        query_vec = embedder.embed(question)
        results   = self._store.search(
            query_vec,
            top_k=top_k,
            filter_key="patient_id" if patient_id else None,
            filter_val=patient_id,
        )

        if not results:
            return "No relevant records found. Ensure data has been ingested."

        return generator.generate(question, results)
