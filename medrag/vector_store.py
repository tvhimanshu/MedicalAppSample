"""
In-memory vector store backed by numpy cosine similarity.
Intentionally simple — swap for Pinecone/Qdrant in production.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Document:
    id:        str
    text:      str
    metadata:  dict
    embedding: list[float] = field(default_factory=list)


class VectorStore:
    def __init__(self):
        self._docs: list[Document] = []

    def add(self, doc: Document) -> None:
        self._docs.append(doc)

    def search(self, query_embedding: list[float], top_k: int = 5,
               filter_key: str | None = None,
               filter_val: str | None = None) -> list[tuple[Document, float]]:
        candidates = self._docs
        if filter_key and filter_val:
            candidates = [d for d in candidates if d.metadata.get(filter_key) == filter_val]

        if not candidates:
            return []

        q  = np.array(query_embedding, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-10

        scores = []
        for doc in candidates:
            d = np.array(doc.embedding, dtype=np.float32)
            d /= np.linalg.norm(d) + 1e-10
            scores.append((doc, float(np.dot(q, d))))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        return pickle.loads(Path(path).read_bytes())

    def __len__(self) -> int:
        return len(self._docs)
