"""RAG orchestration: retrieve → augment → generate → structured response."""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from medrag.rag.retriever import Retriever
from medrag.rag import generator


@dataclass
class RAGResponse:
    question:   str
    answer:     str
    sources:    list[dict] = field(default_factory=list)
    patient_id: str | None = None
    model:      str = ""
    latency_ms: int = 0

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def has_alerts(self) -> bool:
        return "⚠️" in self.answer


class RAGPipeline:
    """
    High-level RAG interface for the medical query UI.

    Usage:
        rag = RAGPipeline.from_setup_file()

        # Ask about a specific patient
        resp = rag.ask("What medications is this patient on?", patient_id="pat-001")

        # Cross-patient query
        resp = rag.ask("Which patients have abnormal HbA1c values?")

        print(resp.answer)
    """

    def __init__(self, retriever: Retriever):
        self._retriever = retriever

    @classmethod
    def from_setup_file(cls) -> "RAGPipeline":
        return cls(retriever=Retriever.from_setup_file())

    def ask(self, question: str, patient_id: str | None = None,
            top_k: int | None = None) -> RAGResponse:
        t0 = time.monotonic()

        chunks = self._retriever.retrieve(question, top_k=top_k, patient_id=patient_id)

        if not chunks:
            return RAGResponse(
                question=question,
                answer=(
                    "**Summary:** No relevant records found for this query.\n\n"
                    "**Supporting Evidence:** None retrieved — "
                    "ensure data has been ingested and the vector pipeline is active."
                ),
                patient_id=patient_id,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

        answer = generator.generate(question, chunks)
        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            patient_id=patient_id,
            model=generator.Settings.OPENAI_MODEL,
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
