"""Retrieves semantically relevant patient record chunks from VectorSearchCore."""

from __future__ import annotations
import json
from pathlib import Path

import requests

from medrag.config import Settings


class Retriever:
    def __init__(self, slug: str, api_key: str):
        self.slug    = slug
        self.api_key = api_key

    def retrieve(self, query: str, top_k: int | None = None,
                 patient_id: str | None = None) -> list[dict]:
        payload: dict = {
            "query": query,
            "topK":  top_k or Settings.RAG_TOP_K,
        }
        if patient_id:
            payload["filters"] = {"patient_id": patient_id}

        r = requests.post(
            f"{Settings.VECTOR_SEARCH_URL}/api/v1/search/{self.slug}",
            json=payload,
            headers={"X-API-Key": self.api_key},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["data"].get("results", [])

    @classmethod
    def from_setup_file(cls) -> "Retriever":
        setup_path = Path(__file__).parent.parent.parent / ".medrag_setup.json"
        if not setup_path.exists():
            raise FileNotFoundError("Run 01_setup_indexes.py first.")
        setup = json.loads(setup_path.read_text())
        return cls(slug=setup["search_url_slug"], api_key=setup["api_key"])
