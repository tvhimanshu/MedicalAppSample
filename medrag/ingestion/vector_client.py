"""Authenticated client for the VectorSearchCore REST API."""

from __future__ import annotations
import hashlib
import hmac
import json

import requests

from medrag.config import Settings


class VectorSearchClient:
    def __init__(self):
        self._s = requests.Session()
        self._s.headers["Content-Type"] = "application/json"

    # ── Auth ──────────────────────────────────────────────────────────────────

    def authenticate(self) -> None:
        r = self._s.post(f"{Settings.VECTOR_SEARCH_URL}/api/auth/login",
                         json={"email": Settings.VECTOR_SEARCH_EMAIL,
                               "password": Settings.VECTOR_SEARCH_PASSWORD})
        if r.status_code == 401:
            self._s.post(
                f"{Settings.VECTOR_SEARCH_URL}/api/auth/register",
                json={
                    "name": "MedRAG System",
                    "email": Settings.VECTOR_SEARCH_EMAIL,
                    "password": Settings.VECTOR_SEARCH_PASSWORD,
                    "plan": "FREE",
                },
            ).raise_for_status()
            r = self._s.post(f"{Settings.VECTOR_SEARCH_URL}/api/auth/login",
                             json={"email": Settings.VECTOR_SEARCH_EMAIL,
                                   "password": Settings.VECTOR_SEARCH_PASSWORD})
        r.raise_for_status()
        self._s.headers["Authorization"] = "Bearer " + r.json()["data"]["accessToken"]

    # ── Projects ──────────────────────────────────────────────────────────────

    def create_project(self, name: str, description: str) -> dict:
        return self._post("/api/projects", {"name": name, "description": description,
                                            "embeddingModel": Settings.OPENAI_EMBED_MODEL})["project"]

    # ── Datapoints ────────────────────────────────────────────────────────────

    def create_datapoint(self, project_id: str, name: str, doc_type: str,
                         schema: list[dict]) -> dict:
        return self._post(
            f"/api/projects/{project_id}/datapoints",
            {"name": name, "type": doc_type, "sourceType": "WEBHOOK", "schema": schema},
        )["datapoint"]

    def get_webhook_secret(self, project_id: str, datapoint_id: str) -> str:
        r = self._s.get(
            f"{Settings.VECTOR_SEARCH_URL}/api/projects/{project_id}/datapoints/{datapoint_id}"
        )
        r.raise_for_status()
        return r.json()["data"]["datapoint"]["webhookSecret"]

    # ── Search URLs ───────────────────────────────────────────────────────────

    def create_search_url(self, project_id: str, name: str,
                          datapoint_ids: list[str], top_k: int = 6) -> dict:
        return self._post(
            f"/api/projects/{project_id}/search-urls",
            {"name": name, "datapointIds": datapoint_ids,
             "topK": top_k, "minScore": Settings.RAG_MIN_SCORE},
        )["searchUrl"]

    # ── API Keys ──────────────────────────────────────────────────────────────

    def create_api_key(self, label: str) -> str:
        return self._post(
            "/api/settings/api-keys",
            {"label": label, "keyType": "SEARCH", "permissions": ["search:read"]},
        )["rawKey"]

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest(self, datapoint_id: str, webhook_secret: str, record: dict) -> dict:
        body = {"event": "record.created", "data": record}
        sig  = self._sign(webhook_secret, body)
        r = requests.post(
            f"{Settings.VECTOR_SEARCH_URL}/api/ingest/{datapoint_id}",
            json=body,
            headers={"X-VectraSearch-Signature": sig},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["data"]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, slug: str, api_key: str, query: str,
               top_k: int | None = None, filters: dict | None = None) -> list[dict]:
        payload: dict = {"query": query, "topK": top_k or Settings.RAG_TOP_K}
        if filters:
            payload["filters"] = filters
        r = requests.post(
            f"{Settings.VECTOR_SEARCH_URL}/api/v1/search/{slug}",
            json=payload,
            headers={"X-API-Key": api_key},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()["data"].get("results", [])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        r = self._s.post(f"{Settings.VECTOR_SEARCH_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()["data"]

    @staticmethod
    def _sign(secret: str, body: dict) -> str:
        raw    = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        digest = hmac.new(secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
        return f"sha256={digest}"
