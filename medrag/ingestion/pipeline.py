"""Orchestrates the full data ingestion pipeline into VectorSearchCore."""

from __future__ import annotations
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import track

from medrag.models.patient import Patient, LabResult, ClinicalNote
from medrag.ingestion.chunker import chunk_patient, chunk_lab_result, chunk_clinical_note
from medrag.ingestion.vector_client import VectorSearchClient

console = Console()

# Schema definitions for each datapoint
PATIENT_SCHEMA = [
    {"name": "patient_id",      "type": "string", "inVector": False},
    {"name": "patient_name",    "type": "string", "inVector": False},
    {"name": "chunk_type",      "type": "string", "inVector": False},
    {"name": "medication_name", "type": "string", "inVector": False},
    {"name": "date",            "type": "string", "inVector": False},
    {"name": "text",            "type": "string", "inVector": True},
]

LAB_SCHEMA = [
    {"name": "patient_id", "type": "string", "inVector": False},
    {"name": "chunk_type", "type": "string", "inVector": False},
    {"name": "date",       "type": "string", "inVector": False},
    {"name": "panel",      "type": "string", "inVector": False},
    {"name": "text",       "type": "string", "inVector": True},
]

NOTE_SCHEMA = [
    {"name": "patient_id", "type": "string", "inVector": False},
    {"name": "chunk_type", "type": "string", "inVector": False},
    {"name": "date",       "type": "string", "inVector": False},
    {"name": "author",     "type": "string", "inVector": False},
    {"name": "note_type",  "type": "string", "inVector": False},
    {"name": "text",       "type": "string", "inVector": True},
]


class IngestPipeline:
    """
    Full ingest pipeline.

    Usage:
        pipeline = IngestPipeline()
        pipeline.setup()           # creates VectorSearchCore project + datapoints
        pipeline.run(data_dir)     # chunks + uploads all patient data
    """

    SETUP_FILE = Path(__file__).parent.parent.parent / ".medrag_setup.json"

    def __init__(self):
        self.client = VectorSearchClient()
        self.setup_data: dict = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> dict:
        with console.status("Authenticating with VectorSearchCore..."):
            self.client.authenticate()
        console.print("[green]✓[/green] Authenticated")

        with console.status("Creating project..."):
            project = self.client.create_project(
                name="MedicalRAG — Patient Decision Support",
                description=(
                    "RAG-powered clinical decision support system. "
                    "Indexes patient demographics, lab results, and clinical notes "
                    "for semantic retrieval by attending physicians."
                ),
            )
        console.print(f"[green]✓[/green] Project  [dim]{project['id']}[/dim]")

        with console.status("Creating datapoints..."):
            dp_patients = self.client.create_datapoint(
                project["id"], "patient_records", "PATIENT_RECORDS", PATIENT_SCHEMA
            )
            dp_labs = self.client.create_datapoint(
                project["id"], "lab_results", "LAB_RESULTS", LAB_SCHEMA
            )
            dp_notes = self.client.create_datapoint(
                project["id"], "clinical_notes", "CLINICAL_NOTES", NOTE_SCHEMA
            )
        console.print("[green]✓[/green] Datapoints created (patient_records, lab_results, clinical_notes)")

        secrets = {
            "patients": self.client.get_webhook_secret(project["id"], dp_patients["id"]),
            "labs":     self.client.get_webhook_secret(project["id"], dp_labs["id"]),
            "notes":    self.client.get_webhook_secret(project["id"], dp_notes["id"]),
        }

        with console.status("Creating unified search URL..."):
            search_url = self.client.create_search_url(
                project["id"],
                name="Patient Records Search",
                datapoint_ids=[dp_patients["id"], dp_labs["id"], dp_notes["id"]],
                top_k=8,
            )
        console.print(f"[green]✓[/green] Search URL [bold]{search_url['slug']}[/bold]")

        with console.status("Generating API key..."):
            api_key = self.client.create_api_key("MedRAG — Search")
        console.print("[green]✓[/green] API key generated")

        self.setup_data = {
            "project_id":       project["id"],
            "datapoints": {
                "patient_records": {"id": dp_patients["id"], "webhook_secret": secrets["patients"]},
                "lab_results":     {"id": dp_labs["id"],     "webhook_secret": secrets["labs"]},
                "clinical_notes":  {"id": dp_notes["id"],    "webhook_secret": secrets["notes"]},
            },
            "search_url_slug": search_url["slug"],
            "api_key":         api_key,
        }
        self.SETUP_FILE.write_text(json.dumps(self.setup_data, indent=2))
        return self.setup_data

    def load_setup(self) -> dict:
        if not self.SETUP_FILE.exists():
            raise FileNotFoundError("Run setup() first or execute scripts/01_setup_indexes.py")
        self.setup_data = json.loads(self.SETUP_FILE.read_text())
        self.client.authenticate()
        return self.setup_data

    # ── Ingest ────────────────────────────────────────────────────────────────

    def run(self, data_dir: Path) -> dict[str, int]:
        if not self.setup_data:
            self.load_setup()

        dp = self.setup_data["datapoints"]
        counts = {"patient_chunks": 0, "lab_chunks": 0, "note_chunks": 0}

        # Patient records
        patients = [Patient(**p) for p in json.loads((data_dir / "patients.json").read_text())]
        all_patient_chunks = [c for p in patients for c in chunk_patient(p)]

        for chunk in track(all_patient_chunks, description="[cyan]Ingesting patient records"):
            self.client.ingest(dp["patient_records"]["id"], dp["patient_records"]["webhook_secret"], chunk)
            counts["patient_chunks"] += 1
            time.sleep(0.04)

        # Lab results
        labs = [LabResult(**l) for l in json.loads((data_dir / "lab_results.json").read_text())]
        for lab in track(labs, description="[cyan]Ingesting lab results    "):
            self.client.ingest(dp["lab_results"]["id"], dp["lab_results"]["webhook_secret"],
                               chunk_lab_result(lab))
            counts["lab_chunks"] += 1
            time.sleep(0.04)

        # Clinical notes
        notes = [ClinicalNote(**n) for n in json.loads((data_dir / "clinical_notes.json").read_text())]
        for note in track(notes, description="[cyan]Ingesting clinical notes "):
            self.client.ingest(dp["clinical_notes"]["id"], dp["clinical_notes"]["webhook_secret"],
                               chunk_clinical_note(note))
            counts["note_chunks"] += 1
            time.sleep(0.04)

        return counts
