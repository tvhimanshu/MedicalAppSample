#!/usr/bin/env python3
"""
Ingests all patient data into VectorSearchCore.

What gets ingested:
  patients.json      → chunked into demographics, conditions, medication, vitals chunks
  lab_results.json   → one chunk per lab panel
  clinical_notes.json → one chunk per clinical note

Each chunk is HMAC-signed before upload.

Usage: python scripts/02_ingest_patients.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medrag.ingestion.pipeline import IngestPipeline

console = Console()
DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    console.print(Panel.fit(
        "[bold cyan]MedicalRAG[/bold cyan]  ·  Patient Data Ingestion",
        subtitle="[dim]02_ingest_patients.py[/dim]",
    ))
    console.print("[dim]Chunking and uploading 6 patients · labs · clinical notes[/dim]\n")

    pipeline = IngestPipeline()
    counts   = pipeline.run(DATA_DIR)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bold white", justify="right")
    table.add_row("Patient record chunks", str(counts["patient_chunks"]))
    table.add_row("Lab result chunks",     str(counts["lab_chunks"]))
    table.add_row("Clinical note chunks",  str(counts["note_chunks"]))
    table.add_row("Total",                 str(sum(counts.values())))

    console.print(Panel(table, title="[green]Ingestion complete", border_style="green"))
    console.print("\n[dim]Embedding pipeline running async — next: python scripts/03_doctor_query_demo.py[/dim]")


if __name__ == "__main__":
    main()
