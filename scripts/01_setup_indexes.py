#!/usr/bin/env python3
"""
Creates the VectorSearchCore project, datapoints, search URL, and API key
for the MedicalRAG system. Run this once before ingesting data.

Output: .medrag_setup.json (consumed by subsequent scripts)

Usage: python scripts/01_setup_indexes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medrag.ingestion.pipeline import IngestPipeline

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]MedicalRAG[/bold cyan]  ·  Index Setup",
        subtitle="[dim]01_setup_indexes.py[/dim]",
    ))
    console.print("[dim]Creating VectorSearchCore project + 3 datapoints + search URL + API key[/dim]\n")

    pipeline = IngestPipeline()
    setup = pipeline.setup()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="bold white")
    table.add_row("Project ID",        setup["project_id"])
    table.add_row("patient_records DP", setup["datapoints"]["patient_records"]["id"])
    table.add_row("lab_results DP",     setup["datapoints"]["lab_results"]["id"])
    table.add_row("clinical_notes DP",  setup["datapoints"]["clinical_notes"]["id"])
    table.add_row("Search slug",        setup["search_url_slug"])
    table.add_row("API key",            setup["api_key"])

    console.print(Panel(table, title="[green]Setup complete", border_style="green"))
    console.print("\n[dim]Saved to .medrag_setup.json — next: python scripts/02_ingest_patients.py[/dim]")


if __name__ == "__main__":
    main()
