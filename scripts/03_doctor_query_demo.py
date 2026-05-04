#!/usr/bin/env python3
"""
Demonstrates the MedicalRAG system with realistic physician queries.

Runs two modes:
  - Patient-scoped:  query filtered to a single patient's records
  - Cross-patient:   query across all patients (useful for cohort analysis)

Usage:
  python scripts/03_doctor_query_demo.py                     # run full demo
  python scripts/03_doctor_query_demo.py --interactive       # free-text input
  python scripts/03_doctor_query_demo.py --patient pat-001   # filter to one patient
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from medrag.rag.pipeline import RAGPipeline
from medrag.utils.display import render_rag_response, render_sources_table

console = Console()

# ── Demo Query Suites ─────────────────────────────────────────────────────────

PATIENT_QUERIES = [
    # (patient_id, question)
    ("pat-001", "What is this patient's current HbA1c and is it at target?"),
    ("pat-001", "Does this patient have any drug allergies I should know about before prescribing an antibiotic?"),
    ("pat-003", "What is James Wilson's current INR and what is the therapeutic target for his condition?"),
    ("pat-003", "Is it safe to prescribe ibuprofen for this patient's post-op pain?"),
    ("pat-005", "Summarise Harold Thompson's active diagnoses and current medications."),
    ("pat-005", "Why was furosemide dose increased and what monitoring is needed?"),
    ("pat-006", "Is Emily's gestational diabetes under control based on the latest records?"),
    ("pat-002", "What thyroid lab abnormality was found and what was the plan?"),
]

CROSS_PATIENT_QUERIES = [
    "Which patients have abnormal kidney function (elevated creatinine or low eGFR)?",
    "List all patients currently on insulin therapy and their diagnoses.",
    "Which patients have a documented allergy to NSAIDs or aspirin?",
    "Are there any patients with critically elevated BNP values suggesting heart failure?",
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_demo(rag: RAGPipeline, show_sources: bool = False) -> None:
    console.rule("[bold cyan]Patient-Scoped Queries")

    for patient_id, question in PATIENT_QUERIES:
        resp = rag.ask(question, patient_id=patient_id)
        render_rag_response(resp)
        if show_sources:
            render_sources_table(resp.sources)
        console.print()

    console.rule("[bold magenta]Cross-Patient Cohort Queries")

    for question in CROSS_PATIENT_QUERIES:
        resp = rag.ask(question)
        render_rag_response(resp)
        if show_sources:
            render_sources_table(resp.sources)
        console.print()


def run_interactive(rag: RAGPipeline) -> None:
    console.print(Panel.fit(
        "Type a clinical question. Leave patient ID blank for cross-patient search.\n"
        "Type [bold]exit[/bold] to quit.",
        title="[cyan]Interactive Mode",
    ))

    while True:
        console.print()
        question = Prompt.ask("[bold cyan]Question")
        if question.lower() in ("exit", "quit", "q"):
            break

        patient_id = Prompt.ask("[dim]Patient ID (leave blank for all patients)[/dim]",
                                default="") or None

        resp = rag.ask(question, patient_id=patient_id or None)
        render_rag_response(resp)
        render_sources_table(resp.sources)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="Enable free-text interactive query mode")
    parser.add_argument("--patient",     default=None,
                        help="Run all demo queries scoped to one patient ID")
    parser.add_argument("--sources",     action="store_true",
                        help="Show retrieved source chunks below each answer")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]MedicalRAG[/bold cyan]  ·  Clinical Decision Support Demo",
        subtitle="[dim]03_doctor_query_demo.py[/dim]",
    ))

    with console.status("Loading RAG pipeline..."):
        rag = RAGPipeline.from_setup_file()
    console.print("[green]✓[/green] Pipeline ready\n")

    if args.interactive:
        run_interactive(rag)
    elif args.patient:
        console.rule(f"[bold]Queries for patient {args.patient}")
        for _, question in PATIENT_QUERIES:
            resp = rag.ask(question, patient_id=args.patient)
            render_rag_response(resp)
            if args.sources:
                render_sources_table(resp.sources)
            console.print()
    else:
        run_demo(rag, show_sources=args.sources)


if __name__ == "__main__":
    main()
