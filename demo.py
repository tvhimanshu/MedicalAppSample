#!/usr/bin/env python3
"""
Doctor query demo for the MedicalRAG system.

Usage:
  python demo.py                         # run preset demo queries
  python demo.py --interactive           # free-text query mode
  python demo.py --patient pat-005       # scope all queries to one patient
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from medrag.pipeline import RAGPipeline

console     = Console()
STORE_PATH  = Path("data/vector_store.pkl")

DEMO_QUERIES = [
    ("pat-001", "What is Robert Chen's latest HbA1c and is it at target for his condition?"),
    ("pat-001", "Does this patient have any allergies I should know before prescribing antibiotics?"),
    ("pat-003", "What is James Wilson's current INR and is anticoagulation therapeutic?"),
    ("pat-003", "Can I prescribe ibuprofen for this patient's post-operative pain?"),
    ("pat-005", "Why was Harold Thompson's furosemide dose increased and what should I monitor?"),
    ("pat-005", "Are ACE inhibitors safe for this patient?"),
    ("pat-006", "Is Emily Nguyen's gestational diabetes under control?"),
    (None,      "Which patients have critically abnormal kidney function?"),
    (None,      "List all patients currently on insulin therapy."),
    (None,      "Which patients have documented allergies to NSAIDs or aspirin?"),
]


def render(question: str, answer: str, patient_id: str | None) -> None:
    has_alert = "⚠️" in answer
    border    = "red" if has_alert else "cyan"
    subtitle  = f"[dim]patient: {patient_id}[/dim]" if patient_id else "[dim]cross-patient[/dim]"
    console.print(Panel(Markdown(answer), title=f"[bold]{question}",
                        subtitle=subtitle, border_style=border, padding=(1, 2)))


def run_demo(rag: RAGPipeline) -> None:
    console.rule("[bold cyan]Patient-Scoped Queries")
    for patient_id, question in DEMO_QUERIES:
        if patient_id:
            render(question, rag.ask(question, patient_id=patient_id), patient_id)
            console.print()

    console.rule("[bold magenta]Cross-Patient Queries")
    for patient_id, question in DEMO_QUERIES:
        if not patient_id:
            render(question, rag.ask(question), None)
            console.print()


def run_interactive(rag: RAGPipeline) -> None:
    console.print(Panel.fit("Type a clinical question. Leave patient blank for all patients.\n"
                            "Type [bold]exit[/bold] to quit.", title="[cyan]Interactive Mode"))
    while True:
        console.print()
        q = Prompt.ask("[bold cyan]Question")
        if q.lower() in ("exit", "quit"):
            break
        pid = Prompt.ask("[dim]Patient ID (blank = all)[/dim]", default="") or None
        render(q, rag.ask(q, patient_id=pid), pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--patient", default=None)
    args = parser.parse_args()

    if not STORE_PATH.exists():
        console.print("[red]Vector store not found. Run: python ingest.py[/red]")
        return

    with console.status("Loading RAG pipeline..."):
        rag = RAGPipeline.load(STORE_PATH)
    console.print("[green]✓[/green] Pipeline ready\n")

    if args.interactive:
        run_interactive(rag)
    elif args.patient:
        for _, question in DEMO_QUERIES:
            render(question, rag.ask(question, patient_id=args.patient), args.patient)
            console.print()
    else:
        run_demo(rag)


if __name__ == "__main__":
    main()
