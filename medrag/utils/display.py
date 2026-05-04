"""Rich terminal rendering for RAG responses and source citations."""

from __future__ import annotations
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


def render_rag_response(resp) -> None:
    """Renders a full RAGResponse to the terminal."""
    border = "red" if resp.has_alerts else "cyan"

    subtitle = (
        f"[dim]{resp.source_count} sources retrieved  ·  "
        f"{resp.latency_ms} ms"
        + (f"  ·  {resp.model}" if resp.model else "")
        + (f"  ·  patient {resp.patient_id}" if resp.patient_id else "")
        + "[/dim]"
    )

    console.print(
        Panel(
            Markdown(resp.answer),
            title=f"[bold]{resp.question}",
            subtitle=subtitle,
            border_style=border,
            padding=(1, 2),
        )
    )


def render_sources_table(sources: list[dict]) -> None:
    """Renders retrieved source chunks as a compact table."""
    if not sources:
        return

    table = Table(
        title="Retrieved Context",
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("#",          style="dim",    width=3)
    table.add_column("Type",       style="yellow", width=14)
    table.add_column("Score",      style="green",  width=7)
    table.add_column("Patient",    style="cyan",   width=20)
    table.add_column("Preview",    style="white")

    for i, chunk in enumerate(sources, 1):
        meta    = chunk.get("metadata", chunk)
        ctype   = meta.get("chunk_type", "—")
        score   = f"{chunk.get('score', 0):.3f}"
        patient = meta.get("patient_name") or meta.get("patient_id", "—")
        text    = meta.get("text", "")[:90] + ("…" if len(meta.get("text", "")) > 90 else "")
        table.add_row(str(i), ctype, score, patient, text)

    console.print(table)
