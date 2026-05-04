"""Generates clinician-facing answers using GPT-4 grounded in retrieved records."""

from __future__ import annotations

from medrag.config import Settings

# Clinical system prompt — defines response format and safety rules
SYSTEM_PROMPT = """You are ClinicalRAG, a clinical decision support assistant for licensed physicians.

RULES:
1. Answer ONLY from the patient records provided. Never fabricate clinical data.
2. Cite the source type after each claim using brackets:
   [Demographics] [Conditions] [Medication: <drug name>] [Lab: <panel> <date>] [Note: <date> <author>]
3. Flag critical findings with: ⚠️ ALERT:
4. If information is absent, state: "Not documented in available records."
5. This is a decision support tool. All clinical decisions remain with the treating physician.

RESPONSE FORMAT:
**Summary:** Direct 1–3 sentence answer to the physician's question.

**Supporting Evidence:**
• Bullet list with inline source citations

**⚠️ Clinical Alerts:** (omit this entire section if there are no alerts)
• Drug interactions, critical lab values, contraindications, allergy conflicts"""


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk.get("metadata", chunk)
        ctype = meta.get("chunk_type", "record")
        date  = meta.get("date", "")
        text  = meta.get("text", "")
        header = f"[{i}] {ctype.upper()}" + (f" ({date})" if date else "")
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def generate(query: str, chunks: list[dict]) -> str:
    if not Settings.OPENAI_API_KEY:
        return _mock_response(query, chunks)

    import openai
    client = openai.OpenAI(api_key=Settings.OPENAI_API_KEY)

    context = format_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Patient Records:\n\n{context}\n\n"
                f"Physician Question: {query}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=Settings.OPENAI_MODEL,
        messages=messages,
        temperature=0.1,  # low temp for clinical accuracy
        max_tokens=800,
    )
    return response.choices[0].message.content or ""


def _mock_response(query: str, chunks: list[dict]) -> str:
    """Fallback when OPENAI_API_KEY is not set — returns a structured placeholder."""
    source_list = "\n".join(
        f"• [{c.get('metadata', c).get('chunk_type', 'record').upper()}] "
        f"{c.get('metadata', c).get('text', '')[:120]}..."
        for c in chunks[:3]
    )
    return (
        f"**Summary:** [OPENAI_API_KEY not configured — set it in .env to enable generation]\n\n"
        f"**Supporting Evidence (retrieved chunks):**\n{source_list}\n\n"
        f"**⚠️ Clinical Alerts:** Set OPENAI_API_KEY to receive AI-generated clinical alerts."
    )
