import openai
from medrag.config import OPENAI_API_KEY, OPENAI_MODEL

_client = openai.OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are ClinicalRAG, a decision support assistant for licensed physicians.

Rules:
- Answer ONLY from the patient records provided in the context.
- Cite your source after each claim: [Demographics] [Medication] [Lab DATE] [Note DATE]
- Flag critical findings with: ⚠️ ALERT:
- If information is absent, say: "Not documented in available records."
- Never suggest diagnoses. This tool supports — it does not replace — clinical judgment.

Response format:
**Summary:** 1–3 sentence direct answer.
**Evidence:** Bullet points with inline citations.
**⚠️ Alerts:** Drug interactions, critical values, contraindications. Omit if none."""


def generate(question: str, context_docs: list[tuple]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {doc.metadata.get('doc_type','record').upper()} — "
        f"{doc.metadata.get('patient_name', doc.metadata.get('patient_id', ''))}\n{doc.text}"
        for i, (doc, _) in enumerate(context_docs)
    )

    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Patient Records:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content or ""
