# MedicalRAG — Clinical Decision Support via RAG

A production-style RAG system that lets physicians query patient records using
natural language. Built on **VectorSearchCore** (vector store + semantic search)
and **OpenAI GPT-4** for grounded, citation-backed clinical responses.

> **All patient data is fully synthetic and for demonstration purposes only.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Doctor Query                             │
│         "What is Robert Chen's HbA1c and is it at target?"      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   Retriever    │  semantic search via VectorSearchCore
                    │  (top-K=6)     │  → patient_records + lab_results
                    └───────┬────────┘       + clinical_notes datapoints
                            │ retrieved chunks (with scores)
                    ┌───────▼────────┐
                    │   Generator    │  OpenAI GPT-4o-mini
                    │  (temp=0.1)    │  system prompt enforces citation format
                    └───────┬────────┘       + clinical alert rules
                            │
                    ┌───────▼────────────────────────────────────────┐
                    │  **Summary:** HbA1c 7.8% [Lab: HbA1c 2024-10]  │
                    │  **Evidence:** ...with citations               │
                    │  ⚠️ ALERT: Above target of <7.0% for DM        │
                    └────────────────────────────────────────────────┘
```

### Data chunking strategy

Each patient record is split into typed semantic chunks before embedding:

| Chunk type      | Datapoint          | Example                                  |
|-----------------|--------------------|------------------------------------------|
| `demographics`  | patient_records    | Name, DOB, blood type, allergies         |
| `conditions`    | patient_records    | Active diagnoses with ICD-10 codes       |
| `medication`    | patient_records    | One chunk **per drug** for granular retrieval |
| `vitals`        | patient_records    | Latest vital signs                       |
| `lab_result`    | lab_results        | One chunk per panel with abnormal flags  |
| `clinical_note` | clinical_notes     | Full physician note text                 |

---

## Synthetic Patient Dataset

| Patient         | Age | Key Conditions                                 |
|-----------------|-----|------------------------------------------------|
| Robert Chen     | 65  | T2DM, Hypertension, CKD Stage 2                |
| Maria Santos    | 42  | Hypothyroidism (over-replaced), Anxiety, GERD  |
| James Wilson    | 58  | Paroxysmal AFib, Post-hip arthroplasty         |
| Aisha Patel     | 35  | Persistent Asthma, Aspirin-exacerbated AERD    |
| Harold Thompson | 71  | COPD GOLD III, CHF (EF 35%), T2DM, CKD Stage 3 |
| Emily Nguyen    | 29  | Gestational Diabetes, Iron-deficiency Anemia   |

---

## Setup

```bash
cd MedicalRAG
python -m venv venv && source venv/bin/activate
pip install -e .

cp .env.example .env
# Fill in OPENAI_API_KEY and VECTOR_SEARCH_URL
```

**Prerequisites:**
- VectorSearchCore backend running (`npm run dev` in the VectorSearchCore directory)
- OpenAI API key (for embeddings + generation)

---

## Run

```bash
# 1. Create VectorSearchCore project, datapoints, search URL, API key
python scripts/01_setup_indexes.py

# 2. Chunk and ingest all patient data
python scripts/02_ingest_patients.py

# 3a. Run full demo with preset physician queries
python scripts/03_doctor_query_demo.py

# 3b. Show retrieved source chunks alongside answers
python scripts/03_doctor_query_demo.py --sources

# 3c. Interactive free-text query mode
python scripts/03_doctor_query_demo.py --interactive

# 3d. Scope queries to a single patient
python scripts/03_doctor_query_demo.py --patient pat-005
```

---

## Project Structure

```
MedicalRAG/
├── medrag/
│   ├── config.py                 Settings (env-driven)
│   ├── models/
│   │   └── patient.py            Pydantic models: Patient, LabResult, ClinicalNote
│   ├── ingestion/
│   │   ├── vector_client.py      VectorSearchCore API client (auth, ingest, search)
│   │   ├── chunker.py            Splits records into typed semantic text chunks
│   │   └── pipeline.py           Orchestrates setup + ingestion
│   ├── rag/
│   │   ├── retriever.py          Semantic search query via VectorSearchCore
│   │   ├── generator.py          GPT-4 generation with clinical system prompt
│   │   └── pipeline.py           RAGPipeline.ask() — retrieve → generate → respond
│   └── utils/
│       └── display.py            Rich terminal rendering helpers
├── data/
│   ├── patients.json             6 synthetic patients with full medical history
│   ├── lab_results.json          Recent lab panels with abnormal flags
│   └── clinical_notes.json       Physician notes with treatment decisions
└── scripts/
    ├── 01_setup_indexes.py
    ├── 02_ingest_patients.py
    └── 03_doctor_query_demo.py
```

---

## Sample Demo Queries

**Patient-scoped (doctor asks about a specific patient):**
- *"Does this patient have any drug allergies I should know about before prescribing?"*
- *"What is James Wilson's current INR and is it at therapeutic range?"*
- *"Why was furosemide dose increased and what monitoring is required?"*
- *"Is Emily's gestational diabetes under control based on latest records?"*

**Cross-patient cohort:**
- *"Which patients have abnormal kidney function?"*
- *"List all patients on insulin and their diagnoses."*
- *"Which patients have a documented allergy to NSAIDs or aspirin?"*
- *"Are there any critically elevated BNP values suggesting heart failure?"*
