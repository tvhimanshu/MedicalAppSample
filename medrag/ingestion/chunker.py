"""
Converts structured patient records into flat text chunks suitable for
vector embedding. One chunk = one semantically coherent unit.

Routing:
  Patient demographics / conditions / medications / vitals → datapoint: patient_records
  LabResult                                                → datapoint: lab_results
  ClinicalNote                                             → datapoint: clinical_notes
"""

from __future__ import annotations
from datetime import date

from medrag.models.patient import Patient, LabResult, ClinicalNote


def _age(dob: str) -> int:
    born  = date.fromisoformat(dob)
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def chunk_patient(patient: Patient) -> list[dict]:
    chunks: list[dict] = []
    age   = _age(patient.dob)

    # ── Demographics + Allergies ──────────────────────────────────────────────
    allergy_str = "; ".join(
        f"{a.substance} ({a.reaction}, {a.severity})" for a in patient.allergies
    ) or "NKDA — No Known Drug Allergies"

    chunks.append({
        "id":           f"{patient.id}_demographics",
        "patient_id":   patient.id,
        "patient_name": patient.name,
        "chunk_type":   "demographics",
        "text": (
            f"Patient Demographics — {patient.name} | MRN: {patient.mrn} | "
            f"Age {age} | {patient.gender.capitalize()} | DOB: {patient.dob} | "
            f"Blood type: {patient.blood_type} | Allergies: {allergy_str} | "
            f"Primary physician: {patient.primary_physician} | "
            f"Insurance: {patient.insurance}."
        ),
    })

    # ── Active Conditions ─────────────────────────────────────────────────────
    active = [c for c in patient.conditions if c.status in ("active", "chronic")]
    if active:
        cond_str = "; ".join(
            f"{c.name} (ICD-10: {c.icd_code}, onset {c.onset_date}, {c.status})"
            for c in active
        )
        chunks.append({
            "id":           f"{patient.id}_conditions",
            "patient_id":   patient.id,
            "patient_name": patient.name,
            "chunk_type":   "conditions",
            "text": (
                f"Active Diagnoses — {patient.name} (MRN: {patient.mrn}): {cond_str}."
            ),
        })

    # ── Medications (one chunk each for granular retrieval) ───────────────────
    for med in patient.medications:
        safe_name = med.name.lower().replace(" ", "_").replace("/", "_")
        chunks.append({
            "id":              f"{patient.id}_med_{safe_name}",
            "patient_id":      patient.id,
            "patient_name":    patient.name,
            "chunk_type":      "medication",
            "medication_name": med.name,
            "text": (
                f"Medication — {patient.name} (MRN: {patient.mrn}) is on "
                f"{med.name} {med.dose} {med.frequency} {med.route} "
                f"for {med.indication}. Prescribed {med.start_date}."
            ),
        })

    # ── Latest Vital Signs ────────────────────────────────────────────────────
    if patient.vitals:
        v = sorted(patient.vitals, key=lambda x: x.date, reverse=True)[0]
        chunks.append({
            "id":           f"{patient.id}_vitals_{v.date}",
            "patient_id":   patient.id,
            "patient_name": patient.name,
            "chunk_type":   "vitals",
            "date":         v.date,
            "text": (
                f"Vital Signs — {patient.name} (MRN: {patient.mrn}) on {v.date}: "
                f"BP {v.blood_pressure} mmHg | HR {v.heart_rate} bpm | "
                f"Temp {v.temperature_c}°C | Weight {v.weight_kg} kg | "
                f"SpO₂ {v.spo2_pct}%."
            ),
        })

    return chunks


def chunk_lab_result(lab: LabResult) -> dict:
    abnormal = [r for r in lab.results if r.flag]
    normal   = [r for r in lab.results if not r.flag]

    ab_parts = [
        f"{r.test}: {r.value} {r.unit} [FLAG: {r.flag}] (ref {r.reference_range})"
        for r in abnormal
    ]
    norm_parts = [
        f"{r.test}: {r.value} {r.unit} (ref {r.reference_range})"
        for r in normal
    ]

    lines = [f"Lab Results — {lab.panel} for {lab.patient_id} on {lab.date}."]
    if ab_parts:
        lines.append("ABNORMAL: " + "; ".join(ab_parts) + ".")
    if norm_parts:
        lines.append("Normal: " + "; ".join(norm_parts) + ".")
    lines.append(f"Ordered by {lab.ordering_physician}.")

    return {
        "id":           lab.id,
        "patient_id":   lab.patient_id,
        "chunk_type":   "lab_result",
        "date":         lab.date,
        "panel":        lab.panel,
        "text":         " ".join(lines),
    }


def chunk_clinical_note(note: ClinicalNote) -> dict:
    return {
        "id":         note.id,
        "patient_id": note.patient_id,
        "chunk_type": "clinical_note",
        "date":       note.date,
        "author":     note.author,
        "note_type":  note.note_type,
        "text": (
            f"Clinical Note [{note.note_type}] — Patient {note.patient_id} | "
            f"{note.date} | {note.author} ({note.specialty}): {note.content}"
        ),
    }
