from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Allergy(BaseModel):
    substance: str
    reaction: str
    severity: Literal["mild", "moderate", "severe"]


class Condition(BaseModel):
    name: str
    icd_code: str
    onset_date: str
    status: Literal["active", "chronic", "resolved"]


class Medication(BaseModel):
    name: str
    dose: str
    frequency: str
    route: str
    indication: str
    start_date: str


class VitalSigns(BaseModel):
    date: str
    blood_pressure: str
    heart_rate: int
    temperature_c: float
    weight_kg: float
    spo2_pct: int


class Patient(BaseModel):
    id: str
    mrn: str
    name: str
    dob: str
    gender: Literal["male", "female", "other"]
    blood_type: str
    allergies: list[Allergy] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    vitals: list[VitalSigns] = Field(default_factory=list)
    primary_physician: str
    insurance: str


class LabValue(BaseModel):
    test: str
    value: float | str
    unit: str
    reference_range: str
    flag: Literal["H", "L", "HH", "LL", ""] = ""


class LabResult(BaseModel):
    id: str
    patient_id: str
    date: str
    panel: str
    ordering_physician: str
    results: list[LabValue]


class ClinicalNote(BaseModel):
    id: str
    patient_id: str
    date: str
    author: str
    specialty: str
    note_type: str
    content: str
