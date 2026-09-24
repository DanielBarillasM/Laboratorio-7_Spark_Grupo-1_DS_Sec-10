"""Configuracion central del laboratorio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    periodo: str
    year: int
    quarter: int
    expected_rows: int
    expected_columns: int


DATASETS = (
    DatasetSpec("Personas_ENEIC_I_2025.xlsx", "2025T1", 2025, 1, 51_588, 270),
    DatasetSpec("Personas_ENEIC_II_2025.xlsx", "2025T2", 2025, 2, 51_167, 270),
    DatasetSpec("Personas_ENEIC_III_2025.xlsx", "2025T3", 2025, 3, 51_583, 270),
    DatasetSpec("Personas_ENEIC_IV_2025.xlsx", "2025T4", 2025, 4, 49_338, 302),
    DatasetSpec("Personas_ENEIC_I_2026.xlsx", "2026T1", 2026, 1, 49_843, 270),
)

SOURCE_COLUMNS = (
    "NUM_HOGAR",
    "NUM_PERSONA",
    "FACTOR",
    "ANIO",
    "TRIMESTRE",
    "P02A03",
    "OCUPADOS",
    "P05C16",
    "P05D01",
    "P05C07A",
    "P05C07B",
    "P05H01A",
    "P03A03A",
    "DOMINIO",
)

NUMERIC_PREDICTORS = ("edad", "antiguedad", "horas_semanales")
CATEGORICAL_PREDICTORS = (
    "nivel_educativo",
    "categoria_ocupacional",
    "dominio",
)
SUPERVISED_PREDICTORS = NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS
LABEL = "salario_mensual"

EDUCATION_LABELS = {
    "0": "Ninguno",
    "1": "Preprimaria",
    "2": "Primaria",
    "3": "Basico",
    "4": "Diversificado",
    "5": "Superior",
    "6": "Maestria",
    "7": "Doctorado",
}

OCCUPATION_LABELS = {
    "1": "Empleado de gobierno",
    "2": "Empleado de empresa privada",
    "3": "Jornalero o peon",
    "4": "Servicio domestico",
}

DOMAIN_LABELS = {
    "1": "Urbano Metropolitano",
    "2": "Resto Urbano",
    "3": "Rural Nacional",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("ENEIC_RAW_DIR", "/opt/app/working_dir/eneic/raw"))
PARQUET_DIR = Path(
    os.environ.get("ENEIC_PARQUET_DIR", "/opt/app/working_dir/eneic/parquet")
)
MODEL_DIR = Path(os.environ.get("ENEIC_MODEL_DIR", "/opt/app/working_dir/eneic/models"))
OUTPUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_directories() -> None:
    """Create local output and external working directories."""
    for path in (PARQUET_DIR, MODEL_DIR, TABLES_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
