"""Pruebas ligeras del contrato exigido por la rubrica."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab7.config import (  # noqa: E402
    CATEGORICAL_PREDICTORS,
    DATASETS,
    NUMERIC_PREDICTORS,
    SOURCE_COLUMNS,
)


def test_periods_and_expected_source_sizes():
    assert [item.periodo for item in DATASETS] == [
        "2025T1",
        "2025T2",
        "2025T3",
        "2025T4",
        "2026T1",
    ]
    assert [item.expected_rows for item in DATASETS] == [51588, 51167, 51583, 49338, 49843]
    assert [item.expected_columns for item in DATASETS] == [270, 270, 270, 302, 270]


def test_exactly_six_supervised_predictors():
    assert NUMERIC_PREDICTORS == ("edad", "antiguedad", "horas_semanales")
    assert CATEGORICAL_PREDICTORS == (
        "nivel_educativo",
        "categoria_ocupacional",
        "dominio",
    )
    assert len(NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS) == 6


def test_required_audit_columns_are_loaded():
    required = {"NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE"}
    assert required.issubset(SOURCE_COLUMNS)


def test_models_do_not_import_scikit_learn():
    source = (ROOT / "src" / "lab7" / "modeling.py").read_text(encoding="utf-8")
    assert "sklearn" not in source.lower()
    assert "pyspark.ml" in source
