"""Pruebas ligeras del contrato exigido por la rubrica."""

from pathlib import Path
import csv
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab7.config import (  # noqa: E402
    CATEGORICAL_PREDICTORS,
    DATASETS,
    FINAL_FOREST_MODEL,
    FINAL_LINEAR_MODEL,
    FINAL_TRAIN_PERIODS,
    LABEL,
    MODEL_DIR,
    NUMERIC_PREDICTORS,
    SOURCE_COLUMNS,
    SUPERVISED_PREDICTORS,
    TABLES_DIR,
    TEST_PERIODS,
    TRAIN_PERIODS,
    VALIDATION_PERIODS,
)

FORBIDDEN_PREDICTORS = {
    LABEL, "P05D01", "factor", "FACTOR", "cluster", "NUM_HOGAR", "NUM_PERSONA",
    "periodo_archivo", "archivo_origen", "ANIO", "TRIMESTRE", "YLAB_PUBLI",
    "salario_por_hora", "prediction",
}


def _read_table(name: str) -> list[dict[str, str]]:
    path = TABLES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} aun no se genera; ejecute el notebook completo.")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_temporal_partitions_are_disjoint_and_ordered():
    assert TRAIN_PERIODS == ("2025T1", "2025T2", "2025T3")
    assert VALIDATION_PERIODS == ("2025T4",)
    assert FINAL_TRAIN_PERIODS == TRAIN_PERIODS + VALIDATION_PERIODS
    assert TEST_PERIODS == ("2026T1",)
    assert not set(FINAL_TRAIN_PERIODS) & set(TEST_PERIODS)
    assert not set(TRAIN_PERIODS) & set(VALIDATION_PERIODS)


def test_no_target_leakage_in_predictors():
    assert len(SUPERVISED_PREDICTORS) == 6
    assert not FORBIDDEN_PREDICTORS & set(SUPERVISED_PREDICTORS)


def test_pipeline_assembler_only_uses_the_six_predictors():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    from lab7.modeling import _preprocessing_stages

    SparkSession.builder.master("local[1]").appName("lab7-tests").getOrCreate()
    assembler = _preprocessing_stages()[-1]
    inputs = assembler.getInputCols()
    assert inputs == list(NUMERIC_PREDICTORS) + [f"{c}_ohe" for c in CATEGORICAL_PREDICTORS]
    assert not FORBIDDEN_PREDICTORS & set(inputs)


def test_final_evaluation_uses_2026_only_for_scoring():
    source = (ROOT / "scripts" / "build_notebook.py").read_text(encoding="utf-8")
    section_7 = source[source.index("## 7. Entrenamiento final"):]
    assert "fit_linear_candidates" not in section_7
    assert "fit_random_forest_candidates" not in section_7
    modeling = (ROOT / "src" / "lab7" / "modeling.py").read_text(encoding="utf-8")
    assert ".fit(test" not in modeling


def test_both_models_scored_on_same_test_records():
    rows = _read_table("conteo_test_compartido.csv")
    counts = {int(row["registros"]) for row in rows}
    assert len(counts) == 1 and counts.pop() > 0

    metrics = _read_table("metricas_test_2026.csv")
    assert {"modelo", "registros", "MAE", "RMSE", "R2"} <= set(metrics[0])
    assert len({row["registros"] for row in metrics}) == 1


def test_validation_winner_is_the_model_evaluated_in_2026():
    linear = _read_table("metricas_regresion_lineal.csv")
    forest = _read_table("metricas_random_forest.csv")
    best_linear = min(linear, key=lambda row: float(row["RMSE"]))["nombre"]
    best_forest = min(forest, key=lambda row: float(row["RMSE"]))["nombre"]
    evaluated = {row["modelo"] for row in _read_table("metricas_test_2026.csv")}
    assert {best_linear, best_forest} <= evaluated


def test_final_models_were_saved():
    if not MODEL_DIR.exists():
        pytest.skip(f"Directorio externo de modelos no disponible: {MODEL_DIR}")
    for name in (FINAL_LINEAR_MODEL, FINAL_FOREST_MODEL):
        assert (MODEL_DIR / name / "metadata").exists(), name
        assert (MODEL_DIR / name / "stages").exists(), name
