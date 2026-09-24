"""Carga, armonizacion, auditoria y filtrado de ENEIC con Spark."""

from __future__ import annotations

import math
from functools import reduce
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType

from .config import (
    DATASETS,
    DOMAIN_LABELS,
    EDUCATION_LABELS,
    OCCUPATION_LABELS,
    PARQUET_DIR,
    RAW_DIR,
    SOURCE_COLUMNS,
    DatasetSpec,
)


def _spark_string_schema() -> StructType:
    return StructType([StructField(name, StringType(), True) for name in SOURCE_COLUMNS])


def _nullable_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Excel scalars to nullable strings for an explicit Spark schema."""
    normalized = frame.loc[:, list(SOURCE_COLUMNS)].copy()
    for name in SOURCE_COLUMNS:
        normalized[name] = normalized[name].map(
            lambda value: None if pd.isna(value) else str(value).strip()
        )
    return normalized


def convert_excel_sources(
    spark: SparkSession,
    raw_dir: Path = RAW_DIR,
    parquet_dir: Path = PARQUET_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Read each Excel separately, validate it, and persist selected columns as Parquet."""
    selected_root = parquet_dir / "selected_raw"
    selected_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for spec in DATASETS:
        source = raw_dir / spec.filename
        destination = selected_root / spec.periodo
        if not source.exists():
            raise FileNotFoundError(f"No se encontro la fuente requerida: {source}")

        if destination.exists() and (destination / "_SUCCESS").exists() and not force:
            count = spark.read.parquet(str(destination)).count()
            rows.append(
                {
                    "periodo_archivo": spec.periodo,
                    "archivo_origen": spec.filename,
                    "registros": count,
                    "esperados": spec.expected_rows,
                    "estado": "Parquet reutilizado",
                }
            )
            continue

        excel = pd.ExcelFile(source, engine="openpyxl")
        available_columns = pd.read_excel(
            excel, sheet_name=0, nrows=0, engine="openpyxl"
        ).columns.tolist()
        missing = sorted(set(SOURCE_COLUMNS) - set(available_columns))
        if missing:
            raise ValueError(f"{spec.filename} no contiene columnas requeridas: {missing}")
        if len(available_columns) != spec.expected_columns:
            raise ValueError(
                f"{spec.filename}: {len(available_columns)} columnas; "
                f"se esperaban {spec.expected_columns}."
            )

        pdf = pd.read_excel(
            excel,
            sheet_name=0,
            usecols=list(SOURCE_COLUMNS),
            dtype=object,
            engine="openpyxl",
        )
        if len(pdf) != spec.expected_rows:
            raise ValueError(
                f"{spec.filename}: {len(pdf)} filas; se esperaban {spec.expected_rows}."
            )
        pdf = _nullable_text_frame(pdf)
        sdf = spark.createDataFrame(pdf, schema=_spark_string_schema())
        sdf = (
            sdf.withColumn("archivo_origen", F.lit(spec.filename))
            .withColumn("periodo_archivo", F.lit(spec.periodo))
            .withColumn("anio_archivo", F.lit(spec.year).cast("int"))
            .withColumn("trimestre_calendario", F.lit(spec.quarter).cast("int"))
        )
        sdf.write.mode("overwrite").parquet(str(destination))
        rows.append(
            {
                "periodo_archivo": spec.periodo,
                "archivo_origen": spec.filename,
                "registros": len(pdf),
                "esperados": spec.expected_rows,
                "estado": "Convertido desde Excel",
            }
        )
        del pdf, sdf

    return pd.DataFrame(rows).sort_values("periodo_archivo").reset_index(drop=True)


def load_periods(
    spark: SparkSession,
    periods: Iterable[str],
    parquet_dir: Path = PARQUET_DIR,
) -> DataFrame:
    """Load period partitions and combine them explicitly with unionByName."""
    frames = [
        spark.read.parquet(str(parquet_dir / "selected_raw" / period))
        for period in periods
    ]
    if not frames:
        raise ValueError("Debe proporcionarse al menos un periodo.")
    return reduce(lambda left, right: left.unionByName(right), frames)


def _normalized_code(name: str) -> Column:
    text = F.trim(F.col(name).cast("string"))
    return F.regexp_replace(text, r"\.0+$", "")


def _numeric(name: str) -> Column:
    text = F.regexp_replace(F.trim(F.col(name).cast("string")), ",", "")
    return F.when(text.isin("", "NA", "N/A", "NAN", "NULL"), None).otherwise(
        text.cast("double")
    )


def _label_expression(code: Column, labels: dict[str, str]) -> Column:
    pairs: list[Column] = []
    for key, value in labels.items():
        pairs.extend((F.lit(key), F.lit(value)))
    mapping = F.create_map(*pairs)
    return F.coalesce(mapping[code], F.lit("DESCONOCIDO"))


def harmonize(raw: DataFrame) -> DataFrame:
    """Create analytical fields without altering the original source columns."""
    education_code = _normalized_code("P03A03A")
    occupation_code = _normalized_code("P05C16")
    domain_code = _normalized_code("DOMINIO")

    result = (
        raw.withColumn("edad", _numeric("P02A03"))
        .withColumn("ocupado", _numeric("OCUPADOS"))
        .withColumn("salario_mensual", _numeric("P05D01"))
        .withColumn("antiguedad_anios", _numeric("P05C07A"))
        .withColumn("antiguedad_meses", _numeric("P05C07B"))
        .withColumn("horas_semanales", _numeric("P05H01A"))
        .withColumn("factor", _numeric("FACTOR"))
        .withColumn("nivel_educativo_codigo", education_code)
        .withColumn("categoria_ocupacional_codigo", occupation_code)
        .withColumn("dominio_codigo", domain_code)
        .withColumn(
            "nivel_educativo", _label_expression(education_code, EDUCATION_LABELS)
        )
        .withColumn(
            "categoria_ocupacional",
            _label_expression(occupation_code, OCCUPATION_LABELS),
        )
        .withColumn("dominio", _label_expression(domain_code, DOMAIN_LABELS))
        .withColumn(
            "antiguedad",
            F.col("antiguedad_anios") + F.col("antiguedad_meses") / F.lit(12.0),
        )
    )
    return result


def _finite(column: str) -> Column:
    value = F.col(column)
    return value.isNotNull() & (~F.isnan(value)) & (F.abs(value) < F.lit(float("inf")))


def filter_analytical_population(frame: DataFrame) -> tuple[DataFrame, pd.DataFrame]:
    """Apply the rubric filters in a fixed order and audit every exclusion."""
    conditions = (
        ("Edad finita y >= 15", _finite("edad") & (F.col("edad") >= 15)),
        ("Persona ocupada (OCUPADOS = 1)", F.col("ocupado") == 1),
        (
            "Categoria asalariada (P05C16 en 1,2,3,4)",
            F.col("categoria_ocupacional_codigo").isin("1", "2", "3", "4"),
        ),
        (
            "Salario finito y positivo",
            _finite("salario_mensual") & (F.col("salario_mensual") > 0),
        ),
        (
            "Componentes de antiguedad validos",
            _finite("antiguedad_anios")
            & _finite("antiguedad_meses")
            & (F.col("antiguedad_anios") >= 0)
            & (F.col("antiguedad_meses") >= 0)
            & (F.col("antiguedad_meses") <= 11)
            & (F.col("antiguedad_meses") == F.floor(F.col("antiguedad_meses"))),
        ),
        ("Antiguedad <= edad", F.col("antiguedad") <= F.col("edad")),
        (
            "Horas semanales en (0, 168]",
            _finite("horas_semanales")
            & (F.col("horas_semanales") > 0)
            & (F.col("horas_semanales") <= 168),
        ),
    )

    current = frame
    previous_count = current.count()
    audit = [
        {
            "paso": 0,
            "criterio": "Registros de entrada",
            "antes": previous_count,
            "excluidos": 0,
            "despues": previous_count,
        }
    ]
    for step, (label, condition) in enumerate(conditions, start=1):
        filtered = current.filter(condition)
        after = filtered.count()
        audit.append(
            {
                "paso": step,
                "criterio": label,
                "antes": previous_count,
                "excluidos": previous_count - after,
                "despues": after,
            }
        )
        current = filtered
        previous_count = after

    return current, pd.DataFrame(audit)


def missingness_before_filters(raw: DataFrame) -> pd.DataFrame:
    """Count source missingness in one Spark aggregation."""
    total = raw.count()
    missing_tokens = ("", "NA", "N/A", "NAN", "NULL")
    expressions = []
    for name in SOURCE_COLUMNS:
        text = F.upper(F.trim(F.col(name).cast("string")))
        expressions.append(
            F.sum(F.when(F.col(name).isNull() | text.isin(*missing_tokens), 1).otherwise(0)).alias(name)
        )
    counts = raw.agg(*expressions).first().asDict()
    return pd.DataFrame(
        [
            {
                "variable": name,
                "faltantes": int(counts[name]),
                "porcentaje": 100.0 * int(counts[name]) / total if total else math.nan,
            }
            for name in SOURCE_COLUMNS
        ]
    ).sort_values(["porcentaje", "variable"], ascending=[False, True])


def counts_by_file(before: DataFrame, after: DataFrame) -> pd.DataFrame:
    before_counts = before.groupBy("periodo_archivo").count().withColumnRenamed(
        "count", "antes"
    )
    after_counts = after.groupBy("periodo_archivo").count().withColumnRenamed(
        "count", "despues"
    )
    return (
        before_counts.join(after_counts, "periodo_archivo", "left")
        .fillna(0, subset=["despues"])
        .withColumn("excluidos", F.col("antes") - F.col("despues"))
        .withColumn("retencion_pct", 100.0 * F.col("despues") / F.col("antes"))
        .orderBy("periodo_archivo")
        .toPandas()
    )


def duplicate_key_audit(frame: DataFrame) -> pd.DataFrame:
    """Investigate duplicate period-household-person keys without dropping rows."""
    keys = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]
    payload = [name for name in frame.columns if name not in keys]
    fingerprint = F.sha2(
        F.concat_ws(
            "||",
            *[F.coalesce(F.col(name).cast("string"), F.lit("<NULL>")) for name in payload],
        ),
        256,
    )
    groups = (
        frame.withColumn("_fingerprint", fingerprint)
        .groupBy(*keys)
        .agg(
            F.count("*").alias("filas"),
            F.countDistinct("_fingerprint").alias("versiones_distintas"),
        )
        .filter(F.col("filas") > 1)
        .cache()
    )
    summary = groups.agg(
        F.count("*").alias("grupos_duplicados"),
        F.coalesce(F.sum("filas"), F.lit(0)).alias("filas_involucradas"),
        F.coalesce(
            F.sum(F.when(F.col("versiones_distintas") == 1, 1).otherwise(0)), F.lit(0)
        ).alias("grupos_repeticion_exacta"),
        F.coalesce(
            F.sum(F.when(F.col("versiones_distintas") > 1, 1).otherwise(0)), F.lit(0)
        ).alias("grupos_en_conflicto"),
    ).toPandas()
    groups.unpersist()
    return summary


def observed_source_quarters(frame: DataFrame) -> pd.DataFrame:
    return (
        frame.select("periodo_archivo", _normalized_code("TRIMESTRE").alias("TRIMESTRE"))
        .groupBy("periodo_archivo", "TRIMESTRE")
        .count()
        .orderBy("periodo_archivo", "TRIMESTRE")
        .toPandas()
    )


def save_prepared_sets(
    prepared_2025: DataFrame,
    prepared_2026: DataFrame,
    parquet_dir: Path = PARQUET_DIR,
) -> None:
    prepared_2025.write.mode("overwrite").parquet(str(parquet_dir / "prepared_2025"))
    prepared_2026.write.mode("overwrite").parquet(str(parquet_dir / "prepared_2026"))
