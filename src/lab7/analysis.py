"""Estadistica descriptiva, correlaciones y segmentacion con Spark MLlib."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, functions as F

from .config import (
    EDUCATION_LABELS,
    LABEL,
    MODEL_DIR,
    PARQUET_DIR,
    RECORD_KEYS,
    SEED,
    TEST_PREDICTIONS,
)


DESCRIPTIVE_COLUMNS = (
    "salario_mensual",
    "edad",
    "antiguedad",
    "horas_semanales",
)


def descriptive_statistics(frame: DataFrame) -> pd.DataFrame:
    """Compute full-data summaries with Spark; no plotting sample is used."""
    rows: list[dict[str, float | int | str]] = []
    for name in DESCRIPTIVE_COLUMNS:
        aggregate = frame.agg(
            F.count(F.col(name)).alias("n"),
            F.avg(name).alias("media"),
            F.stddev(name).alias("desviacion_estandar"),
            F.min(name).alias("minimo"),
            F.max(name).alias("maximo"),
        ).first()
        q25, median, q75, q95 = frame.approxQuantile(
            name, [0.25, 0.50, 0.75, 0.95], 0.001
        )
        rows.append(
            {
                "variable": name,
                "n": int(aggregate["n"]),
                "media": float(aggregate["media"]),
                "mediana": float(median),
                "desviacion_estandar": float(aggregate["desviacion_estandar"]),
                "minimo": float(aggregate["minimo"]),
                "p25": float(q25),
                "p75": float(q75),
                "p95": float(q95),
                "maximo": float(aggregate["maximo"]),
            }
        )
    return pd.DataFrame(rows)


def categorical_distribution(frame: DataFrame, column: str) -> pd.DataFrame:
    total = frame.count()
    return (
        frame.groupBy(column)
        .count()
        .withColumn("porcentaje", 100.0 * F.col("count") / F.lit(total))
        .orderBy(F.desc("count"))
        .toPandas()
        .rename(columns={column: "categoria", "count": "n"})
        .assign(variable=column)
        [["variable", "categoria", "n", "porcentaje"]]
    )


def median_salary_by(frame: DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupBy(column)
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias(
                "salario_mediano"
            ),
            F.avg("salario_mensual").alias("salario_promedio"),
        )
        .orderBy(F.desc("salario_mediano"))
        .toPandas()
        .rename(columns={column: "categoria"})
        .assign(variable=column)
        [["variable", "categoria", "n", "salario_mediano", "salario_promedio"]]
    )


def quarterly_summary(frame: DataFrame) -> pd.DataFrame:
    return (
        frame.groupBy("periodo_archivo")
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(salario_mensual, 0.5, 10000)").alias(
                "salario_mediano"
            ),
            F.avg("salario_mensual").alias("salario_promedio"),
        )
        .orderBy("periodo_archivo")
        .toPandas()
    )


def salary_plot_sample(frame: DataFrame, limit: int = 5_000) -> pd.DataFrame:
    """Return a deterministic plotting-only sample within the rubric limit."""
    return (
        frame.select("salario_mensual")
        .orderBy(F.rand(SEED))
        .limit(limit)
        .toPandas()
    )


def pearson_correlation(frame: DataFrame) -> pd.DataFrame:
    names = ["salario_mensual", "edad", "antiguedad", "horas_semanales"]
    assembled = VectorAssembler(inputCols=names, outputCol="corr_features").transform(
        frame
    )
    matrix = Correlation.corr(assembled, "corr_features", "pearson").head()[0]
    return pd.DataFrame(matrix.toArray(), index=names, columns=names)


def kmeans_search(
    frame: DataFrame,
    model_dir: Path = MODEL_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, PipelineModel, str, int]:
    """Compare K=2..5 with and without salary and retain the best silhouette."""
    scenarios = {
        "Perfil laboral (sin salario)": ["edad", "antiguedad", "horas_semanales"],
        "Perfil economico (con salario)": [
            "edad",
            "antiguedad",
            "horas_semanales",
            "salario_mensual",
        ],
    }
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    candidates: list[dict[str, object]] = []
    fitted: dict[tuple[str, int], PipelineModel] = {}

    for scenario, features in scenarios.items():
        for k in range(2, 6):
            pipeline = Pipeline(
                stages=[
                    VectorAssembler(inputCols=features, outputCol="raw_features"),
                    StandardScaler(
                        inputCol="raw_features",
                        outputCol="scaled_features",
                        withMean=True,
                        withStd=True,
                    ),
                    KMeans(
                        featuresCol="scaled_features",
                        predictionCol="cluster",
                        k=k,
                        seed=SEED,
                        maxIter=50,
                    ),
                ]
            )
            model = pipeline.fit(frame)
            predictions = model.transform(frame)
            silhouette = float(evaluator.evaluate(predictions))
            candidates.append(
                {"escenario": scenario, "k": k, "silhouette": silhouette}
            )
            fitted[(scenario, k)] = model

    results = pd.DataFrame(candidates).sort_values(
        ["silhouette", "escenario"], ascending=[False, True]
    )
    best = results.iloc[0]
    scenario = str(best["escenario"])
    k = int(best["k"])
    features = scenarios[scenario]
    best_model = fitted[(scenario, k)]
    predictions = best_model.transform(frame).cache()

    aggregations = [F.count("*").alias("n")]
    for name in features:
        aggregations.extend(
            [
                F.avg(name).alias(f"{name}_media"),
                F.expr(f"percentile_approx({name}, 0.5, 10000)").alias(
                    f"{name}_mediana"
                ),
            ]
        )
    profiles = predictions.groupBy("cluster").agg(*aggregations).orderBy("cluster").toPandas()
    total = profiles["n"].sum()
    profiles["porcentaje"] = 100.0 * profiles["n"] / total

    global_means = {name: float(frame.agg(F.avg(name)).first()[0]) for name in features}

    def describe(row: pd.Series) -> str:
        labels = []
        for name in features:
            value = float(row[f"{name}_media"])
            relation = "alto" if value >= global_means[name] else "bajo"
            labels.append(f"{name.replace('_', ' ')} {relation}")
        return ", ".join(labels).capitalize()

    profiles["descripcion"] = profiles.apply(describe, axis=1)
    destination = model_dir / "kmeans_best"
    best_model.write().overwrite().save(str(destination))
    predictions.unpersist()
    return results.reset_index(drop=True), profiles, best_model, scenario, k


# ---------------------------------------------------------------------------
# Actividad 8: analisis de errores sobre las predicciones de 2026T1
# ---------------------------------------------------------------------------

# Modelo -> columna de prediccion en el Parquet guardado por la actividad 7.
MODEL_COLUMNS = {
    "Regresión lineal": "prediccion_lr",
    "Random Forest": "prediccion_rf",
}
EDUCATION_ORDER = [EDUCATION_LABELS[key] for key in sorted(EDUCATION_LABELS)] + [
    "DESCONOCIDO"
]
PERCENTILE_CUTS = (0.25, 0.50, 0.75, 0.90, 0.95)
PLOT_SAMPLE_SIZE = 5_000


def load_test_predictions(spark, path: Path | None = None) -> DataFrame:
    """Read the 2026T1 predictions written by activity 7 (one row per test record).

    Residual convention (identical for every table and plot):
    ``residuo = salario_real - salario_predicho``. Positive means the model
    underestimated the salary; negative means it overestimated it.
    """
    source = path or (PARQUET_DIR / TEST_PREDICTIONS)
    frame = spark.read.parquet(str(source))
    needed = {LABEL, "nivel_educativo", "dominio", *RECORD_KEYS, *MODEL_COLUMNS.values()}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en las predicciones: {sorted(missing)}")
    return frame


def _long_errors(predictions: DataFrame) -> DataFrame:
    """Stack both models into one long frame: modelo, prediccion, residuo, error_abs."""
    pieces = []
    for model, column in MODEL_COLUMNS.items():
        pieces.append(
            predictions.select(
                *RECORD_KEYS,
                LABEL,
                "nivel_educativo",
                "dominio",
                F.lit(model).alias("modelo"),
                F.col(column).alias("prediccion"),
            )
        )
    stacked = pieces[0]
    for piece in pieces[1:]:
        stacked = stacked.unionByName(piece)
    return stacked.withColumn("residuo", F.col(LABEL) - F.col("prediccion")).withColumn(
        "error_abs", F.abs(F.col("residuo"))
    )


def plot_comparison_sample(
    predictions: DataFrame, limit: int = PLOT_SAMPLE_SIZE
) -> pd.DataFrame:
    """One reproducible sample (<= 5,000 rows) shared by both models for plotting only.

    Rows are ordered by a hash of the record key salted with the global seed, so
    the sample does not depend on Spark partitioning and is identical between runs.
    """
    if limit > PLOT_SAMPLE_SIZE:
        raise ValueError("La muestra grafica no debe superar 5,000 registros.")
    columns = [
        *RECORD_KEYS,
        LABEL,
        "nivel_educativo",
        "dominio",
        *MODEL_COLUMNS.values(),
    ]
    sample = (
        predictions.select(*columns)
        .withColumn("_orden", F.xxhash64(*[F.col(k) for k in RECORD_KEYS], F.lit(SEED)))
        .orderBy("_orden")
        .limit(limit)
        .drop("_orden")
        .toPandas()
    )
    for model, column in MODEL_COLUMNS.items():
        sample[f"residuo_{column}"] = sample[LABEL] - sample[column]
    return sample


def error_by_group(predictions: DataFrame, column: str) -> pd.DataFrame:
    """MAE and mean error per group and model using every test record (no sampling)."""
    table = (
        _long_errors(predictions)
        .groupBy("modelo", column)
        .agg(
            F.count("*").alias("n"),
            F.avg("error_abs").alias("MAE"),
            F.avg("residuo").alias("error_medio"),
            F.avg(LABEL).alias("salario_promedio"),
        )
        .toPandas()
        .rename(columns={column: "grupo"})
    )
    table["variable"] = column
    if column == "nivel_educativo":
        order = {name: i for i, name in enumerate(EDUCATION_ORDER)}
        table["_o"] = table["grupo"].map(order).fillna(len(order))
        table = table.sort_values(["modelo", "_o"]).drop(columns="_o")
    else:
        table = table.sort_values(["modelo", "grupo"])
    table["tendencia"] = table["error_medio"].map(
        lambda value: "subestima" if value > 0 else "sobreestima"
    )
    return table[
        ["modelo", "variable", "grupo", "n", "MAE", "error_medio", "salario_promedio", "tendencia"]
    ].reset_index(drop=True)


def salary_percentile_bands(predictions: DataFrame) -> tuple[DataFrame, pd.DataFrame]:
    """Label each test record with a band of the observed 2026T1 salary distribution.

    Cut points come from the full test salary (approxQuantile, relative error 0.001).
    Bands: <=P25, P25-P50, P50-P75, P75-P90, P90-P95 and >P95. The salary is
    heavily tied at round values, so bands are closed on the right and the table
    reports the real size of each band.
    """
    cuts = predictions.approxQuantile(LABEL, list(PERCENTILE_CUTS), 0.001)
    labels = ["P00-P25", "P25-P50", "P50-P75", "P75-P90", "P90-P95", "P95-P100"]
    expression = F.when(F.col(LABEL) <= F.lit(cuts[0]), labels[0])
    for index in range(1, len(cuts)):
        expression = expression.when(F.col(LABEL) <= F.lit(cuts[index]), labels[index])
    expression = expression.otherwise(labels[-1])
    thresholds = pd.DataFrame(
        {
            "banda": labels,
            "salario_limite_superior": [*cuts, float("nan")],
        }
    )
    return predictions.withColumn("banda_salarial", expression), thresholds


def error_by_salary_band(banded: DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """MAE, mean error and share of underestimated records per salary band and model."""
    table = (
        _long_errors(banded.select(*banded.columns))
        .join(banded.select(*RECORD_KEYS, "banda_salarial"), list(RECORD_KEYS), "inner")
        .groupBy("modelo", "banda_salarial")
        .agg(
            F.count("*").alias("n"),
            F.avg(LABEL).alias("salario_promedio"),
            F.avg("prediccion").alias("prediccion_promedio"),
            F.avg("error_abs").alias("MAE"),
            F.avg("residuo").alias("error_medio"),
            F.avg((F.col("residuo") > 0).cast("double")).alias("prop_subestimado"),
        )
        .toPandas()
        .rename(columns={"banda_salarial": "banda"})
    )
    table = table.merge(thresholds, on="banda", how="left")
    order = {name: i for i, name in enumerate(thresholds["banda"])}
    table["_o"] = table["banda"].map(order)
    table = table.sort_values(["modelo", "_o"]).drop(columns="_o")
    table["error_medio_pct_salario"] = 100 * table["error_medio"] / table["salario_promedio"]
    table["tendencia"] = table["error_medio"].map(
        lambda value: "subestima" if value > 0 else "sobreestima"
    )
    return table.reset_index(drop=True)


def residual_summary(predictions: DataFrame) -> pd.DataFrame:
    """Global residual diagnostics on the full test set (both models)."""
    return (
        _long_errors(predictions)
        .groupBy("modelo")
        .agg(
            F.count("*").alias("n"),
            F.avg("residuo").alias("error_medio"),
            F.avg("error_abs").alias("MAE"),
            F.expr("percentile_approx(residuo, 0.5, 10000)").alias("residuo_mediano"),
            F.avg((F.col("residuo") > 0).cast("double")).alias("prop_subestimado"),
        )
        .toPandas()
        .sort_values("modelo")
        .reset_index(drop=True)
    )
