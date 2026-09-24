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

from .config import MODEL_DIR, SEED


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
