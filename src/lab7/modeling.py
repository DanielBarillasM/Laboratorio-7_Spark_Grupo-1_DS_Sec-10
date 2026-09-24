"""Pipelines supervisados de Spark MLlib para el salario mensual."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame, functions as F

from .config import (
    CATEGORICAL_PREDICTORS,
    LABEL,
    MODEL_DIR,
    NUMERIC_PREDICTORS,
    SEED,
)


def regression_metrics(predictions: DataFrame) -> dict[str, float]:
    metrics = {}
    for metric_name, label in (("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")):
        evaluator = RegressionEvaluator(
            labelCol=LABEL,
            predictionCol="prediction",
            metricName=metric_name,
        )
        metrics[label] = float(evaluator.evaluate(predictions))
    return metrics


def baseline_metrics(train: DataFrame, validation: DataFrame) -> dict[str, float]:
    mean_salary = float(train.agg(F.avg(LABEL)).first()[0])
    predictions = validation.withColumn("prediction", F.lit(mean_salary))
    return {"modelo": "Referencia: media de entrenamiento", **regression_metrics(predictions)}


def _preprocessing_stages() -> list:
    index_outputs = [f"{name}_idx" for name in CATEGORICAL_PREDICTORS]
    vector_outputs = [f"{name}_ohe" for name in CATEGORICAL_PREDICTORS]
    indexers = [
        StringIndexer(
            inputCol=name,
            outputCol=output,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for name, output in zip(CATEGORICAL_PREDICTORS, index_outputs)
    ]
    encoder = OneHotEncoder(
        inputCols=index_outputs,
        outputCols=vector_outputs,
        handleInvalid="keep",
        dropLast=True,
    )
    assembler = VectorAssembler(
        inputCols=list(NUMERIC_PREDICTORS) + vector_outputs,
        outputCol="features",
        handleInvalid="error",
    )
    return [*indexers, encoder, assembler]


def fit_linear_candidates(
    train: DataFrame,
    validation: DataFrame,
    model_dir: Path = MODEL_DIR,
) -> tuple[pd.DataFrame, PipelineModel, dict[str, float]]:
    configs = (
        {"nombre": "LR_sin_regularizacion", "regParam": 0.0, "elasticNetParam": 0.0},
        {"nombre": "LR_ridge_0_1", "regParam": 0.1, "elasticNetParam": 0.0},
        {"nombre": "LR_elastic_0_1", "regParam": 0.1, "elasticNetParam": 0.5},
    )
    rows: list[dict[str, object]] = []
    fitted: dict[str, PipelineModel] = {}
    config_map = {item["nombre"]: item for item in configs}

    for config in configs:
        estimator = LinearRegression(
            featuresCol="features",
            labelCol=LABEL,
            predictionCol="prediction",
            maxIter=150,
            tol=1e-6,
            standardization=True,
            regParam=float(config["regParam"]),
            elasticNetParam=float(config["elasticNetParam"]),
        )
        model = Pipeline(stages=[*_preprocessing_stages(), estimator]).fit(train)
        metrics = regression_metrics(model.transform(validation))
        rows.append({**config, **metrics})
        fitted[str(config["nombre"])] = model

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = str(results.iloc[0]["nombre"])
    best_model = fitted[best_name]
    best_model.write().overwrite().save(str(model_dir / "linear_regression_best"))
    return results, best_model, config_map[best_name]


def fit_random_forest_candidates(
    train: DataFrame,
    validation: DataFrame,
    model_dir: Path = MODEL_DIR,
) -> tuple[pd.DataFrame, PipelineModel, dict[str, int | str]]:
    configs = (
        {"nombre": "RF_25_arboles_d5", "numTrees": 25, "maxDepth": 5},
        {"nombre": "RF_50_arboles_d7", "numTrees": 50, "maxDepth": 7},
    )
    rows: list[dict[str, object]] = []
    fitted: dict[str, PipelineModel] = {}
    config_map = {item["nombre"]: item for item in configs}

    for config in configs:
        estimator = RandomForestRegressor(
            featuresCol="features",
            labelCol=LABEL,
            predictionCol="prediction",
            numTrees=int(config["numTrees"]),
            maxDepth=int(config["maxDepth"]),
            seed=SEED,
            featureSubsetStrategy="auto",
            subsamplingRate=0.8,
            maxBins=32,
            maxMemoryInMB=64,
            cacheNodeIds=False,
        )
        model = Pipeline(stages=[*_preprocessing_stages(), estimator]).fit(train)
        metrics = regression_metrics(model.transform(validation))
        rows.append({**config, **metrics})
        fitted[str(config["nombre"])] = model

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = str(results.iloc[0]["nombre"])
    best_model = fitted[best_name]
    best_model.write().overwrite().save(str(model_dir / "random_forest_best"))
    return results, best_model, config_map[best_name]


def comparison_table(
    baseline: dict[str, float],
    linear_results: pd.DataFrame,
    forest_results: pd.DataFrame,
) -> pd.DataFrame:
    best_linear = linear_results.iloc[0]
    best_forest = forest_results.iloc[0]
    return pd.DataFrame(
        [
            baseline,
            {
                "modelo": str(best_linear["nombre"]),
                "MAE": float(best_linear["MAE"]),
                "RMSE": float(best_linear["RMSE"]),
                "R2": float(best_linear["R2"]),
            },
            {
                "modelo": str(best_forest["nombre"]),
                "MAE": float(best_forest["MAE"]),
                "RMSE": float(best_forest["RMSE"]),
                "R2": float(best_forest["R2"]),
            },
        ]
    ).sort_values("RMSE").reset_index(drop=True)


def final_stage_plan() -> tuple[str, ...]:
    """Explicit remaining work; intentionally not executed in the 75% advance."""
    return (
        "Reajustar cada configuracion ganadora con los cuatro trimestres de 2025.",
        "Generar predicciones para todos los registros elegibles de 2026T1.",
        "Calcular MAE, RMSE y R2 sobre exactamente el mismo test de 2026.",
        "Construir diagnosticos de residuos y errores por educacion y dominio.",
        "Redactar la discusion final y las limitaciones del estudio.",
    )
