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
    FINAL_FOREST_MODEL,
    FINAL_LINEAR_MODEL,
    LABEL,
    MODEL_DIR,
    NUMERIC_PREDICTORS,
    PARQUET_DIR,
    RECORD_KEYS,
    SEED,
    TEST_PREDICTIONS,
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


def _linear_estimator(config: dict) -> LinearRegression:
    return LinearRegression(
        featuresCol="features",
        labelCol=LABEL,
        predictionCol="prediction",
        maxIter=150,
        tol=1e-6,
        standardization=True,
        regParam=float(config["regParam"]),
        elasticNetParam=float(config["elasticNetParam"]),
    )


def _forest_estimator(config: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
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
        estimator = _linear_estimator(config)
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
        estimator = _forest_estimator(config)
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


def _assert_unique_keys(frame: DataFrame, name: str) -> int:
    total = frame.count()
    distinct = frame.select(*RECORD_KEYS).distinct().count()
    if total != distinct:
        raise ValueError(f"{name}: {total} filas pero {distinct} claves distintas.")
    return total


def fit_final_models(
    full_train: DataFrame,
    test: DataFrame,
    linear_config: dict,
    forest_config: dict,
    model_dir: Path = MODEL_DIR,
    parquet_dir: Path = PARQUET_DIR,
) -> dict[str, object]:
    """Refit the validation winners on all of 2025 and score them on the same 2026 rows.

    The configurations arrive already chosen on 2025T4; 2026 is used only to score.
    """
    train_rows = _assert_unique_keys(full_train, "Entrenamiento final 2025")
    test_rows = _assert_unique_keys(test, "Test 2026")

    linear_model = Pipeline(
        stages=[*_preprocessing_stages(), _linear_estimator(linear_config)]
    ).fit(full_train)
    forest_model = Pipeline(
        stages=[*_preprocessing_stages(), _forest_estimator(forest_config)]
    ).fit(full_train)
    linear_model.write().overwrite().save(str(model_dir / FINAL_LINEAR_MODEL))
    forest_model.write().overwrite().save(str(model_dir / FINAL_FOREST_MODEL))

    context = [*RECORD_KEYS, LABEL, *NUMERIC_PREDICTORS, *CATEGORICAL_PREDICTORS]
    linear_pred = linear_model.transform(test).select(
        *context, F.col("prediction").alias("prediccion_lr")
    )
    forest_pred = forest_model.transform(test).select(
        *RECORD_KEYS, F.col("prediction").alias("prediccion_rf")
    )
    linear_rows = linear_pred.count()
    forest_rows = forest_pred.count()

    # Residuo = real - predicho: positivo indica subestimacion, negativo sobreestimacion.
    baseline_mean = float(full_train.agg(F.avg(LABEL)).first()[0])
    shared = (
        linear_pred.join(forest_pred, list(RECORD_KEYS), "inner")
        .withColumn("prediccion_referencia", F.lit(baseline_mean))
        .withColumn("residuo_lr", F.col(LABEL) - F.col("prediccion_lr"))
        .withColumn("residuo_rf", F.col(LABEL) - F.col("prediccion_rf"))
    )
    predictions_path = parquet_dir / TEST_PREDICTIONS
    shared.write.mode("overwrite").parquet(str(predictions_path))
    shared = shared.sparkSession.read.parquet(str(predictions_path))
    shared_rows = shared.count()
    shared_keys = shared.select(*RECORD_KEYS).distinct().count()

    counts = pd.DataFrame(
        [
            {"conjunto": "Test elegible 2026T1", "registros": test_rows},
            {"conjunto": f"Predicciones {linear_config['nombre']}", "registros": linear_rows},
            {"conjunto": f"Predicciones {forest_config['nombre']}", "registros": forest_rows},
            {"conjunto": "Registros compartidos (join por clave)", "registros": shared_rows},
            {"conjunto": "Claves distintas compartidas", "registros": shared_keys},
        ]
    )
    if counts["registros"].nunique() != 1:
        raise ValueError(f"Los modelos no se evaluaron sobre los mismos registros:\n{counts}")

    rows = []
    for name, column, config in (
        ("Referencia: media 2025", "prediccion_referencia", "media salarial 2025 completo"),
        (str(linear_config["nombre"]), "prediccion_lr", _describe(linear_config)),
        (str(forest_config["nombre"]), "prediccion_rf", _describe(forest_config)),
    ):
        scored = shared.withColumn("prediction", F.col(column))
        rows.append(
            {"modelo": name, "configuracion": config, "registros": shared_rows,
             **regression_metrics(scored)}
        )
    metrics = pd.DataFrame(rows)

    return {
        "metrics": metrics,
        "counts": counts,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "baseline_mean": baseline_mean,
        "linear_model": linear_model,
        "forest_model": forest_model,
        "predictions": shared,
        "predictions_path": predictions_path,
    }


def _describe(config: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in config.items() if key != "nombre")


def final_comparison_table(
    validation_comparison: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Place validation (2025T4) and test (2026T1) metrics side by side."""
    validation = validation_comparison.replace(
        {"modelo": {"Referencia: media de entrenamiento": "Referencia: media 2025"}}
    ).rename(columns={"MAE": "MAE_validacion", "RMSE": "RMSE_validacion", "R2": "R2_validacion"})
    table = test_metrics.merge(validation, on="modelo", how="left")
    baseline_rmse = float(table.loc[table["modelo"] == "Referencia: media 2025", "RMSE"].iloc[0])
    table["mejora_RMSE_vs_referencia_pct"] = 100 * (baseline_rmse - table["RMSE"]) / baseline_rmse
    table["cambio_RMSE_validacion_a_test"] = table["RMSE"] - table["RMSE_validacion"]
    return table.sort_values("RMSE").reset_index(drop=True)

