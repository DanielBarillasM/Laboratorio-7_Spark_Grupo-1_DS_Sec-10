"""Build the polished experiment notebook from small, auditable cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Laboratorio_7_Spark_MLlib_Final.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (PySpark)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "lab7": {"scope": "actividades_1_a_8", "seed": 42},
        }
    )
    notebook.cells = [
        md(
            """
<style>
:root { --navy:#172033; --blue:#2563eb; --cyan:#06b6d4; --paper:#f8fafc; --line:#dbeafe; }
.lab-hero { padding:32px 36px; border-radius:20px; color:white; background:linear-gradient(125deg,#172033 0%,#1d4ed8 60%,#06b6d4 100%); box-shadow:0 12px 28px rgba(37,99,235,.22); }
.lab-hero h1 { margin:0 0 8px; font-size:34px; letter-spacing:-.7px; }
.lab-hero p { margin:6px 0; opacity:.94; }
.lab-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:18px 0; }
.lab-card { padding:16px; border:1px solid var(--line); border-radius:14px; background:var(--paper); }
.lab-card b { color:var(--blue); }
.lab-note { border-left:5px solid var(--cyan); padding:12px 16px; background:#ecfeff; border-radius:8px; }
.lab-pending { border-left:5px solid #f59e0b; padding:12px 16px; background:#fffbeb; border-radius:8px; }
table { font-size:13px !important; }
</style>

<div class="lab-hero">
  <h1>Laboratorio 7 · Spark MLlib</h1>
  <p><b>Actividades 1–8: exploración, segmentación, modelado, evaluación en 2026 y análisis de errores</b> · ENEIC Personas 2025–2026</p>
  <p>Universidad del Valle de Guatemala · CC3084 Data Science · Sección 10 · Grupo 1 · Segundo semestre 2026</p>
</div>

<div class="lab-grid">
  <div class="lab-card"><b>Jorge Gabriel Palacios Sales</b><br>231385</div>
  <div class="lab-card"><b>Pablo Daniel Barillas Moreno</b><br>22193</div>
  <div class="lab-card"><b>Roberto Emiliano Otoniel</b><br>23968</div>
</div>
"""
        ),
        md(
            """
## 0. Objetivo, preguntas y alcance

Este notebook identifica perfiles de trabajadores asalariados y evalúa qué tan bien puede estimarse su salario mensual mediante Spark 3.5.1 y `pyspark.ml`. El procedimiento conserva la trazabilidad por archivo, evita interpretar asociaciones como causalidad y calcula las métricas sobre los conjuntos completos.

**Alcance ejecutado:** actividades 1–8. La sección 8 analiza los errores sobre las predicciones de 2026T1 que deja guardadas la sección 7.

**Diseño temporal:** 2025T1–2025T3 se utilizan para entrenamiento; 2025T4 para validación y selección. Después, cada configuración ganadora se reentrena con los cuatro trimestres de 2025 y se evalúa una sola vez en 2026T1. 2026T1 nunca se consulta para escoger configuraciones.
"""
        ),
        code(
            """
from pathlib import Path
import platform
import sys

from IPython.display import HTML, Image, Markdown, display
import pandas as pd
import pyspark
from pyspark.sql import SparkSession, functions as F

cwd = Path.cwd().resolve()
REPO_ROOT = cwd if (cwd / "src").exists() else cwd.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lab7.config import (
    FIGURES_DIR, FINAL_FOREST_MODEL, FINAL_LINEAR_MODEL, FINAL_TRAIN_PERIODS,
    MODEL_DIR, TABLES_DIR, TEST_PERIODS, TRAIN_PERIODS, VALIDATION_PERIODS,
    ensure_directories,
)
from lab7.data import (
    convert_excel_sources, counts_by_file, duplicate_key_audit,
    filter_analytical_population, harmonize, load_periods,
    missingness_before_filters, observed_source_quarters, save_prepared_sets,
)
from lab7.analysis import (
    categorical_distribution, descriptive_statistics, error_by_group,
    error_by_salary_band, kmeans_search, load_test_predictions, median_salary_by,
    pearson_correlation, plot_comparison_sample, quarterly_summary,
    residual_summary, salary_percentile_bands, salary_plot_sample,
)
from lab7.interpretation import (
    band_context, band_findings, final_discussion, group_findings,
    plot_findings, question_answers, salary_scale_sentence,
)
from lab7.modeling import (
    baseline_metrics, comparison_table, final_comparison_table,
    fit_final_models, fit_linear_candidates, fit_random_forest_candidates,
)
from lab7.visualization import (
    plot_actual_vs_predicted, plot_categorical_distributions,
    plot_correlation_heatmap, plot_errors_by_domain, plot_errors_by_education,
    plot_errors_by_salary_band, plot_group_medians, plot_kmeans_selection,
    plot_model_comparison, plot_quarterly_summary, plot_residuals,
    plot_salary_distribution,
)

ensure_directories()
spark = (
    SparkSession.builder
    .appName("Lab7-ENEIC-SparkMLlib")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

versions = pd.DataFrame({
    "componente": ["Python", "PySpark", "Java", "Semilla"],
    "valor": [platform.python_version(), pyspark.__version__, spark.sparkContext._jvm.java.lang.System.getProperty("java.version"), "42"],
})
display(versions)
assert pyspark.__version__.startswith("3.5"), "La guía requiere Spark 3.5.x"
"""
        ),
        md(
            """
## 1. Carga, armonización y calidad

Cada Excel se procesa por separado y únicamente con las 14 columnas requeridas. Primero se crea un DataFrame de Spark con esquema explícito de texto —para evitar inferencias distintas entre trimestres— y luego se guarda en Parquet. Las conversiones numéricas y categóricas se realizan posteriormente en Spark.
"""
        ),
        code(
            """
conversion_audit = convert_excel_sources(spark, force=False)
display(conversion_audit)
assert (conversion_audit["registros"] == conversion_audit["esperados"]).all()

raw_2025 = load_periods(spark, ["2025T1", "2025T2", "2025T3", "2025T4"]).cache()
raw_2026 = load_periods(spark, ["2026T1"]).cache()
harmonized_2025 = harmonize(raw_2025).cache()
harmonized_2026 = harmonize(raw_2026).cache()

selected_view = [
    "periodo_archivo", "archivo_origen", "TRIMESTRE", "NUM_HOGAR", "NUM_PERSONA",
    "edad", "ocupado", "categoria_ocupacional", "salario_mensual", "antiguedad",
    "horas_semanales", "nivel_educativo", "dominio", "factor",
]
harmonized_2025.select(*selected_view).printSchema()
harmonized_2025.select(*selected_view).show(5, truncate=False)
"""
        ),
        code(
            """
missingness_2025 = missingness_before_filters(raw_2025)
prepared_2025, filter_audit_2025 = filter_analytical_population(harmonized_2025)
prepared_2026, filter_audit_2026 = filter_analytical_population(harmonized_2026)
prepared_2025 = prepared_2025.cache()
prepared_2026 = prepared_2026.cache()

file_counts = counts_by_file(harmonized_2025, prepared_2025)
source_quarters = observed_source_quarters(raw_2025)
key_audit = duplicate_key_audit(harmonized_2025)

for name, table in {
    "conversion_fuentes.csv": conversion_audit,
    "faltantes_antes_filtros_2025.csv": missingness_2025,
    "auditoria_filtros_2025.csv": filter_audit_2025,
    "auditoria_filtros_2026.csv": filter_audit_2026,
    "conteos_por_archivo.csv": file_counts,
    "trimestre_fuente_observado.csv": source_quarters,
    "auditoria_claves.csv": key_audit,
}.items():
    table.to_csv(TABLES_DIR / name, index=False)

display(HTML("<h4>Faltantes antes de filtrar (2025)</h4>"))
display(missingness_2025.style.format({"porcentaje": "{:.2f}%"}))
display(HTML("<h4>Embudo de filtros 2025</h4>"))
display(filter_audit_2025)
display(HTML("<h4>Conteos por archivo</h4>"))
display(file_counts.style.format({"retencion_pct": "{:.2f}%"}))
display(HTML("<h4>TRIMESTRE original conservado</h4>"))
display(source_quarters)
display(HTML("<h4>Unicidad de periodo + hogar + persona</h4>"))
display(key_audit)

save_prepared_sets(prepared_2025, prepared_2026)
"""
        ),
        md(
            """
### Interpretación de calidad

- **2025-IV no se apila por posición:** contiene 302 columnas frente a 270 en los otros archivos. `unionByName` evita asociar variables distintas por su posición física.
- **Ausente por diseño vs. no respuesta:** una pregunta puede no corresponder por el flujo del cuestionario; una no respuesta sí era aplicable, pero quedó sin registrar. Ambos casos pueden verse vacíos, aunque tienen significados estadísticos diferentes.
- **Longitudinalidad:** la misma persona en dos períodos es una observación válida distinta; solo se audita la unicidad dentro de `periodo_archivo`.
- **Alcance:** la base filtrada no representa por sí sola a todos los trabajadores de Guatemala. Este análisis no ponderado describe registros elegibles. `FACTOR` se conserva para una futura estimación poblacional con el diseño muestral apropiado.
- Los códigos categóricos desconocidos se representan como `DESCONOCIDO`; el código educativo 0 se conserva como `Ninguno`.
"""
        ),
        md("## 2. Estadística descriptiva y exploración"),
        code(
            """
descriptive = descriptive_statistics(prepared_2025)
distributions = pd.concat([
    categorical_distribution(prepared_2025, "categoria_ocupacional"),
    categorical_distribution(prepared_2025, "nivel_educativo"),
    categorical_distribution(prepared_2025, "dominio"),
], ignore_index=True)
salary_groups = pd.concat([
    median_salary_by(prepared_2025, "nivel_educativo"),
    median_salary_by(prepared_2025, "categoria_ocupacional"),
], ignore_index=True)
quarters = quarterly_summary(prepared_2025)
salary_sample = salary_plot_sample(prepared_2025, 5_000)

descriptive.to_csv(TABLES_DIR / "estadistica_descriptiva_2025.csv", index=False)
distributions.to_csv(TABLES_DIR / "distribuciones_categoricas.csv", index=False)
salary_groups.to_csv(TABLES_DIR / "salario_por_grupo.csv", index=False)
quarters.to_csv(TABLES_DIR / "resumen_trimestral.csv", index=False)

plot_categorical_distributions(distributions)
plot_salary_distribution(salary_sample)
plot_group_medians(salary_groups)
plot_quarterly_summary(quarters)

display(descriptive.style.format({c: "{:,.2f}" for c in descriptive.columns if c not in ["variable", "n"]}))
display(Image(filename=str(FIGURES_DIR / "distribuciones_categoricas.png")))
display(Image(filename=str(FIGURES_DIR / "distribucion_salario.png")))
display(Image(filename=str(FIGURES_DIR / "salario_mediano_grupos.png")))
display(Image(filename=str(FIGURES_DIR / "evolucion_trimestral.png")))

salary_row = descriptive.loc[descriptive["variable"] == "salario_mensual"].iloc[0]
shape = "asimetría positiva" if salary_row["media"] > salary_row["mediana"] else "asimetría negativa o débil"
display(HTML(f'''
<div class='lab-note'><b>Lectura descriptiva.</b> El salario presenta {shape}: la media es Q{salary_row['media']:,.2f} y la mediana Q{salary_row['mediana']:,.2f}. La diferencia y el máximo elevado justifican conservar los extremos para modelar, pero interpretar RMSE junto con MAE.</div>
'''))
"""
        ),
        md("## 3. Relaciones entre variables numéricas"),
        code(
            """
correlation = pearson_correlation(prepared_2025)
correlation.to_csv(TABLES_DIR / "correlaciones_pearson.csv")
plot_correlation_heatmap(correlation)
display(correlation.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1))
display(Image(filename=str(FIGURES_DIR / "correlaciones_pearson.png")))

salary_associations = correlation["salario_mensual"].drop("salario_mensual").abs().sort_values(ascending=False)
strongest = salary_associations.index[0]
age_tenure = correlation.loc["edad", "antiguedad"]
display(HTML(f'''
<div class='lab-note'><b>Interpretación.</b> Entre los predictores numéricos, <code>{strongest}</code> tiene la mayor asociación lineal absoluta con el salario (|r|={salary_associations.iloc[0]:.3f}). La relación edad-antigüedad es r={age_tenure:.3f}; es esperable que exista asociación, pero no equivalencia, porque personas de la misma edad pueden tener trayectorias laborales distintas.</div>
'''))
"""
        ),
        md(
            """
## 4. Segmentación de perfiles con KMeans

Se comparan K=2,3,4,5 en dos escenarios estandarizados: un perfil laboral sin salario y un perfil económico que sí lo incorpora. Esta comparación permite decidir con evidencia si el salario aporta separación o domina artificialmente los perfiles.
"""
        ),
        code(
            """
kmeans_results, cluster_profiles, kmeans_model, best_scenario, best_k = kmeans_search(prepared_2025)
kmeans_results.to_csv(TABLES_DIR / "seleccion_kmeans.csv", index=False)
cluster_profiles.to_csv(TABLES_DIR / "perfiles_clusters.csv", index=False)
plot_kmeans_selection(kmeans_results)

display(kmeans_results.style.format({"silhouette": "{:.4f}"}))
display(cluster_profiles.style.format({c: "{:,.2f}" for c in cluster_profiles.columns if c not in ["cluster", "n", "descripcion"]}))
display(Image(filename=str(FIGURES_DIR / "seleccion_kmeans.png")))
display(HTML(f'''
<div class='lab-note'><b>Selección:</b> el mejor resultado fue <i>{best_scenario}</i> con K={best_k}. La selección maximiza silhouette y la tabla de perfiles asigna descripciones relativas a las medias globales; son segmentos descriptivos, no clases naturales ni juicios sobre las personas.</div>
'''))
"""
        ),
        md(
            """
## 5–6. Modelado supervisado y validación temporal

Se usan exactamente seis predictores: edad, antigüedad, horas semanales, nivel educativo, categoría ocupacional y dominio. Todos los transformadores se ajustan únicamente con 2025T1–T3. La validación se realiza en 2025T4 y se compara contra la media salarial del entrenamiento.
"""
        ),
        code(
            """
train = prepared_2025.filter(F.col("periodo_archivo").isin(*TRAIN_PERIODS)).cache()
validation = prepared_2025.filter(F.col("periodo_archivo").isin(*VALIDATION_PERIODS)).cache()
split_counts = pd.DataFrame({
    "conjunto": ["Entrenamiento 2025T1-T3", "Validacion 2025T4", "Test reservado 2026T1"],
    "registros": [train.count(), validation.count(), prepared_2026.count()],
})
display(split_counts)
split_counts.to_csv(TABLES_DIR / "particiones_modelado.csv", index=False)

baseline = baseline_metrics(train, validation)
linear_results, linear_model, best_linear_config = fit_linear_candidates(train, validation)
forest_results, forest_model, best_forest_config = fit_random_forest_candidates(train, validation)
comparison = comparison_table(baseline, linear_results, forest_results)

linear_results.to_csv(TABLES_DIR / "metricas_regresion_lineal.csv", index=False)
forest_results.to_csv(TABLES_DIR / "metricas_random_forest.csv", index=False)
comparison.to_csv(TABLES_DIR / "comparacion_modelos_validacion.csv", index=False)
plot_model_comparison(comparison)

display(HTML("<h4>Regresión lineal: configuraciones</h4>"))
display(linear_results.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}))
display(HTML("<h4>Random Forest: configuraciones</h4>"))
display(forest_results.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}))
display(HTML("<h4>Comparación con referencia</h4>"))
display(comparison.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}))
display(Image(filename=str(FIGURES_DIR / "comparacion_modelos_validacion.png")))

winner = comparison.iloc[0]
baseline_rmse = float(comparison.loc[comparison["modelo"] == "Referencia: media de entrenamiento", "RMSE"].iloc[0])
improvement = 100 * (baseline_rmse - float(winner["RMSE"])) / baseline_rmse
display(HTML(f'''
<div class='lab-note'><b>Resultado de validación.</b> El menor RMSE corresponde a <code>{winner['modelo']}</code>: MAE Q{winner['MAE']:,.2f}, RMSE Q{winner['RMSE']:,.2f} y R² {winner['R2']:.4f}. Frente a la referencia, el RMSE mejora {improvement:.2f}%. Random Forest puede capturar relaciones no lineales e interacciones; la regresión lineal ofrece una estructura más parsimoniosa. La comparación definitiva se hace en la sección 7 con el test 2026.</div>
'''))
"""
        ),
        md(
            """
## 7. Entrenamiento final y evaluación en 2026

La selección de hiperparámetros ya quedó cerrada con 2025T4 en la sección anterior: se toma la configuración con menor RMSE de validación por algoritmo. Aquí **no se vuelve a escoger nada**; cada pipeline completo (`StringIndexer` → `OneHotEncoder` → `VectorAssembler` → modelo) se ajusta de nuevo con los cuatro trimestres de 2025 y se aplica una sola vez a 2026T1.

- 2026T1 pasó por las mismas reglas de preparación (`harmonize` + `filter_analytical_population`) que 2025; su embudo se guardó en `auditoria_filtros_2026.csv`.
- Ambos modelos se unen por `periodo_archivo`, `NUM_HOGAR` y `NUM_PERSONA` antes de calcular métricas, de modo que se evalúan sobre exactamente los mismos registros.
- La referencia predice la media salarial de 2025 completo (el mismo conjunto con el que se reentrenan los modelos).
- Las métricas son no ponderadas y se calculan sobre todo el test; el salario objetivo no se recorta ni se transforma.
- Residuo = salario real − salario predicho: positivo indica subestimación y negativo sobreestimación.
"""
        ),
        code(
            """
full_train_2025 = prepared_2025.filter(F.col("periodo_archivo").isin(*FINAL_TRAIN_PERIODS)).cache()
test_2026 = prepared_2026.filter(F.col("periodo_archivo").isin(*TEST_PERIODS)).cache()
assert full_train_2025.count() == prepared_2025.count(), "El reentrenamiento debe usar todo 2025"
assert test_2026.count() == prepared_2026.count()

test_key_audit = duplicate_key_audit(harmonized_2026)
assert int(test_key_audit["grupos_duplicados"].iloc[0]) == 0, "Claves duplicadas en 2026"

selected_configs = pd.DataFrame([
    {"algoritmo": "Regresion lineal", **best_linear_config,
     "RMSE_validacion_2025T4": float(linear_results.iloc[0]["RMSE"])},
    {"algoritmo": "Random Forest", **best_forest_config,
     "RMSE_validacion_2025T4": float(forest_results.iloc[0]["RMSE"])},
])
display(HTML("<h4>Configuraciones seleccionadas con 2025T4</h4>"))
display(selected_configs)

final = fit_final_models(full_train_2025, test_2026, best_linear_config, best_forest_config)
test_metrics = final["metrics"]
shared_counts = final["counts"]
final_comparison = final_comparison_table(comparison, test_metrics)

test_metrics.to_csv(TABLES_DIR / "metricas_test_2026.csv", index=False)
final_comparison.to_csv(TABLES_DIR / "comparacion_final_modelos.csv", index=False)
shared_counts.to_csv(TABLES_DIR / "conteo_test_compartido.csv", index=False)
test_key_audit.to_csv(TABLES_DIR / "auditoria_claves_2026.csv", index=False)

display(HTML("<h4>Registros de entrenamiento final y prueba</h4>"))
display(pd.DataFrame({
    "conjunto": ["Entrenamiento final 2025T1-T4", "Prueba 2026T1"],
    "registros": [final["train_rows"], final["test_rows"]],
}))
display(HTML("<h4>Mismo test para ambos modelos</h4>"))
display(shared_counts)
assert shared_counts["registros"].nunique() == 1
display(HTML("<h4>Métricas en 2026T1 (no ponderadas, test completo)</h4>"))
display(test_metrics.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R2": "{:.4f}"}))
display(HTML("<h4>Validación 2025T4 frente a prueba 2026T1</h4>"))
money = [c for c in final_comparison.columns if c.startswith(("MAE", "RMSE", "mejora", "cambio"))]
display(final_comparison.style.format({**{c: "{:,.2f}" for c in money}, "R2": "{:.4f}", "R2_validacion": "{:.4f}"}))

for name in (FINAL_LINEAR_MODEL, FINAL_FOREST_MODEL):
    assert (MODEL_DIR / name / "metadata").exists(), f"No se guardó {name}"
print("Modelos finales:", MODEL_DIR / FINAL_LINEAR_MODEL, "|", MODEL_DIR / FINAL_FOREST_MODEL)
print("Predicciones 2026 (Parquet):", final["predictions_path"])
"""
        ),
        code(
            """
models_only = final_comparison[~final_comparison["modelo"].str.startswith("Referencia")]
test_winner = models_only.iloc[0]
test_runner = models_only.iloc[1]
reference = final_comparison[final_comparison["modelo"].str.startswith("Referencia")].iloc[0]
validation_winner = comparison[~comparison["modelo"].str.startswith("Referencia")].iloc[0]["modelo"]
same_winner = "coincide" if validation_winner == test_winner["modelo"] else "no coincide"

display(HTML(f'''
<div class='lab-note'><b>Interpretación de la prueba 2026.</b>
<ul>
<li><b>Mejor modelo en 2026T1:</b> <code>{test_winner['modelo']}</code> con MAE Q{test_winner['MAE']:,.2f}, RMSE Q{test_winner['RMSE']:,.2f} y R² {test_winner['R2']:.4f}. El otro algoritmo (<code>{test_runner['modelo']}</code>) obtuvo MAE Q{test_runner['MAE']:,.2f}, RMSE Q{test_runner['RMSE']:,.2f} y R² {test_runner['R2']:.4f}.</li>
<li><b>Frente a la referencia:</b> la media de 2025 produce RMSE Q{reference['RMSE']:,.2f} y R² {reference['R2']:.4f}; el ganador reduce el RMSE en {test_winner['mejora_RMSE_vs_referencia_pct']:.2f}%. Que la referencia tenga R² cercano a cero es esperable, porque predice una constante.</li>
<li><b>Estabilidad temporal:</b> el RMSE del ganador cambia Q{test_winner['cambio_RMSE_validacion_a_test']:,.2f} entre validación 2025T4 y prueba 2026T1, y el ganador de validación ({validation_winner}) {same_winner} con el de prueba. Esto indica qué tan bien se sostiene fuera de tiempo la selección hecha solo con 2025.</li>
<li><b>Por qué un algoritmo generaliza mejor:</b> Random Forest captura no linealidades (por ejemplo, rendimientos decrecientes de la edad o la antigüedad) e interacciones entre educación, categoría ocupacional y dominio sin especificarlas; la regresión lineal impone efectos aditivos y constantes. Aun así, ambos dejan sin explicar una parte importante de la variabilidad: faltan variables relevantes (ocupación, rama de actividad, tamaño de empresa) y los salarios extremos pesan mucho en el RMSE.</li>
</ul>
Estas métricas describen los registros elegibles analizados; no son estimaciones oficiales ni implican causalidad.</div>
'''))
"""
        ),
        md(
            """
## 8. Visualización y análisis de errores

Esta sección **no reentrena ni vuelve a seleccionar nada**: lee las predicciones de 2026T1 que guardó la sección 7 (una fila por registro de prueba, con la predicción de ambos modelos) y estudia dónde se equivocan.

- **Residuo = salario real − salario predicho.** Un residuo **positivo** indica **subestimación** (el modelo predijo menos de lo observado); uno **negativo** indica **sobreestimación**.
- Los gráficos usan **una misma muestra reproducible de 5,000 registros** (orden por un hash de la clave del registro con semilla 42) para poder comparar ambos modelos sobre los mismos puntos.
- **Todas las tablas y métricas por grupo se calculan con los 13 mil y tantos registros completos de prueba**, no con la muestra gráfica, y no están ponderadas.
"""
        ),
        code(
            """
predictions_2026 = load_test_predictions(spark).cache()
n_predictions = predictions_2026.count()
n_keys = predictions_2026.select("periodo_archivo", "NUM_HOGAR", "NUM_PERSONA").distinct().count()
assert n_predictions == n_keys == test_2026.count(), "Las predicciones deben cubrir exactamente el test 2026"

# Coherencia con la sección 7: las métricas recalculadas desde el Parquet deben coincidir.
recomputed = residual_summary(predictions_2026)
for algorithm, column in (("Regresión lineal", "prediccion_lr"), ("Random Forest", "prediccion_rf")):
    mae_here = float(recomputed.loc[recomputed["modelo"] == algorithm, "MAE"].iloc[0])
    row = test_metrics[test_metrics["modelo"].str.startswith("LR" if column == "prediccion_lr" else "RF")].iloc[0]
    assert abs(mae_here - float(row["MAE"])) < 1e-6, f"MAE inconsistente para {algorithm}"

plot_sample = plot_comparison_sample(predictions_2026, 5_000)
assert len(plot_sample) <= 5_000
plot_actual_vs_predicted(plot_sample)
plot_residuals(plot_sample)
recomputed.to_csv(TABLES_DIR / "resumen_residuos_2026.csv", index=False)

display(HTML(f"<h4>Registros de prueba: {n_predictions:,} (mismas claves para ambos modelos) · muestra gráfica: {len(plot_sample):,}</h4>"))
display(HTML("<h4>Residuos globales sobre todo el test (residuo = real − predicho)</h4>"))
display(recomputed.style.format({"error_medio": "{:,.2f}", "MAE": "{:,.2f}", "residuo_mediano": "{:,.2f}", "prop_subestimado": "{:.1%}"}))
display(Image(filename=str(FIGURES_DIR / "real_vs_predicho_2026.png")))
display(Image(filename=str(FIGURES_DIR / "residuos_vs_predicho_2026.png")))
"""
        ),
        code(
            """
summary = recomputed.set_index("modelo")
plot_reading = plot_findings(plot_sample)
(TABLES_DIR / "interpretacion_graficos_2026.txt").write_text(plot_reading, encoding="utf-8")
lines = []
for model in summary.index:
    mean_error = float(summary.loc[model, "error_medio"])
    lines.append(
        f"- **{model}:** error medio global {mean_error:+,.2f} Q "
        f"({'subestima' if mean_error > 0 else 'sobreestima'} en promedio); "
        f"subestima al {summary.loc[model, 'prop_subestimado']:.1%} de los registros."
    )
display(Markdown(
    "### Interpretación de los gráficos de predicción y residuos\\n\\n"
    "**Cómo leerlos.** En el gráfico real vs. predicho, una predicción perfecta caería sobre la línea y = x: "
    "los puntos **por encima** de la línea son salarios reales mayores que la predicción (subestimación) y los "
    "puntos por debajo son sobreestimaciones. En el gráfico de residuos, la línea en cero separa subestimación "
    "(arriba) de sobreestimación (abajo); una nube sin estructura alrededor de cero indicaría un ajuste sin sesgo "
    "sistemático.\\n\\n"
    "**Resultados globales (todo el test).**\\n\\n" + "\\n".join(lines) + "\\n\\n"
    "**Lo que muestran las gráficas (muestra común de 5,000 registros).**\\n\\n" + plot_reading + "\\n\\n"
    "_Estas lecturas usan la muestra gráfica; las métricas y tablas por grupo se calculan con todo el test._"
))
"""
        ),
        md(
            """
### 8.1 Errores por nivel educativo y por dominio

Para cada grupo se reportan el **número de observaciones**, el **MAE** y el **error medio** (real − predicho; positivo = subestima) con todos los registros de prueba.
"""
        ),
        code(
            """
by_education = error_by_group(predictions_2026, "nivel_educativo")
by_domain = error_by_group(predictions_2026, "dominio")
by_education.to_csv(TABLES_DIR / "errores_por_educacion_2026.csv", index=False)
by_domain.to_csv(TABLES_DIR / "errores_por_dominio_2026.csv", index=False)
plot_errors_by_education(by_education)
plot_errors_by_domain(by_domain)

fmt = {"n": "{:,}", "MAE": "{:,.2f}", "error_medio": "{:,.2f}", "salario_promedio": "{:,.2f}"}
display(HTML("<h4>Por nivel educativo</h4>"))
display(by_education.drop(columns="variable").style.format(fmt))
display(Image(filename=str(FIGURES_DIR / "errores_por_educacion_2026.png")))
display(HTML("<h4>Por dominio</h4>"))
display(by_domain.drop(columns="variable").style.format(fmt))
display(Image(filename=str(FIGURES_DIR / "errores_por_dominio_2026.png")))
"""
        ),
        code(
            """
display(Markdown(
    "### Interpretación por nivel educativo\\n\\n" + group_findings(by_education, "nivel educativo") + "\\n\\n"
    + salary_scale_sentence(by_education, "nivel educativo") + " Un error medio positivo significa que, en promedio, "
    "el modelo predijo menos de lo observado para ese grupo.\\n\\n"
    "### Interpretación por dominio\\n\\n" + group_findings(by_domain, "dominio") + "\\n\\n"
    + salary_scale_sentence(by_domain, "dominio")
))"""
        ),
        md(
            """
### 8.2 Errores según el percentil del salario

Se divide el test en bandas del **salario real** de 2026T1 (P0–P25, P25–P50, P50–P75, P75–P90, P90–P95 y P95–P100; los cortes salen del test completo con `approxQuantile`). Como muchos salarios se repiten en cifras redondas, las bandas se cierran por la derecha y la tabla informa su tamaño real.

**Advertencia de lectura:** agrupar por el salario real induce, incluso en un modelo razonable, una sobreestimación en las bandas bajas y una subestimación en las altas (regresión a la media). Por eso el hallazgo importa sobre todo por su **magnitud** y por compararlo entre algoritmos.
"""
        ),
        code(
            """
banded, thresholds = salary_percentile_bands(predictions_2026)
by_band = error_by_salary_band(banded, thresholds)
by_band.to_csv(TABLES_DIR / "errores_por_percentil_salarial_2026.csv", index=False)
plot_errors_by_salary_band(by_band)

display(HTML("<h4>Cortes de percentil del salario real (Q)</h4>"))
display(thresholds.style.format({"salario_limite_superior": "{:,.2f}"}, na_rep="—"))
display(HTML("<h4>Error por banda y modelo</h4>"))
display(by_band.style.format({
    "n": "{:,}", "salario_promedio": "{:,.2f}", "prediccion_promedio": "{:,.2f}", "MAE": "{:,.2f}",
    "error_medio": "{:,.2f}", "prop_subestimado": "{:.1%}", "salario_limite_superior": "{:,.2f}",
    "error_medio_pct_salario": "{:+.1f}%",
}, na_rep="—"))
display(Image(filename=str(FIGURES_DIR / "errores_por_percentil_salarial_2026.png")))
"""
        ),
        code(
            """
display(Markdown(
    "### Interpretación por percentil salarial\\n\\n" + band_findings(by_band) + "\\n\\n"
    "**Salarios bajos, medios y altos.** " + band_context(test_metrics, by_band)
))"""
        ),
        md("### 8.3 Respuestas a las preguntas de la actividad"),
        code(
            """
display(Markdown(question_answers(test_metrics, by_education, by_domain, by_band)))
"""
        ),
        md("## Conclusiones finales"),
        code(
            """
display(Markdown(final_discussion(test_metrics, by_education, by_domain, by_band)))
"""
        ),
        md(
            """
### Estado de las actividades

| Actividad | Estado |
|---|---|
| 1. Carga, armonización y calidad | Completa |
| 2. Estadística descriptiva | Completa |
| 3. Correlaciones con MLlib | Completa |
| 4. Segmentación KMeans | Completa |
| 5. Pipeline de regresión lineal | Completa |
| 6. Pipeline de Random Forest | Completa |
| 7. Entrenamiento final y evaluación en 2026 | Completa |
| 8. Visualización y análisis de errores | Completa |
"""
        ),
        code(
            """
progress = pd.DataFrame({
    "actividad": [
        "1. Carga, armonizacion y calidad", "2. Estadistica descriptiva", "3. Correlaciones",
        "4. KMeans", "5. Regresion lineal", "6. Random Forest",
        "7. Evaluacion final 2026", "8. Analisis de errores",
    ],
    "estado": ["Completo"] * 8,
})
progress.to_csv(TABLES_DIR / "estado_avance.csv", index=False)

# Liberación explícita de caché al terminar la ejecución reproducible.
for frame in [raw_2025, raw_2026, harmonized_2025, harmonized_2026, prepared_2025, prepared_2026,
              train, validation, full_train_2025, test_2026, predictions_2026]:
    frame.unpersist()
print("Notebook ejecutado correctamente de principio a fin. Spark queda disponible para inspección interactiva.")
"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)
    print(f"Notebook construido: {NOTEBOOK}")


if __name__ == "__main__":
    build()
