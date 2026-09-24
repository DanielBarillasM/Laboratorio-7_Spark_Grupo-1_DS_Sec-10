"""Build the polished experiment notebook from small, auditable cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Laboratorio_7_Spark_MLlib_Avance.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> None:
    notebook = nbf.read(NOTEBOOK, as_version=4)
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (PySpark)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "lab7": {"scope": "avance_75", "seed": 42},
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
  <p><b>Avance funcional del 75%</b> · ENEIC Personas 2025–2026</p>
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

**Alcance ejecutado en este avance:** actividades 1–6 completas. Se preparan por separado los datos elegibles de 2026, pero se reservan la evaluación final (actividad 7) y el análisis exhaustivo de errores (actividad 8) para la entrega final.

**Diseño temporal:** 2025T1–2025T3 se utilizan para entrenamiento; 2025T4 para validación y selección. 2026T1 no se consulta para escoger configuraciones.
"""
        ),
        code(
            """
from pathlib import Path
import platform
import sys

from IPython.display import HTML, Image, display
import pandas as pd
import pyspark
from pyspark.sql import SparkSession

cwd = Path.cwd().resolve()
REPO_ROOT = cwd if (cwd / "src").exists() else cwd.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lab7.config import FIGURES_DIR, TABLES_DIR, ensure_directories
from lab7.data import (
    convert_excel_sources, counts_by_file, duplicate_key_audit,
    filter_analytical_population, harmonize, load_periods,
    missingness_before_filters, observed_source_quarters, save_prepared_sets,
)
from lab7.analysis import (
    categorical_distribution, descriptive_statistics, kmeans_search,
    median_salary_by, pearson_correlation, quarterly_summary, salary_plot_sample,
)
from lab7.modeling import (
    baseline_metrics, comparison_table, final_stage_plan,
    fit_linear_candidates, fit_random_forest_candidates,
)
from lab7.visualization import (
    plot_categorical_distributions, plot_correlation_heatmap,
    plot_group_medians, plot_kmeans_selection, plot_model_comparison,
    plot_quarterly_summary, plot_salary_distribution,
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
train = prepared_2025.filter("periodo_archivo IN ('2025T1','2025T2','2025T3')").cache()
validation = prepared_2025.filter("periodo_archivo = '2025T4'").cache()
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
<div class='lab-note'><b>Resultado de validación.</b> El menor RMSE corresponde a <code>{winner['modelo']}</code>: MAE Q{winner['MAE']:,.2f}, RMSE Q{winner['RMSE']:,.2f} y R² {winner['R2']:.4f}. Frente a la referencia, el RMSE cambia {improvement:.2f}%. Random Forest puede capturar relaciones no lineales e interacciones; la regresión lineal ofrece una estructura más parsimoniosa. La comparación definitiva aún depende del test 2026.</div>
'''))
"""
        ),
        md("## 7. Estado del avance y trabajo reservado"),
        code(
            """
pending = final_stage_plan()
display(HTML("<div class='lab-pending'><b>25% reservado para la entrega final</b><ol>" + "".join(f"<li>{item}</li>" for item in pending) + "</ol></div>"))

progress = pd.DataFrame([
    ("1. Carga, armonizacion y calidad", "Completo"),
    ("2. Estadistica descriptiva", "Completo"),
    ("3. Correlaciones", "Completo"),
    ("4. KMeans", "Completo"),
    ("5. Regresion lineal", "Completo"),
    ("6. Random Forest", "Completo"),
    ("7. Evaluacion final 2026", "Preparado; ejecucion reservada"),
    ("8. Analisis de errores", "Pendiente para entrega final"),
], columns=["actividad", "estado"])
display(progress)
progress.to_csv(TABLES_DIR / "estado_avance.csv", index=False)
"""
        ),
        md(
            """
## Conclusiones provisionales

El avance deja una ruta reproducible desde los Excel oficiales hasta Parquet, documenta el efecto de cada filtro, responde la exploración solicitada, compara alternativas de segmentación y selecciona configuraciones supervisadas sin tocar el test 2026. Los resultados no son estimaciones oficiales porque las métricas son deliberadamente no ponderadas. La entrega final deberá comprobar estabilidad fuera de tiempo y estudiar dónde se concentran los errores, especialmente en salarios altos.
"""
        ),
        code(
            """
# Liberación explícita de caché al terminar la ejecución reproducible.
for frame in [raw_2025, raw_2026, harmonized_2025, harmonized_2026, prepared_2025, prepared_2026, train, validation]:
    frame.unpersist()
print("Avance ejecutado correctamente. Spark queda disponible para inspección interactiva.")
"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK)
    print(f"Notebook construido: {NOTEBOOK}")


if __name__ == "__main__":
    build()
