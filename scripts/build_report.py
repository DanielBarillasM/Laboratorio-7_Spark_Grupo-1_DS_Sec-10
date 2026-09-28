"""Genera los fragmentos LaTeX del informe final a partir de los CSV del notebook ejecutado.

Uso (despues de ejecutar el notebook completo en Docker):

    python scripts/build_report.py
    cd reports && pdflatex informe_final.tex && pdflatex informe_final.tex

Todas las cifras y la interpretacion salen de las tablas calculadas con Spark sobre
los conjuntos completos; este script no recalcula ni inventa resultados.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab7 import interpretation as words  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "reports"

REQUIRED = [
    "conteos_por_archivo.csv",
    "estadistica_descriptiva_2025.csv",
    "correlaciones_pearson.csv",
    "seleccion_kmeans.csv",
    "perfiles_clusters.csv",
    "comparacion_modelos_validacion.csv",
    "metricas_test_2026.csv",
    "comparacion_final_modelos.csv",
    "conteo_test_compartido.csv",
    "errores_por_educacion_2026.csv",
    "errores_por_dominio_2026.csv",
    "errores_por_percentil_salarial_2026.csv",
    "resumen_residuos_2026.csv",
    "interpretacion_graficos_2026.txt",
]


def tex(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def money(value: float) -> str:
    return f"Q{value:,.2f}"


def inline(text: str) -> str:
    """Escape LaTeX and convert the small Markdown subset used by interpretation.py."""
    escaped = tex(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
    escaped = re.sub(r"`(.+?)`", r"\\texttt{\1}", escaped)
    escaped = re.sub(r"\bR2\b", r"$R^2$", escaped)
    escaped = escaped.replace("→", r"$\to$")
    return escaped


def blocks(markdown: str) -> str:
    """Convert paragraphs, '- ' bullets and '1. ' items into LaTeX."""
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", markdown.strip()):
        lines = [line for line in paragraph.split("\n") if line.strip()]
        if all(line.startswith("- ") for line in lines):
            out.append(r"\begin{itemize}\setlength{\itemsep}{2pt}")
            out.extend(rf"\item {inline(line[2:])}" for line in lines)
            out.append(r"\end{itemize}")
        elif len(lines) == 1 and re.match(r"\d+\. ", lines[0]):
            number, rest = lines[0].split(". ", 1)
            out.append(rf"\par\noindent\textbf{{{number}.}} {inline(rest)}")
        else:
            out.append(inline(" ".join(lines)))
        out.append("")
    return "\n".join(out)


def table(header: list[str], rows: list[list[str]], spec: str, caption: str, size: str = "small") -> list[str]:
    lines = [rf"\begin{{table}}[htbp]\centering\{size}", rf"\begin{{tabular}}{{{spec}}}\toprule"]
    lines.append(" & ".join(header) + r" \\ \midrule")
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"])
    return lines


def wide_group_table(data: pd.DataFrame, first_header: str, caption: str) -> list[str]:
    """One row per group with n, MAE and mean error for both algorithms."""
    lr = data[data["modelo"] == "Regresión lineal"].set_index("grupo")
    rf = data[data["modelo"] == "Random Forest"].set_index("grupo")
    rows = []
    for group in lr.index:
        rows.append(
            [
                tex(group),
                f"{int(lr.loc[group, 'n']):,}",
                f"{lr.loc[group, 'MAE']:,.2f}",
                f"{lr.loc[group, 'error_medio']:+,.2f}",
                f"{rf.loc[group, 'MAE']:,.2f}",
                f"{rf.loc[group, 'error_medio']:+,.2f}",
            ]
        )
    header = [first_header, "n", "MAE RL", "Err. medio RL", "MAE RF", "Err. medio RF"]
    return table(header, rows, "lrrrrr", caption, "footnotesize")


def build() -> None:
    missing = [name for name in REQUIRED if not (TABLES / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan resultados del notebook. Ejecute Laboratorio_7_Spark_MLlib_Final.ipynb completo "
            f"dentro de Docker antes de generar el informe: {missing}"
        )

    counts = pd.read_csv(TABLES / "conteos_por_archivo.csv")
    stats = pd.read_csv(TABLES / "estadistica_descriptiva_2025.csv")
    corr = pd.read_csv(TABLES / "correlaciones_pearson.csv", index_col=0)
    kmeans = pd.read_csv(TABLES / "seleccion_kmeans.csv")
    validation = pd.read_csv(TABLES / "comparacion_modelos_validacion.csv")
    test_metrics = pd.read_csv(TABLES / "metricas_test_2026.csv")
    final = pd.read_csv(TABLES / "comparacion_final_modelos.csv")
    shared = pd.read_csv(TABLES / "conteo_test_compartido.csv")
    education = pd.read_csv(TABLES / "errores_por_educacion_2026.csv")
    domain = pd.read_csv(TABLES / "errores_por_dominio_2026.csv")
    bands = pd.read_csv(TABLES / "errores_por_percentil_salarial_2026.csv")

    salary = stats.loc[stats["variable"] == "salario_mensual"].iloc[0]
    salary_corr = corr["salario_mensual"].drop("salario_mensual").abs().sort_values(ascending=False)
    best_k = kmeans.sort_values("silhouette", ascending=False).iloc[0]
    winner_val = validation[~validation["modelo"].str.startswith("Referencia")].sort_values("RMSE").iloc[0]
    winner_test, runner_test = words.best_model_names(test_metrics)
    metrics = test_metrics.set_index("modelo")
    total = int(counts["despues"].sum())
    n_test = int(metrics.loc[winner_test, "registros"])

    # ---- Resumen, retencion, descriptivos y validacion -------------------------------------
    lines = [
        r"\subsection{Hallazgos principales}",
        (
            f"Después de aplicar los criterios de elegibilidad se conservaron \\textbf{{{total:,}}} registros "
            f"de 2025. El salario medio fue \\textbf{{{money(float(salary['media']))}}} y la mediana "
            f"\\textbf{{{money(float(salary['mediana']))}}}; esta separación evidencia asimetría y la "
            "influencia de valores altos."
        ),
        (
            "La asociación lineal absoluta más alta con salario entre los predictores numéricos corresponde a "
            f"\\texttt{{{tex(salary_corr.index[0])}}} ($|r|={salary_corr.iloc[0]:.3f}$)."
        ),
        (
            f"KMeans seleccionó \\textbf{{{tex(best_k['escenario'])}}} con $K={int(best_k['k'])}$ y "
            f"silhouette {float(best_k['silhouette']):.4f}."
        ),
        (
            f"En validación temporal (2025T4), \\textbf{{{tex(winner_val['modelo'])}}} obtuvo el menor RMSE "
            f"({money(float(winner_val['RMSE']))})."
        ),
        (
            f"En la prueba final (2026T1, {n_test:,} registros), \\textbf{{{tex(winner_test)}}} obtuvo el mejor "
            f"desempeño: MAE {money(float(metrics.loc[winner_test, 'MAE']))}, RMSE "
            f"{money(float(metrics.loc[winner_test, 'RMSE']))} y $R^2={float(metrics.loc[winner_test, 'R2']):.4f}$."
        ),
    ]
    rows = [
        [tex(r["periodo_archivo"]), f"{int(r['antes']):,}", f"{int(r['despues']):,}",
         f"{int(r['excluidos']):,}", f"{float(r['retencion_pct']):.2f}\\%"]
        for _, r in counts.iterrows()
    ]
    lines.append(r"\subsection{Retención por período}")
    lines += table(["Período", "Antes", "Después", "Excluidos", "Retención"], rows, "lrrrr",
                   "Registros antes y después de los filtros comunes.")
    rows = [
        [tex(r["variable"]), f"{r['media']:,.2f}", f"{r['mediana']:,.2f}", f"{r['desviacion_estandar']:,.2f}",
         f"{r['p25']:,.2f}", f"{r['p75']:,.2f}", f"{r['p95']:,.2f}"]
        for _, r in stats.iterrows()
    ]
    lines.append(r"\subsection{Descriptivos principales}")
    lines += table(["Variable", "Media", "Mediana", "DE", "P25", "P75", "P95"], rows, "lrrrrrr",
                   "Estadísticas calculadas sobre todos los registros elegibles de 2025.", "scriptsize")
    rows = [
        [tex(r["modelo"]), f"{r['MAE']:,.2f}", f"{r['RMSE']:,.2f}", f"{r['R2']:.4f}"]
        for _, r in validation.iterrows()
    ]
    lines.append(r"\subsection{Comparación de modelos en 2025T4}")
    lines += table(["Modelo", "MAE (Q)", "RMSE (Q)", "$R^2$"], rows, "lrrr",
                   "Métricas no ponderadas sobre el mismo conjunto de validación.")
    (REPORTS / "generated_results.tex").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    # ---- Actividad 7: evaluacion final ----------------------------------------------------
    rows = [
        [tex(r["modelo"]), tex(r["configuracion"]), f"{int(r['registros']):,}", f"{r['MAE']:,.2f}",
         f"{r['RMSE']:,.2f}", f"{r['R2']:.4f}"]
        for _, r in test_metrics.sort_values("RMSE").iterrows()
    ]
    lines = table(["Modelo", "Configuración", "Registros", "MAE (Q)", "RMSE (Q)", "$R^2$"], rows, "llrrrr",
                  "Métricas en 2026T1, no ponderadas y calculadas sobre todo el conjunto de prueba.", "footnotesize")
    rows = [[tex(r["conjunto"]), f"{int(r['registros']):,}"] for _, r in shared.iterrows()]
    lines += table(["Conjunto", "Registros"], rows, "lr",
                   "Ambos modelos se evalúan sobre exactamente los mismos registros (misma clave y mismo conteo).")
    rows = [
        [tex(r["modelo"]), f"{r['RMSE_validacion']:,.2f}", f"{r['RMSE']:,.2f}",
         f"{r['cambio_RMSE_validacion_a_test']:+,.2f}", f"{r['mejora_RMSE_vs_referencia_pct']:.2f}\\%"]
        for _, r in final.iterrows()
    ]
    lines += table(["Modelo", "RMSE 2025T4", "RMSE 2026T1", "Cambio", "Mejora vs. referencia"], rows, "lrrrr",
                   "Estabilidad temporal del RMSE y mejora frente a la referencia de la media de 2025.", "footnotesize")
    ref = test_metrics[test_metrics["modelo"].str.startswith("Referencia")].iloc[0]
    improvement = 100 * (float(ref["RMSE"]) - float(metrics.loc[winner_test, "RMSE"])) / float(ref["RMSE"])
    val_names = validation[~validation["modelo"].str.startswith("Referencia")].sort_values("RMSE")
    same = "coincide" if val_names.iloc[0]["modelo"] == winner_test else "no coincide"
    interpretation = (
        f"En 2026T1, **{winner_test}** obtuvo el menor error (MAE {money(float(metrics.loc[winner_test, 'MAE']))}, "
        f"RMSE {money(float(metrics.loc[winner_test, 'RMSE']))}, R2 {float(metrics.loc[winner_test, 'R2']):.4f}); "
        f"el otro algoritmo, **{runner_test}**, obtuvo MAE {money(float(metrics.loc[runner_test, 'MAE']))}, "
        f"RMSE {money(float(metrics.loc[runner_test, 'RMSE']))} y R2 {float(metrics.loc[runner_test, 'R2']):.4f}. "
        f"La referencia (media de 2025) tiene RMSE {money(float(ref['RMSE']))} y R2 {float(ref['R2']):.4f}, y el "
        f"ganador reduce ese RMSE en {improvement:.2f}%. El ganador de validación {same} con el de prueba."
    )
    lines += ["", blocks(interpretation)]
    (REPORTS / "generated_test.tex").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    # ---- Actividad 8: errores ---------------------------------------------------------------
    lines = [r"\subsection{Errores por nivel educativo}"]
    lines += wide_group_table(education, "Nivel educativo",
                              "MAE y error medio (real $-$ predicho) por nivel educativo; RL = regresión lineal, RF = Random Forest.")
    lines += [blocks(words.group_findings(education, "nivel educativo") + "\n\n" +
                     words.salary_scale_sentence(education, "nivel educativo"))]
    lines.append(r"\subsection{Errores por dominio}")
    lines += wide_group_table(domain, "Dominio",
                              "MAE y error medio (real $-$ predicho) por dominio; RL = regresión lineal, RF = Random Forest.")
    lines += [blocks(words.group_findings(domain, "dominio") + "\n\n" +
                     words.salary_scale_sentence(domain, "dominio"))]
    lines.append(r"\subsection{Errores según el percentil del salario real}")
    lr = bands[bands["modelo"] == "Regresión lineal"].set_index("banda")
    rf = bands[bands["modelo"] == "Random Forest"].set_index("banda")
    rows = []
    for band in lr.index:
        rows.append(
            [tex(band), f"{int(lr.loc[band, 'n']):,}", f"{lr.loc[band, 'salario_promedio']:,.0f}",
             f"{lr.loc[band, 'MAE']:,.0f}", f"{lr.loc[band, 'error_medio']:+,.0f}",
             f"{rf.loc[band, 'MAE']:,.0f}", f"{rf.loc[band, 'error_medio']:+,.0f}",
             f"{100 * rf.loc[band, 'prop_subestimado']:.1f}\\%"]
        )
    lines += table(["Banda", "n", "Salario prom.", "MAE RL", "Err. RL", "MAE RF", "Err. RF", "RF subestima"],
                   rows, "lrrrrrrr",
                   "Error por banda del salario real de 2026T1 (todos los registros de prueba); "
                   "error = real $-$ predicho.", "footnotesize")
    lines += [blocks(words.band_findings(bands) + "\n\n**Salarios bajos, medios y altos.** " +
                     words.band_context(test_metrics, bands))]
    lines.append(r"\subsection{Respuestas a las preguntas de la actividad}")
    lines += [blocks(words.question_answers(test_metrics, education, domain, bands))]
    residuals = pd.read_csv(TABLES / "resumen_residuos_2026.csv").set_index("modelo")
    global_lines = "\n".join(
        f"- **{model}:** error medio global {float(row['error_medio']):+,.2f} Q "
        f"({words.direction(float(row['error_medio']))} en promedio); subestima al "
        f"{100 * float(row['prop_subestimado']):.1f}% de los registros."
        for model, row in residuals.iterrows()
    )
    reading = (TABLES / "interpretacion_graficos_2026.txt").read_text(encoding="utf-8")
    plot_text = blocks(
        "**Resultados globales (todo el test).**\n\n" + global_lines + "\n\n"
        "**Lo que muestran las gráficas (muestra común de 5,000 registros).**\n\n" + reading
    )
    (REPORTS / "generated_plots.tex").write_text(
        plot_text.rstrip() + "\n", encoding="utf-8"
    )
    (REPORTS / "generated_errors.tex").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    conclusion_text = blocks(words.final_discussion(test_metrics, education, domain, bands))
    (REPORTS / "generated_conclusions.tex").write_text(
        conclusion_text.rstrip() + "\n", encoding="utf-8"
    )
    print("Fragmentos generados en", REPORTS)


if __name__ == "__main__":
    build()
