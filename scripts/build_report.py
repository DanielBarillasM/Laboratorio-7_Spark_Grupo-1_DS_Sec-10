"""Generate LaTeX result fragments from executed Spark artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
DESTINATION = ROOT / "reports" / "generated_results.tex"


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


def build() -> None:
    required = [
        "conteos_por_archivo.csv",
        "estadistica_descriptiva_2025.csv",
        "correlaciones_pearson.csv",
        "seleccion_kmeans.csv",
        "perfiles_clusters.csv",
        "comparacion_modelos_validacion.csv",
    ]
    missing = [name for name in required if not (TABLES / name).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan resultados del notebook: {missing}")

    counts = pd.read_csv(TABLES / "conteos_por_archivo.csv")
    stats = pd.read_csv(TABLES / "estadistica_descriptiva_2025.csv")
    corr = pd.read_csv(TABLES / "correlaciones_pearson.csv", index_col=0)
    kmeans = pd.read_csv(TABLES / "seleccion_kmeans.csv")
    profiles = pd.read_csv(TABLES / "perfiles_clusters.csv")
    models = pd.read_csv(TABLES / "comparacion_modelos_validacion.csv")

    salary = stats.loc[stats["variable"] == "salario_mensual"].iloc[0]
    salary_corr = corr["salario_mensual"].drop("salario_mensual").abs().sort_values(ascending=False)
    best_k = kmeans.sort_values("silhouette", ascending=False).iloc[0]
    winner = models.sort_values("RMSE").iloc[0]
    total = int(counts["despues"].sum())

    lines = [
        r"\subsection{Hallazgos principales}",
        (
            f"Después de aplicar los criterios de elegibilidad se conservaron "
            rf"\textbf{{{total:,}}} registros de 2025. El salario medio fue "
            rf"\textbf{{{money(float(salary['media']))}}} y la mediana "
            rf"\textbf{{{money(float(salary['mediana']))}}}; esta separación evidencia "
            "asimetría y la influencia de valores altos."
        ),
        (
            f"La asociación lineal absoluta más alta con salario entre los predictores "
            rf"numéricos corresponde a \texttt{{{tex(salary_corr.index[0])}}} "
            f"($|r|={salary_corr.iloc[0]:.3f}$)."
        ),
        (
            rf"KMeans seleccionó \textbf{{{tex(best_k['escenario'])}}} con "
            f"$K={int(best_k['k'])}$ y silhouette {float(best_k['silhouette']):.4f}."
        ),
        (
            rf"En validación temporal, \textbf{{{tex(winner['modelo'])}}} obtuvo el menor "
            f"RMSE ({money(float(winner['RMSE']))}), con MAE {money(float(winner['MAE']))} "
            f"y $R^2={float(winner['R2']):.4f}$. La prueba 2026 permanece reservada."
        ),
        r"\subsection{Retención por período}",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Período & Antes & Después & Excluidos & Retención \\ \midrule",
    ]
    for _, row in counts.iterrows():
        lines.append(
            f"{tex(row['periodo_archivo'])} & {int(row['antes']):,} & {int(row['despues']):,} & "
            rf"{int(row['excluidos']):,} & {float(row['retencion_pct']):.2f}\% \\"
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{Registros antes y después de los filtros comunes.}",
            r"\end{table}",
            r"\subsection{Descriptivos principales}",
            r"\begin{table}[htbp]\centering\scriptsize",
            r"\begin{tabular}{lrrrrrr}\toprule",
            r"Variable & Media & Mediana & DE & P25 & P75 & P95 \\ \midrule",
        ]
    )
    for _, row in stats.iterrows():
        lines.append(
            f"{tex(row['variable'])} & {float(row['media']):,.2f} & {float(row['mediana']):,.2f} & "
            f"{float(row['desviacion_estandar']):,.2f} & {float(row['p25']):,.2f} & "
            rf"{float(row['p75']):,.2f} & {float(row['p95']):,.2f} \\"
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{Estadísticas calculadas sobre todos los registros elegibles de 2025.}",
            r"\end{table}",
            r"\subsection{Comparación de modelos en 2025T4}",
            r"\begin{table}[htbp]\centering\small",
            r"\begin{tabular}{lrrr}\toprule",
            r"Modelo & MAE (Q) & RMSE (Q) & $R^2$ \\ \midrule",
        ]
    )
    for _, row in models.iterrows():
        lines.append(
            rf"{tex(row['modelo'])} & {float(row['MAE']):,.2f} & {float(row['RMSE']):,.2f} & {float(row['R2']):.4f} \\"
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}",
            r"\caption{Métricas no ponderadas sobre el mismo conjunto de validación.}",
            r"\end{table}",
        ]
    )
    DESTINATION.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Fragmento generado: {DESTINATION}")


if __name__ == "__main__":
    build()
