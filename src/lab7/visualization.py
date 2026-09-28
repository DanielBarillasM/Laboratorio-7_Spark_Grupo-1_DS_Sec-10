"""Visualizaciones publicables a partir de agregados o muestras acotadas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR


BLUE = "#2563EB"
CYAN = "#06B6D4"
TEAL = "#0F766E"
ORANGE = "#F59E0B"
INK = "#172033"
GRID = "#D9E2F1"


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, filename: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def plot_categorical_distributions(distributions: pd.DataFrame) -> Path:
    apply_theme()
    variables = ["categoria_ocupacional", "nivel_educativo", "dominio"]
    titles = ["Categoria ocupacional", "Nivel educativo", "Dominio"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    for ax, variable, title in zip(axes, variables, titles):
        data = distributions[distributions["variable"] == variable].sort_values("n")
        ax.barh(data["categoria"], data["porcentaje"], color=BLUE, alpha=0.88)
        ax.set_title(title)
        ax.set_xlabel("Porcentaje de registros (%)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Composicion de la poblacion analitica ENEIC 2025", fontsize=16, weight="bold")
    return _save(fig, "distribuciones_categoricas.png")


def plot_salary_distribution(sample: pd.DataFrame) -> Path:
    apply_theme()
    values = sample["salario_mensual"].dropna().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].hist(values, bins=45, color=BLUE, alpha=0.85, edgecolor="white")
    axes[0].set_title("Escala original")
    axes[0].set_xlabel("Salario mensual (Q)")
    axes[0].set_ylabel("Frecuencia en muestra")
    axes[0].grid(axis="y")
    axes[1].hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 45), color=TEAL, alpha=0.85, edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_title("Escala logaritmica solo para visualizar")
    axes[1].set_xlabel("Salario mensual (Q, escala log)")
    axes[1].grid(axis="y")
    fig.suptitle("Distribucion del salario mensual - muestra de hasta 5,000 registros", fontsize=15, weight="bold")
    return _save(fig, "distribucion_salario.png")


def plot_group_medians(grouped: pd.DataFrame) -> Path:
    apply_theme()
    variables = ["nivel_educativo", "categoria_ocupacional"]
    titles = ["Por nivel educativo", "Por categoria ocupacional"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, variable, title in zip(axes, variables, titles):
        data = grouped[grouped["variable"] == variable].sort_values("salario_mediano")
        ax.barh(data["categoria"], data["salario_mediano"], color=CYAN, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Salario mediano (Q)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Salario mediano por grupos", fontsize=16, weight="bold")
    return _save(fig, "salario_mediano_grupos.png")


def plot_quarterly_summary(summary: pd.DataFrame) -> Path:
    apply_theme()
    fig, ax1 = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax1.bar(summary["periodo_archivo"], summary["n"], color=BLUE, alpha=0.78, label="Registros")
    ax1.set_ylabel("Registros elegibles")
    ax1.set_xlabel("Periodo del archivo")
    ax1.grid(axis="y")
    ax2 = ax1.twinx()
    ax2.plot(summary["periodo_archivo"], summary["salario_mediano"], color=ORANGE, marker="o", linewidth=2.5, label="Salario mediano")
    ax2.set_ylabel("Salario mediano (Q)")
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, loc="upper left")
    ax1.set_title("Tamano analitico y salario mediano por trimestre")
    return _save(fig, "evolucion_trimestral.png")


def plot_correlation_heatmap(correlation: pd.DataFrame) -> Path:
    apply_theme()
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = ax.imshow(correlation.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if abs(value) > 0.55 else INK, weight="bold")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Correlacion de Pearson")
    ax.set_title("Matriz de correlaciones - registros elegibles 2025")
    return _save(fig, "correlaciones_pearson.png")


def plot_kmeans_selection(results: pd.DataFrame) -> Path:
    apply_theme()
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    for scenario, group in results.groupby("escenario"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["silhouette"], marker="o", linewidth=2.4, label=scenario)
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel("Numero de clusters (K)")
    ax.set_ylabel("Silhouette")
    ax.set_title("Seleccion de K y sensibilidad a incluir salario")
    ax.grid(True)
    ax.legend()
    return _save(fig, "seleccion_kmeans.png")


def plot_model_comparison(comparison: pd.DataFrame) -> Path:
    apply_theme()
    ordered = comparison.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    colors = [TEAL if i == 0 else BLUE for i in range(len(ordered))]
    ax.barh(ordered["modelo"], ordered["RMSE"], color=colors, alpha=0.9)
    ax.set_xlabel("RMSE de validacion (Q)")
    ax.set_title("Comparacion de modelos en 2025T4")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return _save(fig, "comparacion_modelos_validacion.png")


# ---------------------------------------------------------------------------
# Actividad 8: graficos de errores (muestra comun de hasta 5,000 registros)
# ---------------------------------------------------------------------------

MODEL_STYLE = {
    "Regresión lineal": ("prediccion_lr", BLUE),
    "Random Forest": ("prediccion_rf", TEAL),
}


def plot_actual_vs_predicted(sample: pd.DataFrame) -> Path:
    """Salario real (eje y) frente a salario predicho (eje x) con la linea y = x."""
    apply_theme()
    limit = float(max(sample["salario_mensual"].max(), *(sample[c].max() for c, _ in MODEL_STYLE.values())))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (model, (column, color)) in zip(axes, MODEL_STYLE.items()):
        ax.scatter(sample[column], sample["salario_mensual"], s=9, alpha=0.32, color=color, edgecolors="none")
        ax.plot([0, limit], [0, limit], color=ORANGE, linewidth=2, label="y = x (prediccion perfecta)")
        ax.set_title(model)
        ax.set_xlabel("Salario predicho (Q)")
        ax.set_xlim(0, limit * 1.02)
        ax.set_ylim(0, limit * 1.02)
        ax.grid(True)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Salario real (Q)")
    fig.suptitle(f"Salario real vs. predicho en 2026T1 - misma muestra de {len(sample):,} registros", fontsize=15, weight="bold")
    return _save(fig, "real_vs_predicho_2026.png")


def plot_residuals(sample: pd.DataFrame) -> Path:
    """Residuo (real - predicho) frente al salario predicho con linea horizontal en cero."""
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True, constrained_layout=True)
    for ax, (model, (column, color)) in zip(axes, MODEL_STYLE.items()):
        residual = sample["salario_mensual"] - sample[column]
        ax.scatter(sample[column], residual, s=9, alpha=0.32, color=color, edgecolors="none")
        ax.axhline(0, color=ORANGE, linewidth=2, label="residuo = 0")
        ax.set_title(model)
        ax.set_xlabel("Salario predicho (Q)")
        ax.grid(True)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Residuo = real - predicho (Q)\n(positivo: subestima; negativo: sobreestima)")
    fig.suptitle(f"Residuos vs. salario predicho en 2026T1 - misma muestra de {len(sample):,} registros", fontsize=15, weight="bold")
    return _save(fig, "residuos_vs_predicho_2026.png")


def _grouped_error_axes(table: pd.DataFrame, group_order: list[str], title: str, filename: str, xlabel: str) -> Path:
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    positions = np.arange(len(group_order))
    width = 0.38
    for offset, (model, (_, color)) in zip((-width / 2, width / 2), MODEL_STYLE.items()):
        data = table[table["modelo"] == model].set_index("grupo").reindex(group_order)
        axes[0].bar(positions + offset, data["MAE"], width, color=color, alpha=0.9, label=model)
        axes[1].bar(positions + offset, data["error_medio"], width, color=color, alpha=0.9, label=model)
    axes[0].set_title("MAE por grupo")
    axes[0].set_ylabel("MAE (Q)")
    axes[1].set_title("Error medio (real - predicho)")
    axes[1].set_ylabel("Error medio (Q): >0 subestima, <0 sobreestima")
    axes[1].axhline(0, color=INK, linewidth=1)
    counts = table[table["modelo"] == "Random Forest"].set_index("grupo").reindex(group_order)["n"]
    labels = [f"{name}\n(n={int(counts[name]):,})" if pd.notna(counts[name]) else name for name in group_order]
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    fig.suptitle(title, fontsize=15, weight="bold")
    return _save(fig, filename)


def plot_errors_by_education(table: pd.DataFrame) -> Path:
    order = [g for g in table["grupo"].drop_duplicates()]
    return _grouped_error_axes(table, order, "Errores por nivel educativo en 2026T1 (todos los registros de prueba)",
                               "errores_por_educacion_2026.png", "Nivel educativo")


def plot_errors_by_domain(table: pd.DataFrame) -> Path:
    order = [g for g in table["grupo"].drop_duplicates()]
    return _grouped_error_axes(table, order, "Errores por dominio en 2026T1 (todos los registros de prueba)",
                               "errores_por_dominio_2026.png", "Dominio")


def plot_errors_by_salary_band(table: pd.DataFrame) -> Path:
    apply_theme()
    bands = list(table["banda"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    positions = np.arange(len(bands))
    width = 0.38
    for offset, (model, (_, color)) in zip((-width / 2, width / 2), MODEL_STYLE.items()):
        data = table[table["modelo"] == model].set_index("banda").reindex(bands)
        axes[0].bar(positions + offset, data["MAE"], width, color=color, alpha=0.9, label=model)
        axes[1].bar(positions + offset, data["error_medio"], width, color=color, alpha=0.9, label=model)
    axes[0].set_title("MAE por banda del salario real")
    axes[0].set_ylabel("MAE (Q)")
    axes[1].set_title("Error medio (real - predicho) por banda")
    axes[1].set_ylabel("Error medio (Q): >0 subestima, <0 sobreestima")
    axes[1].axhline(0, color=INK, linewidth=1)
    sizes = table[table["modelo"] == "Random Forest"].set_index("banda").reindex(bands)["n"]
    labels = [f"{name}\n(n={int(sizes[name]):,})" if pd.notna(sizes[name]) else name for name in bands]
    for ax in axes:
        ax.set_xticks(positions, labels, rotation=30, ha="right")
        ax.set_xlabel("Banda de percentil del salario real 2026T1")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend()
    fig.suptitle("Error segun el percentil del salario real (todos los registros de prueba)", fontsize=15, weight="bold")
    return _save(fig, "errores_por_percentil_salarial_2026.png")
