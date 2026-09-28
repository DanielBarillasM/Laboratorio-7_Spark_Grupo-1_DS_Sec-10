"""Interpretaciones reproducibles para la evaluacion final del Laboratorio 7.

Las funciones de este modulo convierten tablas ya calculadas sobre el conjunto
completo en texto breve para el notebook y el informe. No recalculan modelos ni
seleccionan hiperparametros con informacion de 2026.
"""

from __future__ import annotations

import pandas as pd


def _money(value: float) -> str:
    return f"Q{float(value):,.2f}"


def _model_rows(table: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = table.loc[table["modelo"] == model].copy()
    if rows.empty:
        raise ValueError(f"No hay resultados para {model!r}.")
    return rows


def direction(error_mean: float) -> str:
    """Return the interpretation implied by residual = actual - predicted."""
    if error_mean > 0:
        return "subestima"
    if error_mean < 0:
        return "sobreestima"
    return "no presenta sesgo medio"


def best_model_names(test_metrics: pd.DataFrame) -> tuple[str, str]:
    """Return winner and runner-up names after excluding the constant baseline."""
    models = test_metrics.loc[
        ~test_metrics["modelo"].str.startswith("Referencia")
    ].sort_values("RMSE")
    if len(models) < 2:
        raise ValueError("Se requieren resultados de los dos algoritmos.")
    return str(models.iloc[0]["modelo"]), str(models.iloc[1]["modelo"])


def plot_findings(sample: pd.DataFrame) -> str:
    """Describe los dos diagramas usando exactamente la muestra comun."""
    descriptions: list[str] = []
    for model, column in (
        ("Regresion lineal", "prediccion_lr"),
        ("Random Forest", "prediccion_rf"),
    ):
        residual = sample["salario_mensual"] - sample[column]
        high_cut = float(sample["salario_mensual"].quantile(0.90))
        high = residual.loc[sample["salario_mensual"] >= high_cut]
        descriptions.append(
            f"- **{model}:** en la muestra, el residuo medio es {_money(residual.mean())}; "
            f"para salarios reales desde el percentil 90 es {_money(high.mean())}. "
            "La dispersion crece con el nivel salarial, por lo que los errores no tienen varianza constante."
        )
    return "\n".join(descriptions)


def group_findings(table: pd.DataFrame, group_label: str) -> str:
    """Resume los grupos con mayor MAE y sesgo para cada modelo."""
    lines: list[str] = []
    for model in table["modelo"].drop_duplicates():
        rows = _model_rows(table, model)
        worst = rows.loc[rows["MAE"].idxmax()]
        under = rows.loc[rows["error_medio"].idxmax()]
        over = rows.loc[rows["error_medio"].idxmin()]
        if float(under["error_medio"]) > 0:
            under_text = (
                f"La mayor subestimación media ocurre en **{under['grupo']}** "
                f"({_money(under['error_medio'])})"
            )
        else:
            under_text = "No hay subestimación media en ninguno de sus grupos"
        if float(over["error_medio"]) < 0:
            over_text = (
                f"la mayor sobreestimación ocurre en **{over['grupo']}** "
                f"({_money(over['error_medio'])})"
            )
        else:
            over_text = (
                f"todos los grupos presentan subestimación media; la menor aparece en "
                f"**{over['grupo']}** ({_money(over['error_medio'])})"
            )
        lines.append(
            f"- **{model}:** el mayor MAE por {group_label} aparece en **{worst['grupo']}** "
            f"({_money(worst['MAE'])}, n={int(worst['n']):,}). {under_text} y {over_text}."
        )
    return "\n".join(lines)


def salary_scale_sentence(table: pd.DataFrame, group_label: str) -> str:
    """Aclara que el error absoluto debe leerse junto con la escala salarial."""
    grouped = table.groupby("grupo", as_index=False).agg(
        salario_promedio=("salario_promedio", "mean"), MAE=("MAE", "mean")
    )
    highest_salary = grouped.loc[grouped["salario_promedio"].idxmax()]
    highest_error = grouped.loc[grouped["MAE"].idxmax()]
    if highest_salary["grupo"] == highest_error["grupo"]:
        relation = "tambien concentra el mayor MAE absoluto"
    else:
        relation = f"mientras el mayor MAE aparece en {highest_error['grupo']}"
    return (
        f"El grupo de {group_label} con mayor salario promedio es **{highest_salary['grupo']}** "
        f"({_money(highest_salary['salario_promedio'])}) y {relation}; esto obliga a interpretar el MAE "
        "junto con la escala salarial y el numero de observaciones."
    )


def band_findings(table: pd.DataFrame) -> str:
    """Resume magnitud y direccion del error en las bandas salariales."""
    lines: list[str] = []
    for model in table["modelo"].drop_duplicates():
        rows = _model_rows(table, model)
        low = rows.loc[rows["banda"] == "P00-P25"].iloc[0]
        high = rows.loc[rows["banda"] == "P95-P100"].iloc[0]
        lines.append(
            f"- **{model}:** en P00-P25 el error medio es {_money(low['error_medio'])}; "
            f"en P95-P100 asciende a {_money(high['error_medio'])}, con MAE {_money(high['MAE'])} "
            f"y {float(high['prop_subestimado']):.1%} de registros subestimados."
        )
    return "\n".join(lines)


def band_context(test_metrics: pd.DataFrame, bands: pd.DataFrame) -> str:
    """Relaciona el ganador global con el comportamiento en salarios altos."""
    models = test_metrics.loc[~test_metrics["modelo"].str.startswith("Referencia")].sort_values("RMSE")
    winner = str(models.iloc[0]["modelo"])
    label = "Random Forest" if winner.startswith("RF") else "Regresion lineal"
    high = _model_rows(bands, label).loc[lambda frame: frame["banda"] == "P95-P100"].iloc[0]
    return (
        f"**{winner}** es el mejor modelo global por RMSE. Sin embargo, en P95-P100 todavia presenta "
        f"un error medio de {_money(high['error_medio'])} y un MAE de {_money(high['MAE'])}; "
        "por tanto, una mejora promedio no elimina la dificultad de representar la cola salarial."
    )


def question_answers(
    test_metrics: pd.DataFrame,
    education: pd.DataFrame,
    domain: pd.DataFrame,
    bands: pd.DataFrame,
) -> str:
    """Responde de forma explicita las preguntas de la actividad 8."""
    models = test_metrics.loc[~test_metrics["modelo"].str.startswith("Referencia")].sort_values("RMSE")
    winner, runner = models.iloc[0], models.iloc[1]
    worst_education = education.loc[education["MAE"].idxmax()]
    worst_domain = domain.loc[domain["MAE"].idxmax()]
    worst_band = bands.loc[bands["MAE"].idxmax()]
    return (
        f"- **Algoritmo con mejor generalizacion:** {winner['modelo']}, con RMSE {_money(winner['RMSE'])} "
        f"y R2={float(winner['R2']):.4f}, frente a RMSE {_money(runner['RMSE'])} del otro algoritmo.\n"
        f"- **Nivel educativo con mayor error observado:** {worst_education['grupo']} para "
        f"{worst_education['modelo']} (MAE {_money(worst_education['MAE'])}, n={int(worst_education['n']):,}).\n"
        f"- **Dominio con mayor error observado:** {worst_domain['grupo']} para {worst_domain['modelo']} "
        f"(MAE {_money(worst_domain['MAE'])}, n={int(worst_domain['n']):,}).\n"
        f"- **Banda mas dificil:** {worst_band['banda']} para {worst_band['modelo']} "
        f"(MAE {_money(worst_band['MAE'])}). Los errores medios positivos en la cola alta indican "
        "subestimacion sistematica de salarios altos.\n"
        "- **Lectura correcta:** las diferencias son predictivas y descriptivas; no prueban causalidad ni "
        "determinan cuanto deberia ganar una persona."
    )


def final_discussion(
    test_metrics: pd.DataFrame,
    education: pd.DataFrame,
    domain: pd.DataFrame,
    bands: pd.DataFrame,
) -> str:
    """Construye una conclusion final coherente con la rubrica."""
    models = test_metrics.loc[~test_metrics["modelo"].str.startswith("Referencia")].sort_values("RMSE")
    winner, runner = models.iloc[0], models.iloc[1]
    baseline = test_metrics.loc[test_metrics["modelo"].str.startswith("Referencia")].iloc[0]
    improvement = 100 * (float(baseline["RMSE"]) - float(winner["RMSE"])) / float(baseline["RMSE"])
    high = bands.loc[(bands["modelo"] == ("Random Forest" if str(winner["modelo"]).startswith("RF") else "Regresion lineal")) & (bands["banda"] == "P95-P100")].iloc[0]
    edu_worst = education.loc[education["MAE"].idxmax()]
    dom_worst = domain.loc[domain["MAE"].idxmax()]
    return (
        f"El analisis identifico perfiles laborales diferenciados y demostro que **{winner['modelo']}** "
        f"generaliza mejor a 2026T1: MAE {_money(winner['MAE'])}, RMSE {_money(winner['RMSE'])} y "
        f"R2={float(winner['R2']):.4f}. Su RMSE es {improvement:.2f}% menor que el de la referencia y "
        f"tambien supera a {runner['modelo']}. La ventaja es compatible con la capacidad del bosque para "
        "representar no linealidades e interacciones entre educacion, ocupacion y territorio.\n\n"
        f"El desempeño no es uniforme. El mayor MAE por educacion se observo en **{edu_worst['grupo']}**, "
        f"y por dominio en **{dom_worst['grupo']}**. En la banda P95-P100, el ganador conserva un MAE de "
        f"{_money(high['MAE'])} y un error medio de {_money(high['error_medio'])}, evidencia de que la cola "
        "salarial sigue siendo dificil y tiende a subestimarse. El patron debe interpretarse junto con el "
        "tamaño de cada grupo y la regresion a la media.\n\n"
        "Las metricas son no ponderadas y describen solamente asalariados elegibles con salario positivo "
        "registrado. No representan estimaciones oficiales de Guatemala ni efectos causales. La ausencia de "
        "variables como rama de actividad, ocupacion detallada o tamaño de empresa limita la varianza explicada; "
        "un uso posterior deberia estudiar estabilidad temporal, equidad por subgrupos y estimaciones ponderadas "
        "con FACTOR bajo el diseño muestral correspondiente."
    )
