"""
ui/page_sampling.py
===================
Módulo 1: tamaño muestral, muestreo estratificado y sistemático.
Entradas en la barra lateral; resultados en el panel principal.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import sampling as smp
from core.common import fmt
from ui.components import (SAMPLING_TOOLS, confidence_input, metric_row, render_notes,
                           render_steps)
from ui.help_texts import HELP
from viz import charts

WIDTH = "stretch"


def render() -> None:
    sb = st.sidebar
    sb.header("Parámetros de muestreo")
    tool = sb.radio("Herramienta", SAMPLING_TOOLS, key="sampling_tool")

    st.title("Técnicas y cálculo de muestreo")
    if tool == SAMPLING_TOOLS[0]:
        _sample_size_page("mean")
    elif tool == SAMPLING_TOOLS[1]:
        _sample_size_page("prop")
    elif tool == SAMPLING_TOOLS[2]:
        _stratified_page()
    else:
        _systematic_page()


# ------------------------------------------------------------------ #
# Tamaño muestral
# ------------------------------------------------------------------ #
def _sample_size_page(kind: str) -> None:
    sb = st.sidebar
    conf = confidence_input(f"ss_{kind}")
    pop = sb.radio("Tipo de población", ["Infinita (N desconocido)", "Finita (N conocido)"],
                   key=f"ss_{kind}_pop", help=HELP["population"])
    N = None
    if pop.startswith("Finita"):
        N = int(sb.number_input("Tamaño de la población N", min_value=2, value=1000, step=1,
                                key=f"ss_{kind}_N", help=HELP["N"]))

    if kind == "mean":
        st.subheader("Tamaño de muestra para estimar la media μ")
        sigma = sb.number_input("Desviación estándar σ", min_value=0.0001, value=10.0,
                                step=0.5, format="%.4f", key="ss_sigma", help=HELP["sigma"])
        e = sb.number_input("Margen de error e (unidades de la variable)", min_value=0.0001,
                            value=2.0, step=0.1, format="%.4f", key="ss_e_mean", help=HELP["e_mean"])
        calc = lambda ee: smp.sample_size_mean(sigma, ee, conf, N)  # noqa: E731
        e_label = "Margen de error e"
    else:
        st.subheader("Tamaño de muestra para estimar la proporción P")
        p = sb.number_input("Proporción esperada p", min_value=0.001, max_value=0.999,
                            value=0.5, step=0.05, format="%.3f", key="ss_p", help=HELP["p_prop"])
        e = sb.number_input("Margen de error e (en tanto por uno)", min_value=0.001,
                            max_value=0.5, value=0.05, step=0.005, format="%.3f",
                            key="ss_e_prop", help=HELP["e_prop"])
        calc = lambda ee: smp.sample_size_proportion(p, ee, conf, N)  # noqa: E731
        e_label = "Margen de error e (proporción)"

    try:
        res = calc(e)
    except ValueError as err:
        st.error(str(err))
        return

    st.session_state["last_n"] = res.n  # disponible para el muestreo estratificado
    items = [("Tamaño de muestra n", str(res.n)),
             ("Z crítico", fmt(res.z)),
             ("Valor exacto", fmt(res.n_raw, 3))]
    if res.finite:
        items.append(("% de la población", f"{100 * res.n / N:.2f} %"))
    metric_row(items)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        render_steps(res.steps)
        render_notes(res.notes)
    with col2:
        upper = 0.5 if kind == "prop" else None
        fig = charts.plot_sample_size_curve(
            lambda ee: calc(min(ee, upper) if upper else ee).n, e, res.n, e_label)
        st.plotly_chart(fig, width=WIDTH)
        st.info(f"Interpretación: con una muestra de **{res.n}** elementos, la estimación tendrá "
                f"un margen de error máximo de **±{fmt(e)}** con un **{fmt(conf * 100, 2)} %** de "
                "confianza.")


# ------------------------------------------------------------------ #
# Estratificado
# ------------------------------------------------------------------ #
def _stratified_page() -> None:
    sb = st.sidebar
    st.subheader("Muestreo estratificado · afijación proporcional")

    last_n = st.session_state.get("last_n")
    options = ["Ingresar manualmente"] + ([f"Usar el último n calculado ({last_n})"] if last_n else [])
    src = sb.radio("Origen del tamaño de muestra n", options, key="strat_src")
    if src.startswith("Usar"):
        n = int(last_n)
    else:
        n = int(sb.number_input("Tamaño total de la muestra n", min_value=1, value=100,
                                step=1, key="strat_n", help=HELP["n_strat"]))

    sb.markdown("**Estratos**", help=HELP["strata"])
    default = pd.DataFrame({"Estrato": ["Estrato 1", "Estrato 2", "Estrato 3"],
                            "N_i": [400, 350, 250]})
    table = sb.data_editor(
        default, num_rows="dynamic", key="strata_editor", hide_index=True,
        column_config={
            "Estrato": st.column_config.TextColumn("Estrato", required=True),
            "N_i": st.column_config.NumberColumn("N_i", min_value=1, step=1, required=True),
        },
    )
    table = table.dropna()
    if len(table) < 2:
        st.info("Agrega al menos dos estratos con su tamaño N_i en la barra lateral.")
        return
    try:
        res = smp.stratified_proportional(n, table["Estrato"].tolist(), table["N_i"].tolist())
    except ValueError as err:
        st.error(str(err))
        return

    metric_row([("Población N", str(res.N)), ("Muestra n", str(res.n)),
                ("Fracción de muestreo n/N", f"{100 * res.n / res.N:.2f} %"),
                ("Estratos", str(len(res.rows)))])
    col1, col2 = st.columns([1, 1])
    with col1:
        df = pd.DataFrame(res.rows)
        st.dataframe(df, hide_index=True, width=WIDTH)
        st.download_button("Descargar asignación (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "afijacion_proporcional.csv", "text/csv")
    with col2:
        st.plotly_chart(charts.plot_stratified(res.rows), width=WIDTH)
    render_steps(res.steps)


# ------------------------------------------------------------------ #
# Sistemático
# ------------------------------------------------------------------ #
def _systematic_page() -> None:
    sb = st.sidebar
    st.subheader("Muestreo sistemático")
    N = int(sb.number_input("Tamaño de la población N", min_value=1, value=1000, step=1,
                            key="sys_N", help=HELP["N"]))
    n_default = min(int(st.session_state.get("last_n") or 50), N)
    n = int(sb.number_input("Tamaño de la muestra n", min_value=1, max_value=N,
                            value=n_default, step=1, key="sys_n"))
    mode = sb.radio("Arranque A", ["Aleatorio", "Manual"], key="sys_mode", horizontal=True,
                    help=HELP["start"])
    start, seed = None, None
    k = max(1, N // max(n, 1))
    if mode == "Manual":
        start = int(sb.number_input(f"Arranque A (entre 1 y {k})", min_value=1, max_value=k,
                                    value=1, step=1, key="sys_start"))
    else:
        if sb.checkbox("Usar semilla (reproducible)", value=True, key="sys_use_seed"):
            seed = int(sb.number_input("Semilla", min_value=0, value=2026, step=1,
                                       key="sys_seed", help=HELP["seed"]))
        else:
            sb.button("Sortear de nuevo", key="sys_reroll")  # fuerza una nueva ejecución

    try:
        res = smp.systematic_sample(N, n, start=start, seed=seed)
    except ValueError as err:
        st.error(str(err))
        return

    metric_row([("Intervalo k", str(res.k)), ("Arranque A", str(res.start)),
                ("Primer ID", str(res.ids[0])), ("Último ID", str(res.ids[-1]))])
    st.plotly_chart(charts.plot_systematic(res.N, res.ids, res.k), width=WIDTH)

    col1, col2 = st.columns([1, 1])
    with col1:
        render_steps(res.steps)
        render_notes(res.notes)
    with col2:
        st.markdown("#### Secuencia completa de IDs seleccionados")
        df = pd.DataFrame({"Orden j": range(1, len(res.ids) + 1), "ID seleccionado": res.ids})
        st.dataframe(df, hide_index=True, height=360, width=WIDTH)
        st.download_button("Descargar IDs (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "muestra_sistematica.csv", "text/csv")
