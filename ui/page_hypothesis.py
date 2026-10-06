"""
ui/page_hypothesis.py
=====================
Módulo 2: pruebas de hipótesis paramétricas.

Flujo: barra lateral (configuración y datos) → núcleo `core.hypothesis`
→ `viz.charts` → panel principal (hipótesis, métricas, gráfico, veredicto, pasos).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from core import data_loader as dl
from core import hypothesis as hyp
from core.common import TAIL_LABELS, fmt, fmt_p
from ui.components import alpha_input, metric_row, render_notes, render_steps, verdict_box
from ui.help_texts import HELP
from viz import charts

SUMMARY, FILE = "Datos resumidos", "Archivo CSV / Excel"
WIDTH = "stretch"


@st.cache_data(show_spinner=False)
def _load(content: bytes, name: str) -> pd.DataFrame:
    return dl.read_table(content, name)


# ------------------------------------------------------------------ #
# Recolectores de datos de una muestra (resumen o archivo)
# ------------------------------------------------------------------ #
def _mean_sample(df: Optional[pd.DataFrame], idx: str, defaults: tuple = (0.0, 1.0, 30),
                 sd_label: str = "Desviación estándar muestral s", col_index: int = 0) -> dict:
    sb = st.sidebar
    sfx = f" {idx}" if idx else ""
    if df is None:
        xbar = sb.number_input(f"Media muestral x̄{sfx}", value=float(defaults[0]),
                               format="%.4f", key=f"h_xbar{idx}", help=HELP["xbar"])
        s = sb.number_input(f"{sd_label}{sfx}", min_value=1e-6, value=float(defaults[1]),
                            format="%.4f", key=f"h_s{idx}",
                            help=HELP["sigma_pop"] if "σ" in sd_label else HELP["s"])
        n = int(sb.number_input(f"Tamaño de muestra n{sfx}", min_value=2, value=int(defaults[2]),
                                step=1, key=f"h_n{idx}", help=HELP["n"]))
        return {"mean": xbar, "std": s, "n": n}
    cols = dl.numeric_columns(df)
    if not cols:
        raise ValueError("El archivo no contiene columnas numéricas.")
    col = sb.selectbox(f"Columna de la muestra{sfx}", cols,
                       index=min(col_index, len(cols) - 1), key=f"h_col{idx}")
    out = dl.numeric_summary(df[col])
    out["column"] = col
    return out


def _prop_sample(df: Optional[pd.DataFrame], idx: str, defaults: tuple = (0, 30), col_index: int = 0) -> dict:
    sb = st.sidebar
    sfx = f" {idx}" if idx else ""
    if df is None:
        n = int(sb.number_input(f"Tamaño de muestra n{sfx}", min_value=1, value=int(defaults[1]),
                                step=1, key=f"h_pn{idx}", help=HELP["n"]))
        x = int(sb.number_input(f"Número de éxitos x{sfx}", min_value=0, value=int(defaults[0]),
                                step=1, key=f"h_px{idx}", help=HELP["x_success"]))
        return {"x": x, "n": n}
    cols = list(df.columns)
    col = sb.selectbox(f"Columna de la muestra{sfx}", cols,
                       index=min(col_index, len(cols) - 1), key=f"h_pcol{idx}")
    values = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
    if not values:
        raise ValueError(f"La columna «{col}» está vacía.")
    succ = sb.selectbox(f"Valor considerado «éxito»{sfx}", values, key=f"h_succ{idx}",
                        help=HELP["success_value"])
    out = dl.proportion_summary(df[col], succ)
    out.update(column=col, success=succ)
    return out


# ------------------------------------------------------------------ #
# Ejecución de cada prueba
# ------------------------------------------------------------------ #
def _run(test_key: str, alpha: float, tail: str, df: Optional[pd.DataFrame]):
    """Recoge entradas de la barra lateral, ejecuta la prueba y devuelve (resultado, resúmenes)."""
    sb = st.sidebar
    sb.subheader("Datos de la muestra")
    file_mode = df is not None

    if test_key == "z1m":
        mu0 = sb.number_input("Valor hipotético μ₀", value=50.0, format="%.4f", key="h_mu0",
                              help=HELP["mu0"])
        st.session_state.setdefault("h_sigma_known", True)
        known = sb.checkbox("Se conoce σ poblacional", key="h_sigma_known",
                            help=HELP["sigma_known"])
        lbl = "Desviación estándar poblacional σ" if known and not file_mode else \
              "Desviación estándar muestral s"
        m = _mean_sample(df, "", (52.0, 8.0, 40), sd_label=lbl)
        sigma = m["std"]
        if file_mode and known:
            sigma = sb.number_input("σ poblacional conocida", min_value=1e-6, value=float(m["std"]),
                                    format="%.4f", key="h_sigma_file", help=HELP["sigma_pop"])
        return hyp.z_one_mean(m["mean"], mu0, sigma, m["n"], alpha, tail, sigma_known=known), [m]

    if test_key == "t1m":
        mu0 = sb.number_input("Valor hipotético μ₀", value=50.0, format="%.4f", key="h_mu0",
                              help=HELP["mu0"])
        m = _mean_sample(df, "", (53.0, 6.0, 15))
        return hyp.t_one_mean(m["mean"], mu0, m["std"], m["n"], alpha, tail), [m]

    if test_key in ("z2m", "t2m"):
        d0 = sb.number_input("Diferencia hipotética (μ₁ − μ₂)₀", value=0.0, format="%.4f",
                             key="h_d0", help=HELP["d0"])
        lbl = ("Desviación estándar σ" if test_key == "z2m" and not file_mode
               else "Desviación estándar muestral s")
        big = test_key == "z2m"
        sb.markdown("**Muestra 1**")
        m1 = _mean_sample(df, "1", (78.0, 10.0, 45) if big else (78.0, 6.0, 12), lbl, 0)
        sb.markdown("**Muestra 2**")
        m2 = _mean_sample(df, "2", (74.0, 12.0, 50) if big else (72.5, 7.0, 14), lbl, 1)
        if test_key == "z2m":
            res = hyp.z_two_means(m1["mean"], m1["std"], m1["n"], m2["mean"], m2["std"], m2["n"],
                                  d0, alpha, tail)
        else:
            eq = sb.checkbox("Asumir varianzas iguales", value=True, key="h_equal_var",
                             help=HELP["equal_var"])
            res = hyp.t_two_means(m1["mean"], m1["std"], m1["n"], m2["mean"], m2["std"], m2["n"],
                                  d0, alpha, tail, equal_var=eq)
        return res, [m1, m2]

    if test_key == "z1p":
        p0 = sb.number_input("Proporción hipotética p₀", min_value=0.001, max_value=0.999,
                             value=0.5, step=0.01, format="%.4f", key="h_p0", help=HELP["p0"])
        m = _prop_sample(df, "", (58, 100))
        return hyp.z_one_prop(m["x"], m["n"], p0, alpha, tail), [m]

    if test_key == "z2p":
        sb.markdown("**Muestra 1**")
        m1 = _prop_sample(df, "1", (45, 120), 0)
        sb.markdown("**Muestra 2**")
        m2 = _prop_sample(df, "2", (30, 110), 1)
        return hyp.z_two_props(m1["x"], m1["n"], m2["x"], m2["n"], alpha, tail), [m1, m2]

    if test_key == "chi2v":
        s0 = sb.number_input("Varianza hipotética σ₀²", min_value=1e-6, value=25.0,
                             format="%.4f", key="h_sigma0sq", help=HELP["sigma0_sq"])
        if file_mode:
            m = _mean_sample(df, "")
        else:  # la media no interviene en el estadístico χ²: solo se piden s y n
            s = sb.number_input("Desviación estándar muestral s", min_value=1e-6, value=6.2,
                                format="%.4f", key="h_s_chi", help=HELP["s"])
            n = int(sb.number_input("Tamaño de muestra n", min_value=2, value=20, step=1,
                                    key="h_n_chi", help=HELP["n"]))
            m = {"std": s, "n": n}
        return hyp.chi2_variance(m["std"], m["n"], s0, alpha, tail), [m]

    raise ValueError(f"Prueba no registrada: {test_key}")


# ------------------------------------------------------------------ #
# Página
# ------------------------------------------------------------------ #
def render() -> None:
    sb = st.sidebar
    sb.header("Configuración de la prueba")
    test_key = sb.selectbox("Prueba estadística", list(hyp.TEST_REGISTRY),
                            format_func=lambda k: hyp.TEST_REGISTRY[k]["label"],
                            key="test_key", help=HELP["test"])
    spec = hyp.TEST_REGISTRY[test_key]
    sb.caption(spec["description"])
    tail = sb.radio("Tipo de cola", list(TAIL_LABELS), format_func=TAIL_LABELS.get,
                    key="tail", help=HELP["tail"])
    alpha = alpha_input()
    mode = sb.radio("Fuente de datos", [SUMMARY, FILE], key="data_mode", help=HELP["data_mode"])

    st.title("Pruebas de hipótesis")

    df = None
    if mode == FILE:
        up = sb.file_uploader("Archivo de datos", type=["csv", "xlsx", "xls"], key="h_file",
                              help="Una columna por muestra. La primera fila debe ser el encabezado.")
        if up is None:
            st.info("Sube un archivo CSV o Excel en la barra lateral. Cada muestra debe ocupar "
                    "una columna (con encabezado). Para proporciones, la columna puede contener "
                    "categorías como «Sí/No». Hay un archivo de ejemplo en `sample_data/`.")
            return
        try:
            df = _load(up.getvalue(), up.name)
        except Exception as err:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {err}")
            return

    try:
        res, summaries = _run(test_key, alpha, tail, df)
    except ValueError as err:
        st.error(f"Revisa los datos de entrada: {err}")
        return

    st.subheader(res.name)

    # Resumen calculado desde el archivo
    if df is not None:
        with st.expander("Estadísticos calculados a partir del archivo", expanded=True):
            rows = []
            for i, m in enumerate(summaries, 1):
                if "mean" in m:
                    rows.append({"Muestra": i, "Columna": m["column"], "n": m["n"],
                                 "Media x̄": round(m["mean"], 4), "Desv. est. s": round(m["std"], 4)})
                else:
                    rows.append({"Muestra": i, "Columna": m["column"], "n": m["n"],
                                 "Éxito": m["success"], "x": m["x"], "p̂": round(m["p_hat"], 4)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width=WIDTH)
            st.caption(f"Vista previa del archivo ({len(df)} filas):")
            st.dataframe(df.head(8), hide_index=True, width=WIDTH)

    # Hipótesis
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Hipótesis nula**")
        st.latex(res.h0_latex)
    with c2:
        st.markdown("**Hipótesis alternativa**")
        st.latex(res.h1_latex)

    # Métricas
    lo, hi = res.critical
    if res.tail == "two" and res.distribution in ("norm", "t"):
        crit = f"± {hi:.4f}"
    else:
        crit = " y ".join(f"{c:.4f}" for c in (lo, hi) if c is not None)
    items = [(f"Estadístico {res.label}", f"{res.statistic:.4f}"),
             ("Valor(es) crítico(s)", crit),
             ("p-valor", fmt_p(res.p_value)),
             ("Error estándar", fmt(res.std_error) if res.std_error is not None else "—")]
    if res.df is not None:
        items.append(("Grados de libertad", fmt(res.df, 2)))
    metric_row(items)

    # Gráfico interactivo
    st.plotly_chart(charts.plot_hypothesis_test(res), width=WIDTH)
    st.caption("Pasa el cursor sobre la curva para ver la densidad y la probabilidad acumulada. "
               "Haz clic en «Área del p-valor» en la leyenda para sombrearla.")

    # Veredicto
    st.markdown("#### Conclusión")
    verdict_box(res.conclusion, res.reject)
    render_notes(res.notes)

    # Desarrollo
    with st.expander("Ver el procedimiento académico completo", expanded=True):
        st.markdown("**Planteamiento de hipótesis**")
        st.latex(rf"{res.h0_latex} \qquad {res.h1_latex}")
        render_steps(res.steps, title="Desarrollo")

    with st.expander("¿Cuándo usar t-Student vs Z?"):
        st.markdown(HELP["test"])
