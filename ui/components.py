"""
ui/components.py
================
Componentes de Streamlit reutilizables por todas las páginas.
"""
from __future__ import annotations

import html
from typing import List

import streamlit as st

from core.common import Step, fmt
from ui.help_texts import HELP

# ------------------------------------------------------------------ #
# Navegación (nombres de página compartidos por app.py y el asistente)
# ------------------------------------------------------------------ #
PAGE_WIZARD = "Asistente de selección"
PAGE_SAMPLING = "Técnicas de muestreo"
PAGE_HYPOTHESIS = "Pruebas de hipótesis"

SAMPLING_TOOLS = [
    "Tamaño muestral · media (μ)",
    "Tamaño muestral · proporción (P)",
    "Muestreo estratificado",
    "Muestreo sistemático",
]


def go_to(page: str, **state) -> None:
    """Callback: cambia de página y precarga valores de widgets (p. ej. la prueba)."""
    st.session_state["nav"] = page
    for k, v in state.items():
        st.session_state[k] = v


# ------------------------------------------------------------------ #
# Entradas comunes
# ------------------------------------------------------------------ #
def confidence_input(key: str) -> float:
    sb = st.sidebar
    opt = sb.selectbox("Nivel de confianza", ["90 %", "95 %", "99 %", "Personalizado"],
                       index=1, key=f"{key}_conf_opt", help=HELP["confidence"])
    if opt == "Personalizado":
        val = sb.number_input("Nivel de confianza personalizado (%)", min_value=50.0,
                              max_value=99.99, value=97.0, step=0.5, key=f"{key}_conf_custom",
                              help=HELP["confidence"])
        return val / 100
    return float(opt.split()[0]) / 100


def alpha_input(key: str = "alpha") -> float:
    sb = st.sidebar
    opt = sb.selectbox("Nivel de significancia α", ["0.10", "0.05", "0.01", "Personalizado"],
                       index=1, key=f"{key}_opt", help=HELP["alpha"])
    if opt == "Personalizado":
        return float(sb.number_input("α personalizado", min_value=0.0001, max_value=0.5,
                                     value=0.025, step=0.005, format="%.4f",
                                     key=f"{key}_custom", help=HELP["alpha"]))
    return float(opt)


# ------------------------------------------------------------------ #
# Salidas comunes
# ------------------------------------------------------------------ #
def render_steps(steps: List[Step], title: str = "Procedimiento académico paso a paso") -> None:
    st.markdown(f"#### {title}")
    for i, s in enumerate(steps, 1):
        st.markdown(f"**Paso {i}. {s.title}**")
        if s.latex:
            st.latex(s.latex)
        if s.text:
            st.caption(s.text)


def render_notes(notes: List[str]) -> None:
    for n in notes:
        st.warning(n)


def verdict_box(text: str, reject: bool) -> None:
    """Caja de conclusión con color según la decisión."""
    cls = "verdict-reject" if reject else "verdict-accept"
    head = "Se rechaza H₀" if reject else "No se rechaza H₀"
    st.markdown(
        f'<div class="verdict {cls}"><div class="verdict-head">{head}</div>'
        f'<div>{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_row(items: List[tuple]) -> None:
    """items: [(etiqueta, valor, ayuda opcional), ...]"""
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        label, value = item[0], item[1]
        c.metric(label, value if isinstance(value, str) else fmt(value), help=item[2] if len(item) > 2 else None)


GLOBAL_CSS = """
<style>
.verdict {border-radius: 10px; padding: 1rem 1.2rem; margin: .6rem 0 1rem 0;
          line-height: 1.55; border-left: 6px solid;}
.verdict-head {font-weight: 700; font-size: 1.1rem; margin-bottom: .35rem;}
.verdict-reject {background: rgba(255, 23, 68, .10); border-color: #FF1744;}
.verdict-accept {background: rgba(38, 166, 154, .12); border-color: #26A69A;}
.app-footer {text-align: center; font-size: .9rem; opacity: .75; padding: .4rem 0 1rem 0;}
.hyp-card {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: .2rem .8rem;}
</style>
"""
