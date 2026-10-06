"""
ui/page_wizard.py
=================
Asistente de selección: un cuestionario breve que recomienda la técnica
adecuada y lleva al usuario directamente a ella (precargando la selección).
"""
from __future__ import annotations

import streamlit as st

from core.hypothesis import TEST_REGISTRY
from ui.components import PAGE_HYPOTHESIS, PAGE_SAMPLING, SAMPLING_TOOLS, go_to

GOALS = [
    "Determinar cuántos elementos debo encuestar o medir",
    "Repartir una muestra entre grupos o estratos",
    "Elegir los elementos de una lista de forma sistemática",
    "Comprobar una afirmación sobre la población (prueba de hipótesis)",
]


def _recommend(title: str, why: str, label: str, page: str, **state) -> None:
    st.success(f"**Recomendación: {title}**\n\n{why}")
    st.button(label, type="primary", on_click=go_to, args=(page,), kwargs=state,
              key=f"wiz_go_{title}")


def render() -> None:
    st.sidebar.info("Responde las preguntas del panel principal. Al final podrás ir "
                    "directamente a la herramienta recomendada.")
    st.title("¿Qué deseas calcular?")
    st.write("Responde unas preguntas rápidas y te llevaremos a la técnica correcta para tus datos.")

    goal = st.radio("1. ¿Cuál es tu objetivo?", GOALS, index=None, key="wiz_goal")
    if goal is None:
        return

    # -------- Muestreo -------- #
    if goal == GOALS[0]:
        what = st.radio("2. ¿Qué quieres estimar?",
                        ["Un promedio (ej. ingreso medio, peso medio)",
                         "Un porcentaje (ej. % de votantes, % de defectuosos)"],
                        index=None, key="wiz_ss")
        if what is None:
            return
        if what.startswith("Un promedio"):
            _recommend("Tamaño muestral para la media",
                       "Necesitarás una estimación de σ y el margen de error en unidades de la variable.",
                       "Ir al cálculo de tamaño muestral", PAGE_SAMPLING, sampling_tool=SAMPLING_TOOLS[0])
        else:
            _recommend("Tamaño muestral para la proporción",
                       "Si no conoces la proporción esperada, usa p = 0.5.",
                       "Ir al cálculo de tamaño muestral", PAGE_SAMPLING, sampling_tool=SAMPLING_TOOLS[1])
        return
    if goal == GOALS[1]:
        _recommend("Muestreo estratificado con afijación proporcional",
                   "Cada estrato recibe una parte de la muestra proporcional a su tamaño.",
                   "Ir al muestreo estratificado", PAGE_SAMPLING, sampling_tool=SAMPLING_TOOLS[2])
        return
    if goal == GOALS[2]:
        _recommend("Muestreo sistemático",
                   "Se elige un elemento cada k posiciones a partir de un arranque aleatorio.",
                   "Ir al muestreo sistemático", PAGE_SAMPLING, sampling_tool=SAMPLING_TOOLS[3])
        return

    # -------- Pruebas de hipótesis -------- #
    param = st.radio("2. ¿Sobre qué parámetro es la afirmación?",
                     ["Una media (promedio)", "Una proporción (porcentaje)",
                      "Una varianza (dispersión)"], index=None, key="wiz_param")
    if param is None:
        return
    if param.startswith("Una varianza"):
        _recommend(TEST_REGISTRY["chi2v"]["label"],
                   "La prueba χ² contrasta σ² y exige que la población sea normal.",
                   "Ir a la prueba", PAGE_HYPOTHESIS, test_key="chi2v")
        return

    groups = st.radio("3. ¿Cuántos grupos comparas?",
                      ["Un grupo contra un valor de referencia", "Dos grupos independientes"],
                      index=None, key="wiz_groups")
    if groups is None:
        return
    two = groups.startswith("Dos")

    if param.startswith("Una proporción"):
        key = "z2p" if two else "z1p"
        _recommend(TEST_REGISTRY[key]["label"],
                   "Para proporciones se usa la aproximación normal (verifica np ≥ 5 y n(1−p) ≥ 5).",
                   "Ir a la prueba", PAGE_HYPOTHESIS, test_key=key)
        return

    known = st.radio("4. ¿Conoces la desviación estándar POBLACIONAL σ?",
                     ["Sí, el enunciado la da", "No, solo tengo la desviación de la muestra s"],
                     index=None, key="wiz_known")
    if known is None:
        return
    extra = {}
    if known.startswith("Sí"):
        key = "z2m" if two else "z1m"
        extra = {"h_sigma_known": True}
        why = "Con σ conocida el estadístico sigue una distribución normal estándar exacta."
    else:
        size = st.radio("5. ¿Qué tamaño tiene la muestra" + (" más pequeña?" if two else "?"),
                        ["n ≥ 30 (muestra grande)", "n < 30 (muestra pequeña)"],
                        index=None, key="wiz_size")
        if size is None:
            return
        if size.startswith("n ≥"):
            key = "z2m" if two else "z1m"
            extra = {"h_sigma_known": False}
            why = ("Con n ≥ 30 el Teorema del Límite Central permite usar s en lugar de σ y la "
                   "distribución Z. (La prueba t también sería válida y da resultados muy similares.)")
        else:
            key = "t2m" if two else "t1m"
            why = ("Con σ desconocida y n < 30 se usa t-Student con n − 1 grados de libertad. "
                   "Supuesto: la población es aproximadamente normal.")
    _recommend(TEST_REGISTRY[key]["label"], why, "Ir a la prueba", PAGE_HYPOTHESIS,
               test_key=key, **extra)
