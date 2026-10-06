"""
app.py — Punto de entrada de InferStat U
=========================================
Aplicación de Estadística Inferencial para nivel universitario.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

Arquitectura (capas desacopladas):
    core/  → matemática pura (SciPy/NumPy). Sin Streamlit. Probado con pytest.
    viz/   → figuras Plotly a partir de los resultados del núcleo.
    ui/    → páginas y componentes de Streamlit.
    app.py → configuración global y enrutamiento entre módulos.

Para añadir un módulo nuevo (ANOVA, no paramétricas, regresión):
    1. core/anova.py   (funciones que devuelven un resultado con pasos LaTeX)
    2. viz/charts.py   (o viz/anova_charts.py) con sus figuras
    3. ui/page_anova.py con una función render()
    4. Registrar la página en el diccionario PAGES de este archivo.
"""
import streamlit as st

APP_NAME = "InferStat U"
AUTHOR = "ppaolo.nexus"          # ← autoría mostrada en la barra lateral, el pie y el menú «About»

st.set_page_config(
    page_title=f"{APP_NAME} · Estadística Inferencial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"**{APP_NAME}** — Estadística inferencial paso a paso.\n\n"
                         f"Autor: **{AUTHOR}**"},
)

from ui import page_hypothesis, page_sampling, page_wizard  # noqa: E402
from ui.components import (GLOBAL_CSS, PAGE_HYPOTHESIS, PAGE_SAMPLING,  # noqa: E402
                           PAGE_WIZARD)

# Registro de páginas: nombre visible → función render()
PAGES = {
    PAGE_WIZARD: page_wizard.render,
    PAGE_SAMPLING: page_sampling.render,
    PAGE_HYPOTHESIS: page_hypothesis.render,
}

ROADMAP = ["ANOVA de un factor", "Pruebas no paramétricas", "Regresión lineal"]


def main() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    sb = st.sidebar
    sb.title(APP_NAME)
    sb.caption(f"Estadística inferencial paso a paso · por **{AUTHOR}**")
    page = sb.radio("Módulo", list(PAGES), key="nav", label_visibility="collapsed")
    sb.divider()

    PAGES[page]()

    sb.divider()
    sb.caption("Próximamente: " + ", ".join(ROADMAP) + ".")

    # Pie de página con la autoría
    st.divider()
    st.markdown(
        f'<div class="app-footer">{APP_NAME} · Desarrollado por <b>{AUTHOR}</b> · '
        f'Python, Streamlit, SciPy y Plotly</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
