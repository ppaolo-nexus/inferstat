"""
viz/charts.py
=============
Gráficos Plotly. Recibe objetos de resultado del núcleo (`core`) y devuelve
figuras; no contiene lógica estadística de decisión ni código de Streamlit.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import numpy as np
import plotly.graph_objects as go

from core.common import fmt, fmt_p, get_dist
from core.hypothesis import TestResult

PALETTE = {
    "curve": "#1B3A6B",            # azul tinta
    "reject": "rgba(255, 23, 68, 0.60)",     # rojo brillante (región de rechazo)
    "accept": "rgba(38, 166, 154, 0.28)",    # verde azulado claro (no rechazo)
    "pvalue": "rgba(255, 179, 0, 0.55)",     # ámbar (área del p-valor)
    "stat": "#FF6D00",            # naranja intenso (estadístico calculado)
    "critical": "#B71C1C",
}


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def _area(d, a: float, b: float, color: str, name: str, showlegend=True,
          group: Optional[str] = None, visible=True) -> Optional[go.Scatter]:
    """Polígono cerrado bajo la densidad entre a y b."""
    if b <= a:
        return None
    xs = np.linspace(a, b, 250)
    ys = d.pdf(xs)
    return go.Scatter(
        x=np.concatenate([[a], xs, [b]]), y=np.concatenate([[0], ys, [0]]),
        fill="toself", fillcolor=color, line=dict(width=0), mode="lines",
        name=name, legendgroup=group or name, showlegend=showlegend,
        hoverinfo="skip", visible=visible,
    )


def _x_range(res: TestResult, d) -> tuple[float, float]:
    lo_c, hi_c = res.critical
    s = res.statistic
    if res.distribution in ("norm", "t"):
        m = 4.0
        for v in (s, lo_c, hi_c):
            if v is not None and math.isfinite(v):
                m = max(m, abs(v) * 1.15 + 0.3)
        return -m, m
    hi = float(d.ppf(0.999))
    for v in (s, hi_c):
        if v is not None and math.isfinite(v):
            hi = max(hi, v * 1.15)
    return float(d.ppf(0.0005)), hi


# --------------------------------------------------------------------------- #
# Gráfico principal: distribución + regiones + estadístico
# --------------------------------------------------------------------------- #
def plot_hypothesis_test(res: TestResult, height: int = 470) -> go.Figure:
    d = get_dist(res.distribution, res.df)
    lo, hi = _x_range(res, d)
    lo_c, hi_c = res.critical
    s = res.statistic

    fig = go.Figure()

    # 1) Región de aceptación (no rechazo)
    a_start = max(lo, lo_c) if lo_c is not None else lo
    a_end = min(hi, hi_c) if hi_c is not None else hi
    tr = _area(d, a_start, a_end, PALETTE["accept"], f"No rechazo H₀ (1 − α = {fmt(1 - res.alpha)})")
    if tr:
        fig.add_trace(tr)

    # 2) Regiones de rechazo (α)
    first = True
    for a, b in ((lo, lo_c), (hi_c, hi)):
        if a is None or b is None:
            continue
        tr = _area(d, max(a, lo), min(b, hi), PALETTE["reject"],
                   f"Región de rechazo (α = {fmt(res.alpha)})", showlegend=first, group="reject")
        if tr:
            fig.add_trace(tr)
            first = False

    # 3) Área del p-valor (oculta por defecto; se activa desde la leyenda)
    if res.tail == "left":
        segs = [(lo, s)]
    elif res.tail == "right":
        segs = [(s, hi)]
    elif res.distribution in ("norm", "t"):
        segs = [(lo, -abs(s)), (abs(s), hi)]
    else:
        segs = [(lo, s)] if d.cdf(s) < 0.5 else [(s, hi)]
    first = True
    for a, b in segs:
        tr = _area(d, max(a, lo), min(b, hi), PALETTE["pvalue"],
                   "Área del p-valor (p " + ("= " if res.p_value >= 1e-4 else "") + fmt_p(res.p_value) + ")", showlegend=first,
                   group="pvalue", visible="legendonly")
        if tr:
            fig.add_trace(tr)
            first = False

    # 4) Curva teórica con tooltips (densidad y probabilidad acumulada)
    x = np.linspace(lo, hi, 800)
    y = d.pdf(x)
    cdf = d.cdf(x)
    dist_name = {"norm": "Normal estándar N(0, 1)",
                 "t": f"t-Student (gl = {fmt(res.df, 2)})",
                 "chi2": f"Chi-cuadrada (gl = {fmt(res.df, 2)})"}[res.distribution]
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=dist_name,
        line=dict(color=PALETTE["curve"], width=3),
        customdata=np.column_stack([cdf, 1 - cdf]),
        hovertemplate=(f"<b>{res.label} = %{{x:.4f}}</b><br>"
                       "Densidad f(x) = %{y:.4f}<br>"
                       f"P({res.label} ≤ x) = %{{customdata[0]:.4f}}<br>"
                       f"P({res.label} > x) = %{{customdata[1]:.4f}}<extra></extra>"),
    ))

    y_top = float(np.nanmax(y[np.isfinite(y)]))
    if res.distribution == "chi2":
        y_top = min(y_top, 1.0)  # gl pequeños tienen densidad → ∞ cerca de 0

    # 5) Valores críticos
    for c in (lo_c, hi_c):
        if c is None:
            continue
        fig.add_shape(type="line", x0=c, x1=c, y0=0, y1=y_top * 0.92,
                      line=dict(color=PALETTE["critical"], width=1.5, dash="dot"))
        fig.add_annotation(x=c, y=y_top * 0.97, text=f"crítico<br>{c:.4f}", showarrow=False,
                           font=dict(size=11, color=PALETTE["critical"]))

    # 6) Estadístico calculado: línea punteada + marcador llamativo
    ys = float(min(d.pdf(s), y_top)) if math.isfinite(d.pdf(s)) else y_top
    fig.add_shape(type="line", x0=s, x1=s, y0=0, y1=y_top * 1.08,
                  line=dict(color=PALETTE["stat"], width=3, dash="dash"))
    fig.add_trace(go.Scatter(
        x=[s], y=[ys], mode="markers", name=f"{res.label} calculado",
        marker=dict(symbol="star", size=22, color=PALETTE["stat"],
                    line=dict(color="white", width=1.5)),
        hovertemplate=(f"<b>{res.label} calculado = {s:.4f}</b><br>"
                       f"P({res.label} ≤ {s:.4f}) = {d.cdf(s):.4f}<br>"
                       f"p-valor = {fmt_p(res.p_value)}<extra></extra>"),
    ))
    fig.add_annotation(
        x=s, y=y_top * 1.12, text=f"<b>{res.label}<sub>calc</sub> = {s:.4f}</b>",
        showarrow=False, font=dict(size=14, color="white"),
        bgcolor=PALETTE["stat"], borderpad=5,
    )

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        xaxis=dict(title=f"Valores de {res.label}", zeroline=False, range=[lo, hi]),
        yaxis=dict(title="Densidad f(x)", range=[0, y_top * 1.22], rangemode="tozero"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        title=dict(text=f"{dist_name} · {'Bilateral' if res.tail == 'two' else 'Unilateral'}",
                   x=0.01, font=dict(size=15)),
    )
    return fig


# --------------------------------------------------------------------------- #
# Gráficos del módulo de muestreo
# --------------------------------------------------------------------------- #
def plot_sample_size_curve(func: Callable[[float], int], e_current: float,
                           n_current: int, e_label: str) -> go.Figure:
    """Sensibilidad de n frente al margen de error e."""
    es = np.linspace(e_current * 0.4, e_current * 2.5, 60)
    ns = [func(e) for e in es]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=es, y=ns, mode="lines", line=dict(color=PALETTE["curve"], width=3),
                             name="n requerido",
                             hovertemplate="e = %{x:.4f}<br>n = %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[e_current], y=[n_current], mode="markers+text",
                             marker=dict(size=16, color=PALETTE["stat"], symbol="star"),
                             text=[f"n = {n_current}"], textposition="top right",
                             name="Tu selección",
                             hovertemplate="e = %{x:.4f}<br>n = %{y}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis_title=e_label, yaxis_title="Tamaño de muestra n",
                      title=dict(text="¿Cómo cambia n si modificas el margen de error?",
                                 x=0.01, font=dict(size=14)))
    return fig


def plot_stratified(rows: List[dict]) -> go.Figure:
    names = [r["Estrato"] for r in rows]
    fig = go.Figure(go.Bar(
        x=names, y=[r["n_i asignado"] for r in rows], marker_color=PALETTE["curve"],
        text=[f"n = {r['n_i asignado']}" for r in rows], textposition="outside",
        customdata=[[r["N_i"], r["W_i = N_i/N"]] for r in rows],
        hovertemplate="<b>%{x}</b><br>N_i = %{customdata[0]}<br>W_i = %{customdata[1]:.4f}"
                      "<br>n_i = %{y}<extra></extra>",
    ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="n_i asignado", xaxis_title="Estrato",
                      title=dict(text="Distribución de la muestra por estrato", x=0.01,
                                 font=dict(size=14)))
    return fig


def plot_systematic(N: int, ids: List[int], k: int) -> go.Figure:
    """Línea de la población con los elementos seleccionados resaltados."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, N], y=[0, 0], mode="lines",
                             line=dict(color="rgba(120,120,120,0.5)", width=10),
                             hoverinfo="skip", name="Población"))
    fig.add_trace(go.Scatter(
        x=ids, y=[0] * len(ids), mode="markers", name="Seleccionados",
        marker=dict(size=11 if len(ids) <= 200 else 6, color=PALETTE["stat"],
                    symbol="line-ns-open", line=dict(width=3, color=PALETTE["stat"])),
        customdata=list(range(1, len(ids) + 1)),
        hovertemplate="Elemento #%{customdata}<br>ID = %{x}<extra></extra>",
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis=dict(visible=False, range=[-1, 1]),
                      xaxis=dict(title="ID en el marco muestral", range=[0, N + 1]),
                      title=dict(text=f"Selección sistemática cada k = {k} elementos",
                                 x=0.01, font=dict(size=14)))
    return fig
