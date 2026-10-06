"""
core/common.py
==============
Utilidades compartidas por todos los módulos de cálculo.

Este archivo NO depende de Streamlit ni de Plotly: es lógica pura (SciPy/NumPy),
lo que permite reutilizarla en notebooks, pruebas unitarias o en otra interfaz.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy import stats

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
TAIL_LABELS = {
    "two": "Bilateral (dos colas)",
    "left": "Unilateral izquierda",
    "right": "Unilateral derecha",
}


# --------------------------------------------------------------------------- #
# Paso del procedimiento académico
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """Un paso del desarrollo: título, fórmula LaTeX y/o texto explicativo."""

    title: str
    latex: str = ""
    text: str = ""


# --------------------------------------------------------------------------- #
# Formato numérico (texto plano y LaTeX)
# --------------------------------------------------------------------------- #
def fmt(x, d: int = 4) -> str:
    """Formatea un número con `d` decimales eliminando ceros sobrantes."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    s = f"{x:.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def tex(x, d: int = 4) -> str:
    """Igual que `fmt`, pero encierra negativos entre paréntesis (para sustituciones)."""
    s = fmt(x, d)
    return f"({s})" if s.startswith("-") else s


def fmt_p(p: float) -> str:
    """Formato académico del p-valor en texto plano."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def tex_p(p: float) -> str:
    """Formato del p-valor en LaTeX (notación científica si es muy pequeño)."""
    if p >= 1e-4:
        return f"{p:.4f}"
    mant, exp = f"{p:.3e}".split("e")
    return rf"{mant} \times 10^{{{int(exp)}}}"


# --------------------------------------------------------------------------- #
# Distribuciones
# --------------------------------------------------------------------------- #
def z_from_confidence(confidence: float) -> float:
    """Valor crítico Z_{α/2} para un nivel de confianza (0 < c < 1)."""
    if not 0 < confidence < 1:
        raise ValueError("El nivel de confianza debe estar entre 0% y 100% (exclusivo).")
    return float(stats.norm.ppf(1 - (1 - confidence) / 2))


def get_dist(name: str, df: Optional[float] = None):
    """Devuelve el objeto 'congelado' de SciPy para la distribución indicada."""
    if name == "norm":
        return stats.norm()
    if name == "t":
        return stats.t(df)
    if name == "chi2":
        return stats.chi2(df)
    raise ValueError(f"Distribución no soportada: {name}")


def critical_values(dist: str, alpha: float, tail: str,
                    df: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
    """(crítico_inferior, crítico_superior); None si esa cola no existe."""
    d = get_dist(dist, df)
    if tail == "two":
        return float(d.ppf(alpha / 2)), float(d.ppf(1 - alpha / 2))
    if tail == "left":
        return float(d.ppf(alpha)), None
    if tail == "right":
        return None, float(d.ppf(1 - alpha))
    raise ValueError(f"Tipo de cola no válido: {tail}")


def p_value(dist: str, stat: float, tail: str, df: Optional[float] = None) -> float:
    """p-valor del estadístico observado según el tipo de cola."""
    d = get_dist(dist, df)
    if tail == "left":
        return float(d.cdf(stat))
    if tail == "right":
        return float(d.sf(stat))
    # Bilateral
    if dist in ("norm", "t"):  # distribuciones simétricas
        return float(min(1.0, 2 * d.sf(abs(stat))))
    # Asimétrica (χ²): se duplica la cola más pequeña
    return float(min(1.0, 2 * min(d.cdf(stat), d.sf(stat))))
