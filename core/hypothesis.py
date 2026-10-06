"""
core/hypothesis.py
==================
Pruebas de hipótesis paramétricas.

Cada función de prueba:
  1. Valida entradas.
  2. Calcula error estándar y estadístico.
  3. Registra el desarrollo académico (pasos LaTeX).
  4. Delega en `_build()` el cálculo de valores críticos, p-valor y decisión.

Para agregar una prueba nueva (p. ej. ANOVA → distribución F), basta con:
  - añadir la distribución en `common.get_dist` / `critical_values` / `p_value`,
  - escribir la función de la prueba que retorne `TestResult`,
  - registrarla en `TEST_REGISTRY`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .common import (Step, critical_values, fmt, fmt_p, p_value, tex, tex_p)


# --------------------------------------------------------------------------- #
# Resultado estándar de una prueba
# --------------------------------------------------------------------------- #
@dataclass
class TestResult:
    key: str                      # identificador interno ("z1m", "t2m", ...)
    name: str                     # nombre legible
    symbol: str                   # símbolo LaTeX del estadístico ("Z", "t", r"\chi^2")
    label: str                    # símbolo en texto plano ("Z", "t", "χ²")
    statistic: float
    p_value: float
    alpha: float
    tail: str                     # "two" | "left" | "right"
    distribution: str             # "norm" | "t" | "chi2"
    df: Optional[float]           # grados de libertad (None para Z)
    critical: Tuple[Optional[float], Optional[float]]
    std_error: Optional[float]
    h0_latex: str
    h1_latex: str
    h1_text: str                  # H1 en texto plano para la conclusión
    steps: List[Step]
    notes: List[str] = field(default_factory=list)

    __test__ = False  # evita que pytest la confunda con una clase de prueba

    @property
    def reject(self) -> bool:
        return self.p_value < self.alpha

    @property
    def in_rejection_region(self) -> bool:
        lo, hi = self.critical
        return (lo is not None and self.statistic < lo) or (hi is not None and self.statistic > hi)

    @property
    def conclusion(self) -> str:
        a = fmt(self.alpha)
        stat = f"{self.label} = {self.statistic:.4f}"
        p_txt = "p-valor < 0.0001" if self.p_value < 1e-4 else f"p-valor = {self.p_value:.4f}"
        if self.reject:
            return (f"Con un nivel de significancia de α = {a}, dado que el estadístico calculado "
                    f"({stat}) cae en la región de rechazo ({p_txt} < {a}), se RECHAZA la "
                    f"hipótesis nula (H₀). Existe evidencia estadística suficiente para "
                    f"concluir que {self.h1_text}.")
        return (f"Con un nivel de significancia de α = {a}, dado que el estadístico calculado "
                f"({stat}) cae en la región de no rechazo ({p_txt} ≥ {a}), NO SE RECHAZA la "
                f"hipótesis nula (H₀). No existe evidencia estadística suficiente para "
                f"concluir que {self.h1_text}.")


# --------------------------------------------------------------------------- #
# Validaciones
# --------------------------------------------------------------------------- #
def _positive(**kwargs):
    for name, v in kwargs.items():
        if v is None or not v > 0:
            raise ValueError(f"«{name}» debe ser mayor que 0.")


def _sample_size(n, name="n", minimum=2):
    if n is None or int(n) != n or n < minimum:
        raise ValueError(f"«{name}» debe ser un entero ≥ {minimum}.")


def _alpha_tail(alpha, tail):
    if not 0 < alpha < 1:
        raise ValueError("α debe estar entre 0 y 1.")
    if tail not in ("two", "left", "right"):
        raise ValueError("Tipo de cola no válido.")


# --------------------------------------------------------------------------- #
# Construcción común: hipótesis, críticos, p-valor y decisión
# --------------------------------------------------------------------------- #
_OPS = {  # (operador H0 LaTeX, operador H1 LaTeX, operador H1 texto)
    "two": ("=", r"\neq", "≠"),
    "left": (r"\geq", "<", "<"),
    "right": (r"\leq", ">", ">"),
}


def _build(*, key, name, symbol, label, stat, dist, df, alpha, tail, se,
           param_tex, param_plain, value, steps, notes) -> TestResult:
    h0op, h1op, h1txt = _OPS[tail]
    h0 = rf"H_0:\; {param_tex} {h0op} {tex(value)}"
    h1 = rf"H_1:\; {param_tex} {h1op} {tex(value)}"
    h1_text = f"{param_plain} {h1txt} {fmt(value)}"

    lo, hi = critical_values(dist, alpha, tail, df)
    p = p_value(dist, stat, tail, df)

    # --- Valores críticos -------------------------------------------------- #
    sub_a = {"two": r"\alpha/2", "left": r"\alpha", "right": r"\alpha"}[tail]
    gl = f",\\,{fmt(df, 2)}" if df is not None else ""
    if tail == "two" and dist in ("norm", "t"):
        crit_tex = rf"{symbol}_{{{sub_a}{gl}}} = \pm {fmt(hi)}"
        region = f"Se rechaza H₀ si |{label}| > {fmt(hi)}"
    elif tail == "two":
        crit_tex = (rf"{symbol}_{{\text{{inf}}}} = {fmt(lo)}, \qquad "
                    rf"{symbol}_{{\text{{sup}}}} = {fmt(hi)}")
        region = f"Se rechaza H₀ si {label} < {fmt(lo)} o {label} > {fmt(hi)}"
    elif tail == "left":
        crit_tex = rf"{symbol}_{{\text{{crit}}}} = {fmt(lo)}"
        region = f"Se rechaza H₀ si {label} < {fmt(lo)}"
    else:
        crit_tex = rf"{symbol}_{{\text{{crit}}}} = {fmt(hi)}"
        region = f"Se rechaza H₀ si {label} > {fmt(hi)}"
    steps.append(Step(f"Valor(es) crítico(s) con α = {fmt(alpha)}", crit_tex, region))

    # --- p-valor ----------------------------------------------------------- #
    d_sym = {"norm": "Z", "t": rf"T_{{{fmt(df, 2)}}}", "chi2": rf"\chi^2_{{{fmt(df, 2)}}}"}[dist]
    if tail == "left":
        p_tex = rf"p = P({d_sym} \le {tex(stat)}) = {tex_p(p)}"
    elif tail == "right":
        p_tex = rf"p = P({d_sym} \ge {tex(stat)}) = {tex_p(p)}"
    elif dist in ("norm", "t"):
        p_tex = rf"p = 2\,P({d_sym} \ge |{fmt(stat)}|) = {tex_p(p)}"
    else:
        p_tex = (rf"p = 2\,\min\{{P({d_sym} \le {fmt(stat)}),\; "
                 rf"P({d_sym} \ge {fmt(stat)})\}} = {tex_p(p)}")
    steps.append(Step("Cálculo del p-valor", p_tex))

    # --- Decisión ---------------------------------------------------------- #
    reject = p < alpha
    cmp_op = "<" if reject else r"\geq"
    verdict = "Se rechaza" if reject else "No se rechaza"
    dec_tex = (rf"p = {tex_p(p)} {cmp_op} \alpha = {fmt(alpha)}"
               rf"\;\Rightarrow\; \text{{{verdict} }} H_0")
    steps.append(Step(
        "Regla de decisión", dec_tex,
        f"El estadístico {label} = {stat:.4f} "
        f"{'SÍ' if reject else 'NO'} cae en la región de rechazo."))

    return TestResult(key=key, name=name, symbol=symbol, label=label, statistic=float(stat),
                      p_value=p, alpha=alpha, tail=tail, distribution=dist, df=df,
                      critical=(lo, hi), std_error=se, h0_latex=h0, h1_latex=h1,
                      h1_text=h1_text, steps=steps, notes=notes)


# --------------------------------------------------------------------------- #
# MEDIAS
# --------------------------------------------------------------------------- #
def z_one_mean(xbar, mu0, sigma, n, alpha=0.05, tail="two", sigma_known=True) -> TestResult:
    """Z para una media. σ conocida (o s como aproximación si n ≥ 30)."""
    _positive(**{"σ": sigma}); _sample_size(n); _alpha_tail(alpha, tail)
    sym = r"\sigma" if sigma_known else "s"
    se = sigma / math.sqrt(n)
    z = (xbar - mu0) / se
    steps = [
        Step("Error estándar de la media",
             rf"\sigma_{{\bar{{x}}}} = \frac{{{sym}}}{{\sqrt{{n}}}} = "
             rf"\frac{{{fmt(sigma)}}}{{\sqrt{{{n}}}}} = {fmt(se)}"),
        Step("Fórmula del estadístico de prueba",
             rf"Z = \frac{{\bar{{x}} - \mu_0}}{{{sym}/\sqrt{{n}}}}"),
        Step("Sustitución y resultado",
             rf"Z = \frac{{{tex(xbar)} - {tex(mu0)}}}{{{fmt(se)}}} = {fmt(z)}"),
    ]
    notes = []
    if not sigma_known and n < 30:
        notes.append("σ es desconocida y n < 30: la prueba adecuada es t-Student.")
    elif not sigma_known:
        notes.append("Se usa s como estimador de σ (aproximación válida por el TLC, n ≥ 30).")
    return _build(key="z1m", name="Prueba Z para una media", symbol="Z", label="Z", stat=z,
                  dist="norm", df=None, alpha=alpha, tail=tail, se=se, param_tex=r"\mu",
                  param_plain="μ", value=mu0, steps=steps, notes=notes)


def z_two_means(x1, s1, n1, x2, s2, n2, d0=0.0, alpha=0.05, tail="two") -> TestResult:
    """Z para diferencia de dos medias independientes (σ conocidas o n ≥ 30)."""
    _positive(**{"σ₁": s1, "σ₂": s2}); _sample_size(n1, "n₁"); _sample_size(n2, "n₂")
    _alpha_tail(alpha, tail)
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    z = ((x1 - x2) - d0) / se
    steps = [
        Step("Error estándar de la diferencia",
             rf"\sigma_{{\bar{{x}}_1-\bar{{x}}_2}} = \sqrt{{\frac{{\sigma_1^2}}{{n_1}} + "
             rf"\frac{{\sigma_2^2}}{{n_2}}}} = \sqrt{{\frac{{{fmt(s1)}^2}}{{{n1}}} + "
             rf"\frac{{{fmt(s2)}^2}}{{{n2}}}}} = {fmt(se)}"),
        Step("Fórmula del estadístico de prueba",
             rf"Z = \frac{{(\bar{{x}}_1 - \bar{{x}}_2) - (\mu_1-\mu_2)_0}}"
             rf"{{\sqrt{{\sigma_1^2/n_1 + \sigma_2^2/n_2}}}}"),
        Step("Sustitución y resultado",
             rf"Z = \frac{{({tex(x1)} - {tex(x2)}) - {tex(d0)}}}{{{fmt(se)}}} = {fmt(z)}"),
    ]
    notes = []
    if min(n1, n2) < 30:
        notes.append("Alguna muestra tiene n < 30: si las σ son desconocidas, use t-Student.")
    return _build(key="z2m", name="Prueba Z para dos medias", symbol="Z", label="Z", stat=z,
                  dist="norm", df=None, alpha=alpha, tail=tail, se=se,
                  param_tex=r"\mu_1 - \mu_2", param_plain="μ₁ − μ₂", value=d0,
                  steps=steps, notes=notes)


def t_one_mean(xbar, mu0, s, n, alpha=0.05, tail="two") -> TestResult:
    """t-Student para una media (σ desconocida, población normal)."""
    _positive(s=s); _sample_size(n); _alpha_tail(alpha, tail)
    se = s / math.sqrt(n)
    t = (xbar - mu0) / se
    df = n - 1
    steps = [
        Step("Grados de libertad", rf"gl = n - 1 = {n} - 1 = {df}"),
        Step("Error estándar de la media",
             rf"s_{{\bar{{x}}}} = \frac{{s}}{{\sqrt{{n}}}} = "
             rf"\frac{{{fmt(s)}}}{{\sqrt{{{n}}}}} = {fmt(se)}"),
        Step("Fórmula del estadístico de prueba",
             rf"t = \frac{{\bar{{x}} - \mu_0}}{{s/\sqrt{{n}}}}"),
        Step("Sustitución y resultado",
             rf"t = \frac{{{tex(xbar)} - {tex(mu0)}}}{{{fmt(se)}}} = {fmt(t)}"),
    ]
    notes = ["Supuesto: la población de origen es aproximadamente normal."]
    return _build(key="t1m", name="Prueba t-Student para una media", symbol="t", label="t",
                  stat=t, dist="t", df=df, alpha=alpha, tail=tail, se=se, param_tex=r"\mu",
                  param_plain="μ", value=mu0, steps=steps, notes=notes)


def t_two_means(x1, s1, n1, x2, s2, n2, d0=0.0, alpha=0.05, tail="two",
                equal_var=True) -> TestResult:
    """t-Student para dos medias: varianzas iguales (combinada) o Welch."""
    _positive(s1=s1, s2=s2); _sample_size(n1, "n₁"); _sample_size(n2, "n₂")
    _alpha_tail(alpha, tail)
    steps: List[Step] = []
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        steps += [
            Step("Varianza combinada (pooled)",
                 rf"s_p^2 = \frac{{(n_1-1)s_1^2 + (n_2-1)s_2^2}}{{n_1+n_2-2}} = "
                 rf"\frac{{({n1}-1)({fmt(s1)})^2 + ({n2}-1)({fmt(s2)})^2}}{{{df}}} = {fmt(sp2)}"),
            Step("Grados de libertad", rf"gl = n_1 + n_2 - 2 = {df}"),
            Step("Error estándar",
                 rf"SE = \sqrt{{s_p^2\left(\frac{{1}}{{n_1}}+\frac{{1}}{{n_2}}\right)}} = "
                 rf"\sqrt{{{fmt(sp2)}\left(\frac{{1}}{{{n1}}}+\frac{{1}}{{{n2}}}\right)}} = {fmt(se)}"),
        ]
        name = "Prueba t para dos medias (varianzas iguales)"
    else:
        a, b = s1 ** 2 / n1, s2 ** 2 / n2
        se = math.sqrt(a + b)
        df = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
        steps += [
            Step("Error estándar (Welch)",
                 rf"SE = \sqrt{{\frac{{s_1^2}}{{n_1}}+\frac{{s_2^2}}{{n_2}}}} = "
                 rf"\sqrt{{{fmt(a)} + {fmt(b)}}} = {fmt(se)}"),
            Step("Grados de libertad (Welch–Satterthwaite)",
                 rf"gl = \frac{{\left(\frac{{s_1^2}}{{n_1}}+\frac{{s_2^2}}{{n_2}}\right)^2}}"
                 rf"{{\frac{{(s_1^2/n_1)^2}}{{n_1-1}}+\frac{{(s_2^2/n_2)^2}}{{n_2-1}}}} = {fmt(df, 2)}"),
        ]
        name = "Prueba t de Welch para dos medias (varianzas distintas)"
    t = ((x1 - x2) - d0) / se
    steps += [
        Step("Fórmula del estadístico de prueba",
             rf"t = \frac{{(\bar{{x}}_1 - \bar{{x}}_2) - (\mu_1-\mu_2)_0}}{{SE}}"),
        Step("Sustitución y resultado",
             rf"t = \frac{{({tex(x1)} - {tex(x2)}) - {tex(d0)}}}{{{fmt(se)}}} = {fmt(t)}"),
    ]
    notes = ["Supuestos: muestras independientes y poblaciones aproximadamente normales."]
    return _build(key="t2m", name=name, symbol="t", label="t", stat=t, dist="t", df=df,
                  alpha=alpha, tail=tail, se=se, param_tex=r"\mu_1 - \mu_2",
                  param_plain="μ₁ − μ₂", value=d0, steps=steps, notes=notes)


# --------------------------------------------------------------------------- #
# PROPORCIONES
# --------------------------------------------------------------------------- #
def z_one_prop(x, n, p0, alpha=0.05, tail="two") -> TestResult:
    """Z para una proporción (x éxitos en n ensayos)."""
    _sample_size(n, minimum=1); _alpha_tail(alpha, tail)
    if not 0 <= x <= n:
        raise ValueError("El número de éxitos x debe estar entre 0 y n.")
    if not 0 < p0 < 1:
        raise ValueError("p₀ debe estar entre 0 y 1 (exclusivo).")
    p_hat = x / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se
    steps = [
        Step("Proporción muestral", rf"\hat{{p}} = \frac{{x}}{{n}} = \frac{{{x}}}{{{n}}} = {fmt(p_hat)}"),
        Step("Error estándar bajo H₀",
             rf"\sigma_{{\hat{{p}}}} = \sqrt{{\frac{{p_0(1-p_0)}}{{n}}}} = "
             rf"\sqrt{{\frac{{{fmt(p0)}(1-{fmt(p0)})}}{{{n}}}}} = {fmt(se)}"),
        Step("Fórmula del estadístico de prueba",
             rf"Z = \frac{{\hat{{p}} - p_0}}{{\sqrt{{p_0(1-p_0)/n}}}}"),
        Step("Sustitución y resultado",
             rf"Z = \frac{{{fmt(p_hat)} - {fmt(p0)}}}{{{fmt(se)}}} = {fmt(z)}"),
    ]
    notes = []
    if n * p0 < 5 or n * (1 - p0) < 5:
        notes.append(f"No se cumple np₀ ≥ 5 y n(1−p₀) ≥ 5 (np₀ = {fmt(n * p0, 2)}): "
                     "la aproximación normal puede ser imprecisa; considere la prueba binomial exacta.")
    return _build(key="z1p", name="Prueba Z para una proporción", symbol="Z", label="Z", stat=z,
                  dist="norm", df=None, alpha=alpha, tail=tail, se=se, param_tex="p",
                  param_plain="p", value=p0, steps=steps, notes=notes)


def z_two_props(x1, n1, x2, n2, alpha=0.05, tail="two") -> TestResult:
    """Z para diferencia de dos proporciones (H₀: p₁ − p₂ = 0, proporción combinada)."""
    _sample_size(n1, "n₁", 1); _sample_size(n2, "n₂", 1); _alpha_tail(alpha, tail)
    if not (0 <= x1 <= n1 and 0 <= x2 <= n2):
        raise ValueError("Cada número de éxitos debe estar entre 0 y su n.")
    p1, p2 = x1 / n1, x2 / n2
    pbar = (x1 + x2) / (n1 + n2)
    if pbar in (0, 1):
        raise ValueError("La proporción combinada es 0 o 1: el error estándar es nulo.")
    se = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    steps = [
        Step("Proporciones muestrales",
             rf"\hat{{p}}_1 = \frac{{{x1}}}{{{n1}}} = {fmt(p1)}, \qquad "
             rf"\hat{{p}}_2 = \frac{{{x2}}}{{{n2}}} = {fmt(p2)}"),
        Step("Proporción combinada",
             rf"\bar{{p}} = \frac{{x_1 + x_2}}{{n_1 + n_2}} = \frac{{{x1}+{x2}}}{{{n1}+{n2}}} = {fmt(pbar)}"),
        Step("Error estándar",
             rf"SE = \sqrt{{\bar{{p}}(1-\bar{{p}})\left(\frac{{1}}{{n_1}}+\frac{{1}}{{n_2}}\right)}} = {fmt(se)}"),
        Step("Fórmula del estadístico de prueba",
             rf"Z = \frac{{\hat{{p}}_1 - \hat{{p}}_2}}{{SE}}"),
        Step("Sustitución y resultado",
             rf"Z = \frac{{{fmt(p1)} - {fmt(p2)}}}{{{fmt(se)}}} = {fmt(z)}"),
    ]
    return _build(key="z2p", name="Prueba Z para dos proporciones", symbol="Z", label="Z",
                  stat=z, dist="norm", df=None, alpha=alpha, tail=tail, se=se,
                  param_tex="p_1 - p_2", param_plain="p₁ − p₂", value=0.0,
                  steps=steps, notes=[])


# --------------------------------------------------------------------------- #
# VARIANZA
# --------------------------------------------------------------------------- #
def chi2_variance(s, n, sigma0_sq, alpha=0.05, tail="two") -> TestResult:
    """χ² para una varianza poblacional."""
    _positive(s=s, **{"σ₀²": sigma0_sq}); _sample_size(n); _alpha_tail(alpha, tail)
    df = n - 1
    chi = df * s ** 2 / sigma0_sq
    steps = [
        Step("Grados de libertad", rf"gl = n - 1 = {n} - 1 = {df}"),
        Step("Varianza muestral", rf"s^2 = ({fmt(s)})^2 = {fmt(s ** 2)}"),
        Step("Fórmula del estadístico de prueba",
             r"\chi^2 = \frac{(n-1)\,s^2}{\sigma_0^2}"),
        Step("Sustitución y resultado",
             rf"\chi^2 = \frac{{({n}-1)\cdot {fmt(s ** 2)}}}{{{fmt(sigma0_sq)}}} = {fmt(chi)}"),
    ]
    notes = ["Supuesto crítico: la población es normal (la prueba χ² es sensible a este supuesto)."]
    return _build(key="chi2v", name="Prueba χ² para una varianza", symbol=r"\chi^2", label="χ²",
                  stat=chi, dist="chi2", df=df, alpha=alpha, tail=tail, se=None,
                  param_tex=r"\sigma^2", param_plain="σ²", value=sigma0_sq,
                  steps=steps, notes=notes)


# --------------------------------------------------------------------------- #
# Registro de pruebas (punto de extensión)
# --------------------------------------------------------------------------- #
TEST_REGISTRY = {
    "z1m": {"label": "Z · una media", "param": "mean", "samples": 1, "func": z_one_mean,
            "description": "σ poblacional conocida, o muestra grande (n ≥ 30)."},
    "t1m": {"label": "t-Student · una media", "param": "mean", "samples": 1, "func": t_one_mean,
            "description": "σ desconocida y población normal; ideal con n < 30."},
    "z2m": {"label": "Z · dos medias", "param": "mean", "samples": 2, "func": z_two_means,
            "description": "Dos muestras independientes con σ conocidas o n₁, n₂ ≥ 30."},
    "t2m": {"label": "t-Student · dos medias", "param": "mean", "samples": 2, "func": t_two_means,
            "description": "Dos muestras independientes pequeñas con σ desconocidas."},
    "z1p": {"label": "Z · una proporción", "param": "prop", "samples": 1, "func": z_one_prop,
            "description": "Proporción de éxitos frente a un valor hipotético p₀."},
    "z2p": {"label": "Z · dos proporciones", "param": "prop", "samples": 2, "func": z_two_props,
            "description": "Compara las proporciones de éxito de dos grupos."},
    "chi2v": {"label": "χ² · una varianza", "param": "var", "samples": 1, "func": chi2_variance,
              "description": "Contrasta la varianza poblacional σ² (requiere normalidad)."},
}
