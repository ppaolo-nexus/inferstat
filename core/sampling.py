"""
core/sampling.py
================
Técnicas de muestreo:
  * Tamaño muestral para la media y la proporción (población infinita y finita).
  * Muestreo estratificado con afijación proporcional.
  * Muestreo sistemático.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .common import Step, fmt, z_from_confidence


@dataclass
class SampleSizeResult:
    n: int                      # tamaño final (redondeado hacia arriba)
    n_raw: float                # valor exacto antes del redondeo
    n0: float                   # tamaño para población infinita
    z: float
    finite: bool
    steps: List[Step] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _z_step(confidence: float, z: float) -> Step:
    a = 1 - confidence
    return Step(
        "Valor crítico de la distribución normal",
        rf"\alpha = 1 - {fmt(confidence)} = {fmt(a)} \;\Rightarrow\; "
        rf"Z_{{\alpha/2}} = Z_{{{fmt(a / 2)}}} = {fmt(z)}",
        f"Nivel de confianza del {fmt(confidence * 100, 2)} %.",
    )


def _round_up(n_raw: float) -> int:
    # Se resta un epsilon para que 96.0000000001 (error de coma flotante) no pase a 97.
    return max(1, math.ceil(n_raw - 1e-9))


def sample_size_mean(sigma: float, e: float, confidence: float,
                     N: Optional[int] = None) -> SampleSizeResult:
    """n para estimar μ con margen de error e."""
    if sigma <= 0 or e <= 0:
        raise ValueError("σ y e deben ser mayores que 0.")
    if N is not None and N < 2:
        raise ValueError("N debe ser un entero ≥ 2.")
    z = z_from_confidence(confidence)
    n0 = (z * sigma / e) ** 2
    steps = [_z_step(confidence, z)]
    if N is None:
        n_raw = n0
        steps += [
            Step("Fórmula (población infinita)",
                 r"n = \left(\frac{Z_{\alpha/2}\,\sigma}{e}\right)^2"),
            Step("Sustitución",
                 rf"n = \left(\frac{{{fmt(z)} \cdot {fmt(sigma)}}}{{{fmt(e)}}}\right)^2 = {fmt(n_raw)}"),
        ]
    else:
        n_raw = N * z ** 2 * sigma ** 2 / (e ** 2 * (N - 1) + z ** 2 * sigma ** 2)
        steps += [
            Step("Fórmula (población finita)",
                 r"n = \frac{N\,Z_{\alpha/2}^2\,\sigma^2}{e^2(N-1) + Z_{\alpha/2}^2\,\sigma^2}"),
            Step("Sustitución",
                 rf"n = \frac{{{N}\cdot {fmt(z)}^2 \cdot {fmt(sigma)}^2}}"
                 rf"{{{fmt(e)}^2({N}-1) + {fmt(z)}^2 \cdot {fmt(sigma)}^2}} = {fmt(n_raw)}",
                 f"Sin la corrección por población finita se necesitarían n₀ = {fmt(n0, 2)}."),
        ]
    n = _round_up(n_raw)
    steps.append(Step("Redondeo hacia arriba", rf"n = \lceil {fmt(n_raw)} \rceil = {n}",
                      "Se redondea siempre hacia arriba para no superar el margen de error fijado."))
    return SampleSizeResult(n=n, n_raw=n_raw, n0=n0, z=z, finite=N is not None, steps=steps)


def sample_size_proportion(p: float, e: float, confidence: float,
                           N: Optional[int] = None) -> SampleSizeResult:
    """n para estimar una proporción P con margen de error e."""
    if not 0 < p < 1:
        raise ValueError("p debe estar entre 0 y 1 (exclusivo).")
    if not 0 < e < 1:
        raise ValueError("e debe estar entre 0 y 1 (p. ej. 0.05 = 5 %).")
    if N is not None and N < 2:
        raise ValueError("N debe ser un entero ≥ 2.")
    q = 1 - p
    z = z_from_confidence(confidence)
    n0 = z ** 2 * p * q / e ** 2
    steps = [_z_step(confidence, z)]
    if N is None:
        n_raw = n0
        steps += [
            Step("Fórmula (población infinita)", r"n = \frac{Z_{\alpha/2}^2\,p\,q}{e^2}"),
            Step("Sustitución",
                 rf"n = \frac{{{fmt(z)}^2 \cdot {fmt(p)} \cdot {fmt(q)}}}{{{fmt(e)}^2}} = {fmt(n_raw)}"),
        ]
    else:
        n_raw = N * z ** 2 * p * q / (e ** 2 * (N - 1) + z ** 2 * p * q)
        steps += [
            Step("Fórmula (población finita)",
                 r"n = \frac{N\,Z_{\alpha/2}^2\,p\,q}{e^2(N-1) + Z_{\alpha/2}^2\,p\,q}"),
            Step("Sustitución",
                 rf"n = \frac{{{N}\cdot {fmt(z)}^2 \cdot {fmt(p)} \cdot {fmt(q)}}}"
                 rf"{{{fmt(e)}^2({N}-1) + {fmt(z)}^2 \cdot {fmt(p)} \cdot {fmt(q)}}} = {fmt(n_raw)}",
                 f"Sin la corrección por población finita se necesitarían n₀ = {fmt(n0, 2)}."),
        ]
    n = _round_up(n_raw)
    steps.append(Step("Redondeo hacia arriba", rf"n = \lceil {fmt(n_raw)} \rceil = {n}"))
    notes = []
    if p == 0.5:
        notes.append("p = 0.5 maximiza p·q: es la opción conservadora cuando no hay estudios previos.")
    return SampleSizeResult(n=n, n_raw=n_raw, n0=n0, z=z, finite=N is not None,
                            steps=steps, notes=notes)


# --------------------------------------------------------------------------- #
# Muestreo estratificado — afijación proporcional
# --------------------------------------------------------------------------- #
@dataclass
class StratifiedResult:
    N: int
    n: int
    rows: List[dict]
    steps: List[Step]


def stratified_proportional(n: int, names: Sequence[str], sizes: Sequence[int]) -> StratifiedResult:
    """n_i = n · N_i / N, ajustado con el método del mayor residuo para que Σ n_i = n."""
    if len(sizes) < 2:
        raise ValueError("Ingrese al menos 2 estratos.")
    if any(s is None or s <= 0 or int(s) != s for s in sizes):
        raise ValueError("Cada N_i debe ser un entero positivo.")
    sizes = [int(s) for s in sizes]
    N = sum(sizes)
    if not 1 <= n <= N:
        raise ValueError(f"n debe estar entre 1 y N = {N}.")

    exact = [n * Ni / N for Ni in sizes]
    alloc = [math.floor(v) for v in exact]
    # Reparto de las unidades sobrantes a los mayores residuos decimales
    remaining = n - sum(alloc)
    order = sorted(range(len(sizes)), key=lambda i: exact[i] - alloc[i], reverse=True)
    for i in order[:remaining]:
        alloc[i] += 1

    rows = [{"Estrato": str(names[i]), "N_i": sizes[i], "W_i = N_i/N": round(sizes[i] / N, 4),
             "n_i exacto": round(exact[i], 4), "n_i asignado": alloc[i]}
            for i in range(len(sizes))]

    lines = r" \\ ".join(
        rf"n_{{{i + 1}}} &= {n}\cdot\frac{{{sizes[i]}}}{{{N}}} = {fmt(exact[i])} \;\to\; {alloc[i]}"
        for i in range(len(sizes)))
    steps = [
        Step("Tamaño de la población", rf"N = \sum N_i = {' + '.join(map(str, sizes))} = {N}"),
        Step("Fórmula de afijación proporcional", r"n_i = n \cdot \frac{N_i}{N} = n\cdot W_i"),
        Step("Sustitución por estrato", rf"\begin{{aligned}} {lines} \end{{aligned}}",
             "Los decimales se resuelven con el método del mayor residuo para que la suma sea exactamente n."),
        Step("Verificación", rf"\sum n_i = {' + '.join(map(str, alloc))} = {sum(alloc)} = n"),
    ]
    return StratifiedResult(N=N, n=n, rows=rows, steps=steps)


# --------------------------------------------------------------------------- #
# Muestreo sistemático
# --------------------------------------------------------------------------- #
@dataclass
class SystematicResult:
    N: int
    n: int
    k: int
    k_exact: float
    start: int
    ids: List[int]
    steps: List[Step]
    notes: List[str]


def systematic_sample(N: int, n: int, start: Optional[int] = None,
                      seed: Optional[int] = None) -> SystematicResult:
    """k = ⌊N/n⌋, arranque A ∈ [1, k], IDs: A, A+k, A+2k, ..."""
    if N < 1 or n < 1 or int(N) != N or int(n) != n:
        raise ValueError("N y n deben ser enteros positivos.")
    if n > N:
        raise ValueError("n no puede ser mayor que N.")
    k_exact = N / n
    k = int(math.floor(k_exact))
    if start is None:
        rng = np.random.default_rng(seed)
        start = int(rng.integers(1, k + 1))
        origin = f"aleatorio entre 1 y {k}" + (f" (semilla = {seed})" if seed is not None else "")
    else:
        if not 1 <= start <= k:
            raise ValueError(f"El arranque A debe estar entre 1 y k = {k}.")
        origin = "elegido manualmente"
    ids = [start + j * k for j in range(n)]
    steps = [
        Step("Intervalo de selección",
             rf"k = \frac{{N}}{{n}} = \frac{{{N}}}{{{n}}} = {fmt(k_exact)} \;\Rightarrow\; k = {k}"),
        Step("Arranque aleatorio", rf"A \in \{{1, \dots, {k}\}} \;\Rightarrow\; A = {start}",
             f"Arranque {origin}."),
        Step("Regla de selección", rf"ID_j = A + (j-1)\,k, \quad j = 1, \dots, {n}"),
        Step("Primeros elementos",
             rf"{start},\; {start}+{k}={start + k},\; {start}+2({k})={start + 2 * k},\;\dots,\; "
             rf"{start}+{n - 1}({k}) = {ids[-1]}" if n >= 3 else ", ".join(map(str, ids))),
    ]
    notes = []
    if k_exact != k:
        notes.append(f"N/n = {fmt(k_exact)} no es entero: se usa k = {k} (truncado). Los últimos "
                     f"{N - ids[-1]} elementos de la lista no pueden ser seleccionados; si eso "
                     "introduce sesgo, use el método circular.")
    notes.append("Cuidado con la periodicidad: si la lista tiene un patrón cíclico de longitud k, "
                 "la muestra puede quedar sesgada.")
    return SystematicResult(N=N, n=n, k=k, k_exact=k_exact, start=start, ids=ids,
                            steps=steps, notes=notes)
