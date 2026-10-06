"""
core/data_loader.py
===================
Lectura de archivos CSV/Excel y cálculo de estadísticos resumidos.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """Lee CSV (separador autodetectado: , ; tab) o Excel a un DataFrame."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith((".csv", ".txt")):
        text = content.decode("utf-8-sig", errors="replace")
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    raise ValueError("Formato no soportado. Use .csv, .xlsx o .xls.")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def numeric_summary(series: pd.Series) -> dict:
    """Media, desviación estándar muestral (ddof=1) y n, ignorando vacíos."""
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 2:
        raise ValueError(f"La columna «{series.name}» necesita al menos 2 valores numéricos.")
    return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "n": int(x.size)}


def proportion_summary(series: pd.Series, success) -> dict:
    """Cuenta de éxitos x (valores iguales a `success`) sobre n no vacíos."""
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        raise ValueError(f"La columna «{series.name}» está vacía.")
    x = int((s == str(success).strip()).sum())
    return {"x": x, "n": int(s.size), "p_hat": x / s.size}
