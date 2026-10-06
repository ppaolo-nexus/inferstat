"""Pruebas unitarias del núcleo matemático (ejecutar con: pytest -q)."""
import math

import pytest
from scipy import stats

from core import hypothesis as h
from core import sampling as s


# ---------------- Muestreo ----------------
def test_sample_size_mean_infinite():
    assert s.sample_size_mean(10, 2, 0.95).n == 97          # (1.96·10/2)² = 96.04


def test_sample_size_proportion_infinite_and_finite():
    assert s.sample_size_proportion(0.5, 0.05, 0.95).n == 385
    assert s.sample_size_proportion(0.5, 0.05, 0.95, N=1000).n == 278


def test_finite_formula_equals_correction():
    r = s.sample_size_mean(15, 3, 0.99, N=500)
    n0 = r.n0
    assert math.isclose(r.n_raw, n0 / (1 + (n0 - 1) / 500))


def test_stratified_sum_and_proportions():
    r = s.stratified_proportional(100, ["a", "b", "c"], [333, 333, 334])
    assert sum(row["n_i asignado"] for row in r.rows) == 100


def test_systematic():
    r = s.systematic_sample(1000, 50, start=7)
    assert r.k == 20 and r.ids[0] == 7 and r.ids[-1] == 7 + 49 * 20 and len(r.ids) == 50
    with pytest.raises(ValueError):
        s.systematic_sample(1000, 50, start=21)


# ---------------- Hipótesis ----------------
def test_z_one_mean():
    r = h.z_one_mean(52, 50, 8, 40, 0.05, "two")
    assert math.isclose(r.statistic, 1.5811388, rel_tol=1e-6)
    assert math.isclose(r.p_value, 0.1138463, rel_tol=1e-5)
    assert not r.reject


def test_t_one_mean_matches_scipy():
    r = h.t_one_mean(53, 50, 6, 15, 0.05, "right")
    t = (53 - 50) / (6 / math.sqrt(15))
    assert math.isclose(r.statistic, t) and math.isclose(r.p_value, stats.t.sf(t, 14))


@pytest.mark.parametrize("equal", [True, False])
def test_t_two_means_matches_scipy(equal):
    r = h.t_two_means(78, 6, 12, 72.5, 7, 14, 0, 0.05, "two", equal_var=equal)
    ref = stats.ttest_ind_from_stats(78, 6, 12, 72.5, 7, 14, equal_var=equal)
    assert math.isclose(r.statistic, ref.statistic, rel_tol=1e-9)
    assert math.isclose(r.p_value, ref.pvalue, rel_tol=1e-9)


def test_props():
    r = h.z_one_prop(58, 100, 0.5, 0.05, "right")
    assert math.isclose(r.statistic, 1.6) and not r.reject  # p = 0.0548
    r2 = h.z_two_props(45, 120, 30, 110)
    assert r2.statistic > 0


def test_chi2_variance():
    r = h.chi2_variance(6, 20, 25, 0.05, "two")
    assert math.isclose(r.statistic, 19 * 36 / 25)
    lo, hi = r.critical
    assert math.isclose(lo, stats.chi2.ppf(0.025, 19)) and math.isclose(hi, stats.chi2.ppf(0.975, 19))


def test_decision_consistency():
    for tail in ("two", "left", "right"):
        r = h.z_one_mean(47, 50, 5, 36, 0.05, tail)
        assert r.reject == r.in_rejection_region
