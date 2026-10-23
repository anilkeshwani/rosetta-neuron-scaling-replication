import numpy as np
import pytest

from rosetta_repl.scaling import fit_power_law


def test_recovers_exact_power_law():
    x = np.array([4e4, 1e5, 2e5, 5e5, 1e6])
    fit = fit_power_law(x, 2.20 * x**0.68)
    assert fit.alpha == pytest.approx(0.68)
    assert fit.c == pytest.approx(2.20)
    assert fit.r2 == pytest.approx(1.0)


def test_noisy_fit_has_finite_standard_error():
    rng = np.random.default_rng(0)
    x = np.geomspace(4e4, 2e6, 8)
    y = 12.43 * x**0.57 * np.exp(rng.normal(0, 0.05, size=x.shape))
    fit = fit_power_law(x, y)
    assert abs(fit.alpha - 0.57) < 3 * fit.alpha_se + 0.05
    assert 0.9 < fit.r2 <= 1.0


def test_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        fit_power_law([1e4, 1e5], [0, 10])


def test_rejects_single_distinct_x():
    with pytest.raises(ValueError, match="distinct"):
        fit_power_law([1e5, 1e5, 1e5], [10, 12, 11])
