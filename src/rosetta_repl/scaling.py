"""Power-law fits |R| = c * x^alpha by OLS in log-log space, as in Section 4.1 of the paper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    c: float
    alpha: float
    r2: float
    alpha_se: float
    n_points: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.c * np.asarray(x, dtype=float) ** self.alpha

    def to_dict(self) -> dict:
        return asdict(self)


def fit_power_law(x, y) -> PowerLawFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if len(x) < 2:
        raise ValueError("need at least two points to fit")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit needs strictly positive x and y")
    if len(np.unique(x)) < 2:
        raise ValueError("need at least two distinct x values to fit")

    lx, ly = np.log(x), np.log(y)
    A = np.stack([np.ones_like(lx), lx], axis=1)
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    log_c, alpha = coef
    resid = ly - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = len(x)
    if n > 2:
        sigma2 = ss_res / (n - 2)
        alpha_se = float(np.sqrt(sigma2 / ((lx - lx.mean()) ** 2).sum()))
    else:
        alpha_se = float("nan")
    return PowerLawFit(c=float(np.exp(log_c)), alpha=float(alpha), r2=r2, alpha_se=alpha_se, n_points=n)
