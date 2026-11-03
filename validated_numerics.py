"""Enclosures for the exact-real interpretation of saved binary parameters.

Basic operations use separate NumPy float64 ufuncs (no BLAS reductions/FMA)
and move one representable number outwards after every rounded operation.
Exact zero/identity cases are retained. Transcendentals use MPFR with directed
rounding and IEEE binary64 subnormal handling. No guarantee about a particular
PyTorch/CUDA floating-point execution is inferred from these enclosures.
"""
from __future__ import annotations

from functools import lru_cache
import math

import gmpy2
import numpy as np

SEMANTICS = "exact-real-saved-binary-parameters"
NUMERICS = "outward-float64-ufuncs-and-mpfr"


def _binary(a, b, operation, upper: bool):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite interval operand")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = operation(a, b)
        bound = np.nextafter(value, np.inf if upper else -np.inf)
    if not np.isfinite(bound).all():
        raise ValueError("interval arithmetic overflow")
    if operation is np.add:
        exact = (a == 0) | (b == 0) | (a == -b)
    else:
        exact = (a == 0) | (b == 0) | (np.abs(a) == 1) | (np.abs(b) == 1)
    result = np.where(exact, value, bound)
    return float(result) if result.ndim == 0 else result


def add_down(a, b):
    return _binary(a, b, np.add, False)


def add_up(a, b):
    return _binary(a, b, np.add, True)


def mul_down(a, b):
    return _binary(a, b, np.multiply, False)


def mul_up(a, b):
    return _binary(a, b, np.multiply, True)


def scale(lo, hi, weight):
    if weight >= 0:
        return mul_down(lo, weight), mul_up(hi, weight)
    return mul_down(hi, weight), mul_up(lo, weight)


def affine(lo, hi, weights, bias):
    """Sparse, separately rounded affine transfer; weights have shape [out,in]."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    weights = np.asarray(weights, dtype=float)
    vector = weights.ndim == 1
    weights = np.atleast_2d(weights)
    bias = np.broadcast_to(np.asarray(bias, dtype=float), (weights.shape[0],))
    if lo.shape != hi.shape or lo.shape[-1] != weights.shape[1]:
        raise ValueError("invalid affine shapes")
    if not all(np.isfinite(v).all() for v in (lo, hi, weights, bias)):
        raise ValueError("nonfinite affine input")
    if np.any(lo > hi):
        raise ValueError("reversed interval")
    shape = lo.shape[:-1] + (weights.shape[0],)
    lower, upper = np.broadcast_to(bias, shape).copy(), np.broadcast_to(bias, shape).copy()
    for row in range(weights.shape[0]):
        for col in np.flatnonzero(weights[row]):
            a, b = scale(lo[..., col], hi[..., col], weights[row, col])
            lower[..., row] = add_down(lower[..., row], a)
            upper[..., row] = add_up(upper[..., row], b)
    if vector:
        return lower[..., 0], upper[..., 0]
    return lower, upper


@lru_cache(maxsize=131072)
def _transcendental(x: float, name: str, upper: bool) -> float:
    if not math.isfinite(x):
        raise ValueError("nonfinite transcendental input")
    # A fresh context prevents caller precision/rounding from changing a proof.
    down = gmpy2.ieee(64)
    down.round = gmpy2.RoundDown
    up = gmpy2.ieee(64)
    up.round = gmpy2.RoundUp
    if name == "tanh":
        with (up if upper else down):
            return float(gmpy2.tanh(gmpy2.mpfr(x)))
    if name != "sigmoid":
        raise ValueError(name)
    # Stable formulas only exponentiate a nonpositive argument. Every operation
    # has its own enclosing direction, including the denominator.
    with down:
        e_lo = gmpy2.exp(gmpy2.mpfr(-abs(x)))
        den_lo = gmpy2.add(1, e_lo)
    with up:
        e_hi = gmpy2.exp(gmpy2.mpfr(-abs(x)))
        den_hi = gmpy2.add(1, e_hi)
    with (up if upper else down):
        if x >= 0:
            result = gmpy2.div(1, den_lo if upper else den_hi)
        else:
            result = gmpy2.div(e_hi if upper else e_lo, den_lo if upper else den_hi)
        return float(result)


def monotone(lo, hi, name: str):
    def evaluate(values, upper):
        values = np.asarray(values, dtype=float)
        unique, inverse = np.unique(values, return_inverse=True)
        bounds = np.array([_transcendental(float(x), name, upper) for x in unique])
        result = bounds[inverse].reshape(values.shape)
        return float(result) if result.ndim == 0 else result
    if np.any(np.asarray(lo) > np.asarray(hi)):
        raise ValueError("reversed interval")
    return evaluate(lo, False), evaluate(hi, True)


def sigmoid(lo, hi):
    return monotone(lo, hi, "sigmoid")


def tanh(lo, hi):
    return monotone(lo, hi, "tanh")


def step(lo, hi):
    return np.where(np.asarray(lo) >= 0, 1.0, -1.0), np.where(np.asarray(hi) >= 0, 1.0, -1.0)


def metadata():
    return {"semantics": SEMANTICS, "numerics": NUMERICS,
            "gmpy2": gmpy2.version(), "mpfr": gmpy2.mpfr_version(),
            "numpy": np.__version__, "precision": 53,
            "floating_point_runtime_certified": False}
