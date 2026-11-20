"""Non-linear complexity measures computed on each beat segment."""
from math import factorial, log

import numpy as np


def sample_entropy(series, m, r):
    n = len(series)
    if n < m + 1:
        return 0

    def _phi(m):
        x = np.array([series[i:i + m] for i in range(n - m + 1)])
        c = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) < r, axis=0) - 1
        return np.sum(c) / (n - m + 1)

    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except Exception:
        return 0


def permutation_entropy(series, m):
    n = len(series)
    if n < m:
        return 0
    patterns = {}
    for i in range(n - m + 1):
        key = tuple(np.argsort(series[i:i + m]))
        patterns[key] = patterns.get(key, 0) + 1
    probs = np.array(list(patterns.values()), dtype=float)
    probs /= probs.sum()
    return -np.sum(probs * np.log(probs)) / log(factorial(m))


def approximate_entropy(series, m, r):
    n = len(series)
    if n <= m + 1:
        return 0

    def _phi(m):
        x = np.array([series[i:i + m] for i in range(n - m + 1)])
        c = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (n - m + 1)
        return np.sum(np.log(c)) / (n - m + 1)

    try:
        return _phi(m) - _phi(m + 1)
    except Exception:
        return 0
