# Copyright (c) 2025 Will Zhang

import numpy as np
from arraystubs import Arr1
from pytools.logging.trait import NULL_LOG, ILogger
from pytools.progress import ProgressBar

__all__ = ["opt_index"]


def interp_norm[F: np.floating, I: np.integer](
    data: Arr1[F],
    index: Arr1[I],
    skip: int = 25,
) -> float:
    y = np.interp(np.arange(0, len(data), skip), index, data[index])
    res = data[::skip] - y
    return float(res @ res)


def optimize_i[F: np.floating, I: np.integer](
    data: Arr1[F],
    index: Arr1[I],
    position: int,
    windows: int,
) -> Arr1[I]:
    diff = np.zeros((2 * windows + 1, index.size), dtype=index.dtype)
    diff[:, position] = np.arange(-windows, windows + 1, dtype=index.dtype)
    pars = diff + index
    fit = np.array([interp_norm(data, p) for p in pars])
    return pars[fit.argmin()]


def optimize[F: np.floating, I: np.integer](
    data: Arr1[F],
    index: Arr1[I],
    windows: int,
) -> Arr1[I]:
    bart = ProgressBar(n=index.size - 2)
    for i in range(1, index.size - 1):
        index = optimize_i(data, index, i, windows)
        bart.next()
    return index


def opt_index[F: np.floating, I: np.integer](
    data: Arr1[F],
    index: Arr1[I],
    window: int,
    *,
    max_iter: int = 100,
    log: ILogger = NULL_LOG,
) -> Arr1[I]:
    old_index = index.copy()
    old_index[-1] = index[-1] - 1
    for i in range(max_iter):
        new_index = optimize(data, old_index, window)
        diff = np.abs(new_index - old_index)
        log.disp(f"Iteration {i}: {diff.sum()}")
        if np.array_equal(new_index, old_index):
            break
        old_index = new_index
        window = window - 1 if window > 1 else 1
    old_index[-1] = old_index[-1] + 1
    return old_index
