# Copyright (c) 2025 Will Zhang

from pprint import pformat
from typing import TYPE_CHECKING

import numpy as np
from pytools.logging import NLOGGER, ILogger
from pytools.progress import ProgressBar

if TYPE_CHECKING:
    from pytools.arrays import A1

__all__ = ["opt_index"]


def interp_norm[F: np.floating, I: np.integer](
    data: A1[F],
    index: A1[I],
    skip: int = 25,
) -> float:
    y = np.interp(np.arange(0, len(data), skip), index, data[index])
    res = data[::skip] - y
    return float(res @ res)


def optimize_i[F: np.floating, I: np.integer](
    data: A1[F],
    index: A1[I],
    position: int,
    windows: int,
) -> A1[I]:
    diff = np.zeros((2 * windows + 1, index.size), dtype=index.dtype)
    diff[:, position] = np.arange(-windows, windows + 1, dtype=index.dtype)
    pars = diff + index
    fit = np.array([interp_norm(data, p) for p in pars])
    return pars[fit.argmin()]


_MINIMUM_SIZE = 2


def optimize[F: np.floating, I: np.integer](
    data: A1[F],
    index: A1[I],
    windows: int,
) -> A1[I]:
    if index.size <= _MINIMUM_SIZE:
        return index
    bart = ProgressBar(n=index.size - 2)
    for i in range(1, index.size - 1):
        index = optimize_i(data, index, i, windows)
        bart.next()
    return index


def opt_index[F: np.floating, I: np.integer](
    data: A1[F],
    index: A1[I],
    window: int,
    *,
    max_iter: int = 100,
    log: ILogger = NLOGGER,
) -> A1[I]:
    old_index = index.copy()
    old_index[-1] = len(data) - 1
    for i in range(max_iter):
        new_index = optimize(data, old_index, window)
        diff = np.abs(new_index - old_index)
        log.disp(f"Iteration {i}: {diff.sum()}")
        if np.array_equal(new_index, old_index):
            break
        log.debug(pformat(new_index))
        old_index = new_index
        window = window - 1 if window > 1 else 1
    old_index[-1] = len(data)
    return old_index
