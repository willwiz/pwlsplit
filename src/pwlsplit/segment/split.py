from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok
from scipy.signal import find_peaks

from pwlsplit.trait import Point, PreppedData, Segmentation

if TYPE_CHECKING:
    from collections.abc import Sequence


def _find_next_split_peakpoint[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Ok[int] | Err:
    section = data.ddy[sequence.idx[i - 1] :] / abs(sequence.peaks[i])
    peaks, _ = find_peaks(np.maximum(section, 0), prominence=0.2, height=0.1)
    if len(peaks) == 0:
        msg = "No peak point found."
        return Err(ValueError(msg))
    return Ok(peaks[0])


def _find_next_split_valleypoint[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Ok[int] | Err:
    section = data.ddy[sequence.idx[i - 1] :] / abs(sequence.peaks[i])
    valleys, _ = find_peaks(np.maximum(-section, 0), prominence=0.2, height=0.1)
    if len(valleys) == 0:
        msg = "No valley point found."
        return Err(ValueError(msg))
    return Ok(valleys[0])


def find_next_split_point[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Ok[int] | Err:
    match sequence.points[i]:
        case Point.PEAK:
            return _find_next_split_peakpoint(data, sequence, i)
        case Point.VALLEY:
            return _find_next_split_valleypoint(data, sequence, i)
        case Point.START:
            return Ok(0)
        case Point.END:
            return Ok(data.n - 1 - sequence.idx[i - 1])


def adjust_segmentation[F: np.floating, I: np.integer](
    data: PreppedData[F],
    segmentation: Segmentation[F, I],
    indices: Sequence[int],
) -> Ok[Segmentation[F, I]] | Err:
    for k in indices:
        if k > 0 and k <= segmentation.n_point:
            match find_next_split_point(data, segmentation, k):
                case Ok(i):
                    segmentation.idx[k:-1] = segmentation.idx[k:-1] + i
                case Err(e):
                    return Err(e)
    return Ok(segmentation)
