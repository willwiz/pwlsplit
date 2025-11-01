from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Okay
from scipy.signal import find_peaks

from pwlsplit.plot import plot_segmentation_part
from pwlsplit.trait import Point

if TYPE_CHECKING:
    from pathlib import Path

    from pwlsplit.struct import PreppedData, Segmentation


def _find_next_split_peakpoint[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Okay[int] | Err:
    section = data.ddy[sequence.idx[i - 1] :]
    peaks, _ = find_peaks(np.maximum(section, 0), prominence=0.2, height=0.1)
    if len(peaks) == 0:
        msg = "No peak point found."
        return Err(ValueError(msg))
    return Okay(peaks[0])


def _find_next_split_valleypoint[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Okay[int] | Err:
    section = data.ddy[sequence.idx[i - 1] :]
    valleys, _ = find_peaks(np.maximum(-section, 0), prominence=0.2, height=0.1)
    if len(valleys) == 0:
        msg = "No valley point found."
        return Err(ValueError(msg))
    return Okay(valleys[0])


def find_next_split_point[F: np.floating, I: np.integer](
    data: PreppedData[F], sequence: Segmentation[F, I], i: int
) -> Okay[int] | Err:
    # print("len:", len(data.ddy[sequence.idx[i - 1] :]))
    match sequence.points[i]:
        case Point.PEAK:
            return _find_next_split_peakpoint(data, sequence, i)
        case Point.VALLEY:
            return _find_next_split_valleypoint(data, sequence, i)
        case Point.START:
            return Okay(0)
        case Point.END:
            return Okay(data.n - 1 - sequence.idx[i - 1])


def adjust_segmentation[F: np.floating, I: np.integer](
    data: PreppedData[F], segmentation: Segmentation[F, I], fout: Path | None = None
) -> Segmentation[F, I]:
    for prot, prot_vals in segmentation.prot.items():
        print(f"Adjusting protocol: {prot}")
        for test_vals in prot_vals.values():
            for k in test_vals:
                if k > 0 and k < segmentation.n:
                    match find_next_split_point(data, segmentation, k):
                        case Okay(i):
                            segmentation.idx[k:] = segmentation.idx[k:] + i
                        case Err(e):
                            raise e
        if fout is not None:
            plot_segmentation_part(
                data,
                segmentation,
                prot,
                fout=(fout.parent / (fout.stem + f"_{prot}")).with_suffix(".jpg"),
            )
    return segmentation
