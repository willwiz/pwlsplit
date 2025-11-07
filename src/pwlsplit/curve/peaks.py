import itertools
from typing import TYPE_CHECKING

import numpy as np

from pwlsplit.struct import FinalSegmentation, Segment, TestProtocol
from pwlsplit.trait import Curve, Point

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arraystubs import Arr1


def _estimate_peak(left: Segment, right: Segment) -> tuple[Point, float]:
    match (left, right):
        case Segment(c=Curve.HOLD), Segment(c=Curve.STRETCH, t=t, d=d):
            return (Point.PEAK, abs(d / t))
        case Segment(c=Curve.HOLD), Segment(c=Curve.RECOVER, t=t, d=d):
            return (Point.VALLEY, abs(d / t))
        case Segment(c=Curve.STRETCH, t=t, d=d), Segment(c=Curve.HOLD):
            return (Point.VALLEY, abs(d / t))
        case Segment(c=Curve.RECOVER, t=t, d=d), Segment(c=Curve.HOLD):
            return (Point.PEAK, abs(d / t))
        case Segment(c=Curve.STRETCH, t=tl, d=dl), Segment(c=Curve.RECOVER, t=tr, d=dr):
            return (Point.VALLEY, abs(dr / tr - dl / tl))
        case Segment(c=Curve.RECOVER, t=tl, d=dl), Segment(c=Curve.STRETCH, t=tr, d=dr):
            return (Point.PEAK, abs(dr / tr - dl / tl))
        case Segment(c=Curve.HOLD), Segment(c=Curve.HOLD):
            msg = "Consecutive 'Hold' segments found."
            raise ValueError(msg)
        case Segment(c=Curve.STRETCH), Segment(c=Curve.STRETCH):
            msg = "Consecutive 'Stretch' segments found."
            raise ValueError(msg)
        case Segment(c=Curve.RECOVER), Segment(c=Curve.RECOVER):
            msg = "Consecutive 'Recover' segments found."
            raise ValueError(msg)
        case _:
            msg = f"Unreachable branch occured to pair: {left}, {right}. DEBUG needed."
            raise ValueError(msg)


def estimate_peaks(curves: Sequence[Segment]) -> tuple[Sequence[Point], Arr1[np.float64]]:
    segments = [_estimate_peak(left, right) for left, right in itertools.pairwise(curves)]
    points = [Point.START, *[s[0] for s in segments], Point.END]
    peaks = np.array([s[1] for s in segments], dtype=np.float64)
    heights = np.array([0.0, *(peaks / peaks.max()), 0.0], dtype=np.float64)
    return points, heights


def construct_segmentation(test: TestProtocol) -> FinalSegmentation[np.float64, np.intp]:
    curves = [
        Curve[s["curve"]] for p_vals in test.values() for c_vals in p_vals.values() for s in c_vals
    ]
    segments = [
        Segment(c=Curve(s["curve"]), t=s.get("duration", 0.0), d=s.get("delta", 0.0))
        for p_vals in test.values()
        for c_vals in p_vals.values()
        for s in c_vals
    ]
    prot_map: dict[str, dict[str, list[int]]] = {p: {c: [] for c in test[p]} for p in test}
    k = 0
    for prot, vals in test.items():
        for name, segs in vals.items():
            prot_map[prot][name] = [k := k + 1 for _ in range(len(segs))]
    points, peaks = estimate_peaks(segments)
    return FinalSegmentation(
        prot=prot_map,
        n=len(points),
        curves=curves,
        points=points,
        idx=np.zeros(len(points), dtype=np.intp),
        peaks=peaks,
    )
