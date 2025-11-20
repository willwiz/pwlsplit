import itertools
from typing import TYPE_CHECKING, Literal

import numpy as np
from pytools.result import Err, Ok

from pwlsplit.trait import (
    Curve,
    Point,
    Segment,
    Segmentation,
    SegmentDict,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arraystubs import Arr1


_BREAKPOINT_MAPPING: Mapping[tuple[Curve, Curve], Literal[Point.PEAK, Point.VALLEY] | None] = {
    (Curve.HOLD, Curve.STRETCH): Point.PEAK,
    (Curve.HOLD, Curve.RECOVER): Point.VALLEY,
    (Curve.STRETCH, Curve.HOLD): Point.VALLEY,
    (Curve.STRETCH, Curve.RECOVER): Point.VALLEY,
    (Curve.RECOVER, Curve.HOLD): Point.PEAK,
    (Curve.RECOVER, Curve.STRETCH): Point.PEAK,
    # Other combinations are invalid
}


def _estimate_peak(left: Segment, right: Segment) -> Ok[tuple[Point, float]] | Err:
    match _BREAKPOINT_MAPPING.get((left.c, right.c), None):
        case None:
            msg = f"Invalid segment pair: {left}, {right}."
            return Err(ValueError(msg))
        case Point.PEAK as p:
            return Ok((p, abs(right.r) + abs(left.r)))
        case Point.VALLEY as p:
            return Ok((p, -abs(right.r) - abs(left.r)))


def estimate_peaks(
    curves: Sequence[Segment],
) -> Ok[tuple[Sequence[Point], Arr1[np.float64]]] | Err:
    results = [_estimate_peak(left, right) for left, right in itertools.pairwise(curves)]
    errors = [res for res in results if isinstance(res, Err)]
    if len(errors) > 0:
        msg = f"{len(errors)} errors occurred during peak estimation."
        for e in errors:
            msg += f"\n - {e.val}"
        return Err(ValueError(msg))
    segments = [res.val for res in results if isinstance(res, Ok)]
    points = [Point.START, *[s[0] for s in segments], Point.END]
    peaks = np.array([s[1] for s in segments], dtype=np.float64)
    heights = np.array([1.0, *(peaks / peaks.max(initial=1.0)), 1.0], dtype=np.float64)
    return Ok((points, heights))


def construct_initial_segmentation(
    data: Sequence[SegmentDict],
) -> Ok[Segmentation[np.float64, np.intp]] | Err:
    curves = [Curve[s["curve"]] for s in data]
    duration = [s.get("time", 1.0) for s in data]
    if not all(d > 0.0 for d in duration):
        msg = "All segment durations must be positive."
        return Err(ValueError(msg))
    rate = [
        0 if c is Curve.HOLD else s.get("delta", 0.0) / t
        for c, s, t in zip(curves, data, duration, strict=True)
    ]
    segments = [Segment(c=c, r=r) for c, r in zip(curves, rate, strict=True)]
    match estimate_peaks(segments):
        case Err(e):
            return Err(e)
        case Ok((points, peaks)):
            return Ok(
                Segmentation(
                    n_point=len(points),
                    curves=curves,
                    points=points,
                    idx=np.zeros(len(points), dtype=np.intp),
                    peaks=peaks,
                )
            )
