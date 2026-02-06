import itertools
from typing import TYPE_CHECKING, Literal

import numpy as np
from pytools.result import Err, Ok

from pwlsplit.api import curve_type, parse_curves
from pwlsplit.types import Curve, Point, Segmentation, SegmentDict, SegmentType

from ._types import Segment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1


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
) -> Ok[tuple[Sequence[Point], A1[np.float64]]] | Err:
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
    if len(peaks):
        peaks = peaks / peaks.max()
    heights = np.array([1.0, *peaks, 1.0], dtype=np.float64)
    return Ok((points, heights))


def construct_initial_segmentation(
    data: Sequence[SegmentDict] | Sequence[SegmentType],
) -> Ok[Segmentation[np.float64, np.intp]] | Err:
    match parse_curves(data):
        case Err(e):
            return Err(e)
        case Ok(data):
            pass
    curves = [curve_type(s) for s in data]
    rate = [d.rate for d in data]
    segments = [Segment(c=c, r=r) for c, r in zip(curves, rate, strict=True)]
    initial_index = np.arange(len(segments), dtype=np.intp)
    match estimate_peaks(segments):
        case Err(e):
            return Err(e)
        case Ok((points, peaks)):
            pass
    return Ok(
        Segmentation(
            n_point=len(points), curves=curves, points=points, idx=initial_index, peaks=peaks
        )
    )
