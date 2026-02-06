from typing import TYPE_CHECKING

import numpy as np
from pytools.result import Err, Ok, Result, all_ok
from scipy.ndimage import gaussian_filter1d

from ._types import Curve, Hold, PreppedData, Recover, SegmentType, Stretch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pytools.arrays import A1


def prep_data[F: np.floating](x: A1[F]) -> PreppedData[F]:
    """Prepare raw data for segmentation.

    This function computes the smoothed data, first and second derivatives,
    and normalizes the derivatives.

    Args:
        x: Raw input data.

    Returns:
        PreppedData containing the original data, smoothed data, and normalized derivatives.

    """
    y = gaussian_filter1d(x, sigma=20)
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    return PreppedData(n=len(x), x=x, y=y, dy=dy / dy.max(), ddy=ddy / ddy.max())


def _parse_hold(data: Mapping[str, object]) -> Result[Hold]:
    duration = data.get("duration", 1.0)
    if not isinstance(duration, (int, float)):
        return Err(ValueError(f"Invalid duration for HOLD: {duration}"))
    return Ok(Hold(duration=duration))


def _parse_stretch(data: Mapping[str, object]) -> Result[Stretch]:
    delta = data.get("delta")
    duration = data.get("duration")
    if not isinstance(delta, (int, float)):
        return Err(ValueError(f"Invalid delta for STRETCH: {delta}"))
    if not isinstance(duration, (int, float)):
        return Err(ValueError(f"Invalid duration for STRETCH: {duration}"))
    return Ok(Stretch(delta=delta, duration=duration))


def _parse_recover(data: Mapping[str, object]) -> Result[Recover]:
    delta = data.get("delta")
    duration = data.get("duration")
    if not isinstance(delta, (int, float)):
        return Err(ValueError(f"Invalid delta for RECOVER: {delta}"))
    if not isinstance(duration, (int, float)):
        return Err(ValueError(f"Invalid duration for RECOVER: {duration}"))
    return Ok(Recover(delta=delta, duration=duration))


def parse_curve(data: Mapping[str, object] | SegmentType) -> Result[SegmentType]:
    """Parse a segment dictionary into a SegmentType.

    Args:
        data: A dictionary containing the curve type and its parameters.

    Returns:
        A SegmentType instance corresponding to the input data.

    Raises:
        ValueError: If the curve type is unknown or if required parameters are missing.

    """
    if isinstance(data, (Hold, Stretch, Recover)):
        return Ok(data)
    match data.get("curve"):
        case "HOLD":
            return _parse_hold(data).next()
        case "STRETCH":
            return _parse_stretch(data).next()
        case "RECOVER":
            return _parse_recover(data).next()
        case None:
            return Err(ValueError("Missing 'curve' key in segment dictionary."))
        case c:
            return Err(ValueError(f"Unknown curve type: {c}"))


def parse_curves(
    data: Sequence[Mapping[str, object]] | Sequence[SegmentType],
) -> Result[Sequence[SegmentType]]:
    match all_ok([parse_curve(d) for d in data]):
        case Ok(curves):
            return Ok(curves)
        case Err(e):
            return Err(e)


def curve_type(curve: SegmentType) -> Curve:
    match curve:
        case Hold():
            return Curve.HOLD
        case Stretch():
            return Curve.STRETCH
        case Recover():
            return Curve.RECOVER
