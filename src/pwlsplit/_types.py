import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Literal, Required, TypedDict

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1


# class SegmentDict(TypedDict, total=False):
#     curve: Required[Literal["STRETCH", "HOLD", "RECOVER"]]
#     delta: float
#     time: float


class HoldDict(TypedDict, total=False):
    curve: Required[Literal["HOLD"]]
    duration: float


class StretchDict(TypedDict, total=False):
    curve: Required[Literal["STRETCH"]]
    delta: Required[float]
    duration: Required[float]


class RecoverDict(TypedDict, total=False):
    curve: Required[Literal["RECOVER"]]
    delta: Required[float]
    duration: Required[float]


SegmentDict = StretchDict | HoldDict | RecoverDict

CurveType = Literal["STRETCH", "HOLD", "RECOVER"]


class Curve(enum.StrEnum):
    STRETCH = "STRETCH"
    HOLD = "HOLD"
    RECOVER = "RECOVER"


class Point(enum.StrEnum):
    START = "START"
    END = "END"
    PEAK = "PEAK"
    VALLEY = "VALLEY"


@dc.dataclass(slots=True)
class Hold:
    duration: float

    @property
    def rate(self) -> float:
        return 0.0


@dc.dataclass(slots=True)
class Stretch:
    delta: float
    duration: float

    @property
    def rate(self) -> float:
        return abs(self.delta) / self.duration


@dc.dataclass(slots=True)
class Recover:
    delta: float
    duration: float

    @property
    def rate(self) -> float:
        return -abs(self.delta) / self.duration


SegmentType = Hold | Stretch | Recover


@dc.dataclass(slots=True)
class PreppedData[F: np.floating]:
    """Input for segmentation."""

    n: int
    x: A1[F]
    y: A1[F]
    dy: A1[F]
    ddy: A1[F]


@dc.dataclass(slots=True)
class Segmentation[F: np.floating, I: np.integer]:
    """Segmentation result.

    Need initial guess to be refined by the program.

    Attributes
    ----------
    n_point : int
        Number of break points.
    curves : Sequence[Curve]
        List of curve types for linear segments. Len = n_point - 1
    points : Sequence[Point]
        List of type points at the ends of the segments. Len = n_point
    idx : A1[I]
        Indices of the break points. Len = n_point
    peaks : A1[F]
        Estimated peak heights at the peak points. Len = n_point

    """

    n_point: int
    curves: Sequence[Curve]
    points: Sequence[Point]
    idx: A1[I]
    peaks: A1[F]
