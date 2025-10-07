import dataclasses as dc
from collections.abc import Mapping, Sequence
from typing import Literal, Required, TypedDict

import numpy as np
from arraystubs import Arr1

from .trait import Curve, Point


@dc.dataclass(slots=True)
class Segment:
    c: Curve
    t: float
    d: float


class SegmentDict(TypedDict, total=False):
    curve: Required[Literal["STRETCH", "HOLD", "RECOVER"]]
    delta: float
    duration: float


# {protocol: {curve: {segment: [parameters]}}}
type TestProtocol = Mapping[
    str,
    Mapping[
        str,
        Sequence[SegmentDict],
    ],
]


@dc.dataclass(slots=True)
class Segmentation[F: np.floating, I: np.integer]:
    prot: Mapping[str, Mapping[str, Sequence[int]]]
    n: int
    curves: Sequence[Curve]
    points: Sequence[Point]
    idx: Arr1[I]
    peaks: Arr1[F]


@dc.dataclass(slots=True)
class PreppedData[F: np.floating]:
    n: int
    x: Arr1[F]
    y: Arr1[F]
    dy: Arr1[F]
    ddy: Arr1[F]
