from collections.abc import Mapping, Sequence

from pwlsplit.types import SegmentDict

TestProtocol = Mapping[
    str,
    Mapping[
        str,
        Sequence[SegmentDict],
    ],
]


CurveIndex = Mapping[str, Mapping[str, Sequence[int]]]
