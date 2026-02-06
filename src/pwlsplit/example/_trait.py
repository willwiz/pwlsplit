from collections.abc import Mapping, Sequence

from pwlsplit.types import SegmentDict

# {protocol: {curve: {segment: [parameters]}}}
TestProtocol = Mapping[
    str,
    Mapping[
        str,
        Sequence[SegmentDict],
    ],
]


ProtocolMap = Mapping[str, Mapping[str, Sequence[int]]]
