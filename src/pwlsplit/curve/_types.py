import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pwlsplit.types import Curve


@dc.dataclass(slots=True)
class Segment:
    c: Curve
    r: float
