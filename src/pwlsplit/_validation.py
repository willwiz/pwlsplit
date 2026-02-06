# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeIs

if TYPE_CHECKING:
    from ._trait import SegmentDict


def is_segment_dict(obj: object) -> TypeIs[SegmentDict]:
    if not isinstance(obj, Mapping):
        return False
    match obj.get("curve", None):
        case "HOLD":
            return True
        case "STRETCH" | "RECOVER":
            required_vars = ["delta", "time"]
        case _:
            return False
    for var in required_vars:
        match obj.get(var):
            case int() | float():
                continue
            case _:
                return False
    return True
