from typing import TYPE_CHECKING, Literal, TypeIs

from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Mapping

_SEGMENT_VALID: Mapping[Literal["STRETCH", "HOLD", "RECOVER"], set[str]] = {
    "STRETCH": {"delta", "time"},
    "HOLD": set(),
    "RECOVER": {"delta", "time"},
}


def _valid_curve(val: object) -> TypeIs[Literal["STRETCH", "HOLD", "RECOVER"]]:
    return val in _SEGMENT_VALID


def _is_valid_dict(dct: object) -> TypeIs[Mapping[str, object]]:
    return isinstance(dct, dict)


def validate_curve_segment(
    dct: object,
) -> Ok[None] | Err:
    if not _is_valid_dict(dct):
        return Err(TypeError(f"Segment must be a dictionary not {type(dct)}."))
    match dct.get("curve"):
        case None:
            return Err(LookupError("Missing 'curve' in segment dictionary."))
        case curve_name:
            pass
    if not _valid_curve(curve_name):
        msg = f"Invalid curve type '{curve_name}'."
        return Err(LookupError(msg))
    msg = ""
    for attr in _SEGMENT_VALID[curve_name]:
        match dct.get(attr):
            case None:
                msg += f"Missing required attribute '{attr}' for curve type '{curve_name}'.\n"

            case val:
                if isinstance(val, (int, float)):
                    continue
                msg += f"Attribute '{attr}' for curve type '{curve_name}' must be a number.\n"
                return Err(TypeError(msg))
    if msg:
        return Err(LookupError(msg))
    return Ok(None)
