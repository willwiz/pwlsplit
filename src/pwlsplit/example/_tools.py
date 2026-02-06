from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pwlsplit.types import SegmentDict

    from ._trait import CurveIndex, TestProtocol


def create_bogoni_protocol(max_strain: float) -> TestProtocol:
    loading: dict[str, list[SegmentDict]] = {
        "step_0": [
            {"curve": "STRETCH", "delta": max_strain / 3, "duration": 4.0},
            {"curve": "HOLD"},
        ],
        **{
            f"step_{i}": [
                {"curve": "STRETCH", "delta": max_strain / 3 / 4, "duration": 1.0},
                {"curve": "HOLD"},
            ]
            for i in range(1, 9)
        },
    }
    unloading: dict[str, list[SegmentDict]] = {
        **{
            f"step_{i}": [
                {"curve": "RECOVER", "delta": -max_strain / 3 / 4, "duration": 1.0},
                {"curve": "HOLD"},
            ]
            for i in range(8)
        },
        "step_8": [
            {"curve": "RECOVER", "delta": -max_strain / 3, "duration": 4.0},
            {"curve": "HOLD"},
        ],
    }
    return {
        "FirstLoading": loading,
        "Unloading": unloading,
        "Reloading": loading,
        "Reset": {
            "step_0": [{"curve": "RECOVER", "delta": -max_strain, "duration": 12.0}],
        },
    }


def construct_bogoni_curves(
    protocol: TestProtocol,
) -> tuple[CurveIndex, list[SegmentDict]]:
    prot_map: dict[str, dict[str, list[int]]] = {p: {c: [] for c in v} for p, v in protocol.items()}
    k = 0
    for prot, vals in protocol.items():
        for name, segs in vals.items():
            prot_map[prot][name] = [k := k + 1 for _ in range(len(segs))]
    return prot_map, [s for p_val in protocol.values() for c_vals in p_val.values() for s in c_vals]
