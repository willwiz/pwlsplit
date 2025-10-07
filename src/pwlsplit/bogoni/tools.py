from pwlsplit.struct import SegmentDict, TestProtocol


def create_bogoni_curve(max_strain: float) -> TestProtocol:
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
