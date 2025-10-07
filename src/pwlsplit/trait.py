import enum


class Curve(enum.StrEnum):
    STRETCH = "STRETCH"
    HOLD = "HOLD"
    RECOVER = "RECOVER"


class Point(enum.StrEnum):
    START = "START"
    END = "END"
    PEAK = "PEAK"
    VALLEY = "VALLEY"
