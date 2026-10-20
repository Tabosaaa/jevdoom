"""Reads a Doom map straight from the WAD and classifies the special lines that
matter for finishing a level: exits, doors (with and without keys), switches,
walk-over triggers, lifts and teleporters. Used by level.py (geometry) and
objectives.py (what is left to do to reach the exit)."""

import struct
from dataclasses import dataclass

# Doom/Doom II "special" numbers (linedef types).
EXITS_USE = frozenset({11, 51})
EXITS_WALK = frozenset({52, 124})
MANUAL_DOORS = frozenset({1, 26, 27, 28, 31, 32, 33, 34, 117, 118})
LOCKED_SWITCHES = {99: "blue", 133: "blue", 134: "red", 135: "red", 136: "yellow", 137: "yellow"}
DOOR_KEYS = {26: "blue", 32: "blue", 27: "yellow", 34: "yellow", 28: "red", 33: "red", **LOCKED_SWITCHES}
SWITCHES = frozenset({
    7, 9, 14, 15, 18, 20, 21, 23, 29, 41, 42, 43, 45, 49, 50, 55, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 101, 102, 103, 111, 112, 113, 114, 115, 116, 122, 123, 127,
    131, 132, 138, 139, 140, *LOCKED_SWITCHES,
})
TELEPORTS = frozenset({39, 97})  # 125/126 only carry monsters
LIFTS = frozenset({10, 21, 62, 88, 120, 121, 122, 123})
NON_TRIGGERS = frozenset({48, 85}) | TELEPORTS | {125, 126}  # texture scrolling etc.

KEY_THINGS = {5: "blue", 40: "blue", 6: "yellow", 39: "yellow", 13: "red", 38: "red"}
KEY_NAMES = {"BlueCard": "blue", "BlueSkull": "blue", "YellowCard": "yellow",
             "YellowSkull": "yellow", "RedCard": "red", "RedSkull": "red"}
TELEPORT_DEST = 14
PLAYER_START = 1


@dataclass(frozen=True)
class Line:
    a: tuple[int, int]
    b: tuple[int, int]
    flags: int
    special: int
    tag: int
    front: int  # sector
    back: int   # sector, -1 if the line is one-sided

    @property
    def blocking(self) -> bool:
        return self.back == -1 or bool(self.flags & 1)

    @property
    def is_use(self) -> bool:
        return self.special in SWITCHES or self.special in MANUAL_DOORS or self.special in EXITS_USE

    @property
    def is_walk_trigger(self) -> bool:
        return (self.special != 0 and not self.is_use and self.special not in NON_TRIGGERS
                and self.special not in EXITS_WALK and self.tag != 0)


@dataclass(frozen=True)
class Thing:
    x: int
    y: int
    angle: int
    kind: int


@dataclass(frozen=True)
class MapData:
    lines: tuple[Line, ...]
    sectors: tuple[tuple[int, int, int, int], ...]  # floor, ceiling, special, tag
    things: tuple[Thing, ...]
    vertices: tuple[tuple[int, int], ...]


def _records(raw: bytes, fmt: str) -> list[tuple]:
    size = struct.calcsize(fmt)
    return [struct.unpack(fmt, raw[k: k + size]) for k in range(0, len(raw) - size + 1, size)]


def read_map(wad_path: str, map_name: str) -> MapData:
    with open(wad_path, "rb") as f:
        data = f.read()
    _, count, offset = struct.unpack("<4sii", data[:12])
    lumps = [struct.unpack("<ii8s", data[offset + 16 * i: offset + 16 * i + 16]) for i in range(count)]
    names = [name.rstrip(b"\0").decode("ascii", "replace") for _, _, name in lumps]
    start = names.index(map_name.upper())  # ValueError if the map does not exist
    raw = {names[j]: data[lumps[j][0]: lumps[j][0] + lumps[j][1]] for j in range(start + 1, start + 11)}

    verts = tuple(_records(raw["VERTEXES"], "<hh"))
    sides = _records(raw["SIDEDEFS"], "<hh8s8s8sh")
    sectors = tuple((s[0], s[1], s[5], s[6]) for s in _records(raw["SECTORS"], "<hh8s8shhh"))
    lines = tuple(
        Line(verts[v1], verts[v2], flags, special, tag, sides[right][5], sides[left][5] if left != -1 else -1)
        for v1, v2, flags, special, tag, right, left in _records(raw["LINEDEFS"], "<hhhhhhh")
    )
    things = tuple(Thing(x, y, angle, kind) for x, y, angle, kind, _ in _records(raw["THINGS"], "<hhhhh"))
    return MapData(lines, sectors, things, verts)


def map_names(wad_path: str) -> list[str]:
    """The WAD's maps (MAP01.. or E1M1..), for the UI to offer."""
    with open(wad_path, "rb") as f:
        data = f.read()
    _, count, offset = struct.unpack("<4sii", data[:12])
    names = [data[offset + 16 * i + 8: offset + 16 * i + 16].rstrip(b"\0").decode("ascii", "replace")
             for i in range(count)]
    return [n for n in names if (n.startswith("MAP") and n[3:].isdigit()) or
            (len(n) == 4 and n[0] == "E" and n[2] == "M")]
