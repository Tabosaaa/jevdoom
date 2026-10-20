"""The level's real floor plan, read from the WAD by wad.py: walls, floor/ceiling
heights, doors, lifts and teleporters. It becomes a walkability grid that accounts
for the player's radius, plus distance fields (reverse Dijkstra) to any set of
targets. It is the "mental map" of someone who has played the level before — what
Phases 6–8 were missing. objectives.py picks the targets (exit, key, switch) and asks
this module for the `Route`."""

import heapq
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from jevdoom import wad

CELL = 8
PLAYER_RADIUS = 16
WALL_CLEARANCE = 16  # the player radius: any extra margin erases 40–48 wide corridors from the grid
MAX_STEP = 24
PLAYER_HEIGHT = 56
OPEN_DOOR_GAP = 72
LOOKAHEAD_CELLS = 10
USE_REACH = 80
USE_RANGE = 56
TELEPORT_REACH = 24
COVER_RADIUS = 320  # reaches the door he came through
WAIT_ZONE = 64
FLOOR_REACH = 8

# Solid Doom decorations (THING type -> radius), the ones most common in levels.
SOLID_THINGS = {
    2035: 10, 70: 10, 2028: 16, 85: 16, 86: 16, 48: 16, 30: 16, 31: 16, 32: 16, 33: 16,
    35: 16, 36: 16, 37: 16, 41: 16, 42: 16, 43: 16, 44: 16, 45: 16, 46: 16, 47: 16, 54: 32,
    55: 16, 56: 16, 57: 16, 25: 16, 26: 16, 27: 16, 28: 16, 29: 16,
}

_STEPS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))

Cell = tuple[int, int]


def _shift(a: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc], with `fill` outside the grid."""
    out = np.full_like(a, fill)
    h, w = a.shape
    out[max(-dr, 0): h - max(dr, 0), max(-dc, 0): w - max(dc, 0)] = \
        a[max(dr, 0): h + min(dr, 0), max(dc, 0): w + min(dc, 0)]
    return out


def _neighbourhood(a: np.ndarray, reduce, reach: int = PLAYER_RADIUS) -> np.ndarray:
    """The player is 32 units wide: the effective floor is the highest one under him
    (which is why thin trenches can be walked over) and the ceiling, the lowest."""
    k = reach // CELL
    fill = a.min() if reduce is np.max else a.max()
    span = range(-k, k + 1)
    return reduce([_shift(a, dr, dc, fill) for dr in span for dc in span], axis=0)


def closest_point(p, a, b) -> tuple[float, float]:
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (vx * vx + vy * vy or 1)))
    return a[0] + t * vx, a[1] + t * vy


def segment_distance(p, a, b) -> float:
    return math.dist(p, closest_point(p, a, b))


def front_side(line: wad.Line, p) -> bool:
    """A line's front is the right-hand side of a→b."""
    (ax, ay), (bx, by) = line.a, line.b
    return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) < 0


@dataclass(frozen=True)
class Field:
    """Distance (in cells) from each cell to the nearest target, and which target it is."""
    dist: np.ndarray
    owner: np.ndarray
    base: dict = None  # each target's penalty (its "zero" distance)

    def remaining(self, cell) -> float:
        """Cells to the target, not counting the priority penalty."""
        return float(self.dist[cell]) - (self.base or {}).get(int(self.owner[cell]), 0.0)


@dataclass(frozen=True)
class Route:
    waypoint: tuple[float, float]
    distance: float
    use_ahead: bool
    target: int  # id of the target (owner) the route leads to


class Level:
    def __init__(self, data: wad.MapData) -> None:
        self.data = data
        xs = [v[0] for v in data.vertices]
        ys = [v[1] for v in data.vertices]
        self.origin = (min(xs) - CELL, min(ys) - CELL)
        w = (max(xs) - self.origin[0]) // CELL + 2
        h = (max(ys) - self.origin[1]) // CELL + 2
        self.shape = (h, w)
        gx, gy = np.meshgrid(self.origin[0] + CELL * np.arange(w) + CELL / 2 + 0.01,
                             self.origin[1] + CELL * np.arange(h) + CELL / 2 + 0.01)
        self.sector, self.static_blocked = self._rasterize(gx, gy)
        self.learned = np.zeros(self.shape, bool)  # stretches where it got stuck (mark_stuck)
        self.blocked = self.static_blocked

        self.door_keys: dict[int, str | None] = {}
        for line in data.lines:
            if line.special in wad.MANUAL_DOORS and line.back != -1:
                self.door_keys[line.back] = wad.DOOR_KEYS.get(line.special)
        lift_tags = {line.tag for line in data.lines if line.special in wad.LIFTS and line.tag}
        lift_sectors = [i for i, s in enumerate(data.sectors) if s[3] in lift_tags]
        lift_core = np.isin(self.sector, lift_sectors)
        self._lift_core = lift_core
        # touching the lift means standing on it (player radius): that is "lift" too
        self.lift = _neighbourhood(lift_core, np.max)
        doors = np.isin(self.sector, list(self.door_keys))
        # next to a door or lift, standing still means waiting for it, not being stuck
        self.special_cells = self._grow(self.lift | doors, WAIT_ZONE)
        self.use_lines = [line for line in data.lines if line.is_use]
        self._seg = np.array([(*l.a, *l.b) for l in data.lines], float)
        self._seg_front = np.array([l.front for l in data.lines])
        self._seg_back = np.array([l.back for l in data.lines])
        self.teleports = self._teleports()

        self.assumed_open: set[int] = set()  # sectors assumed open (map without sectors_info)
        self._heights: tuple | None = None
        self.update_heights(None, frozenset())

    def _rasterize(self, gx: np.ndarray, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sector of each cell (point-in-polygon) and cells blocked by walls or solid
        decorations, already grown by the player radius."""
        data = self.data
        inside = np.zeros((len(data.sectors), *self.shape), bool)
        blocked = np.zeros(self.shape, bool)
        ys, xs = gy[:, 0], gx[0]
        for line in data.lines:
            (ax, ay), (bx, by) = line.a, line.b
            # crossing parity (point-in-polygon), only on the grid rows the line spans
            rows = np.nonzero((ay > ys) != (by > ys))[0]
            if len(rows):
                cut = ax + (ys[rows] - ay) * (bx - ax) / ((by - ay) or 1e-9)
                crosses = xs[None, :] < cut[:, None]
                for sector in (line.front, line.back):
                    if sector != -1:
                        inside[sector, rows] ^= crosses
            if line.blocking:  # distance to the wall, only inside the line's bounding box
                r0, c0 = self.cell(min(ax, bx) - WALL_CLEARANCE, min(ay, by) - WALL_CLEARANCE)
                r1, c1 = self.cell(max(ax, bx) + WALL_CLEARANCE, max(ay, by) + WALL_CLEARANCE)
                box = (slice(max(r0, 0), r1 + 1), slice(max(c0, 0), c1 + 1))
                px, py = gx[box], gy[box]
                vx, vy = bx - ax, by - ay
                t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy or 1), 0, 1)
                blocked[box] |= np.hypot(px - ax - t * vx, py - ay - t * vy) < WALL_CLEARANCE
        for thing in data.things:
            if thing.kind in SOLID_THINGS:
                blocked |= np.hypot(gx - thing.x, gy - thing.y) < SOLID_THINGS[thing.kind] + PLAYER_RADIUS
        sector = np.full(self.shape, -1)
        for s in np.argsort(-inside.sum((1, 2))):  # nested sectors: the smallest wins
            sector[inside[s]] = s
        return sector, blocked | (sector < 0)

    @staticmethod
    def _grow(mask: np.ndarray, radius: float) -> np.ndarray:
        out = mask.copy()
        for _ in range(int(radius // CELL)):
            out = out | _shift(out, 1, 0, False) | _shift(out, -1, 0, False) | _shift(out, 0, 1, False) | _shift(out, 0, -1, False)
        return out

    def waiting_zone(self, x: float, y: float) -> bool:
        """Next to a door or lift (where waiting is part of the way)."""
        cell = self.cell(x, y)
        return self.inside(cell) and bool(self.special_cells[cell])

    # ---------- geometry ----------

    def cell(self, x: float, y: float) -> Cell:
        return int((y - self.origin[1]) // CELL), int((x - self.origin[0]) // CELL)

    def center(self, cell: Cell) -> tuple[float, float]:
        return self.origin[0] + CELL * cell[1] + CELL / 2, self.origin[1] + CELL * cell[0] + CELL / 2

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def cells_around(self, x: float, y: float, radius: float, keep=lambda p: True) -> set[Cell]:
        r0, c0 = self.cell(x, y)
        k = int(radius // CELL) + 1
        return {(r, c) for r in range(r0 - k, r0 + k + 1) for c in range(c0 - k, c0 + k + 1)
                if self.inside((r, c)) and not self.static_blocked[r, c]
                and math.dist(self.center((r, c)), (x, y)) <= radius and keep(self.center((r, c)))}

    def cells_near_line(self, line: wad.Line, reach: float, front_only: bool) -> set[Cell]:
        mx, my = (line.a[0] + line.b[0]) / 2, (line.a[1] + line.b[1]) / 2
        half = math.dist(line.a, line.b) / 2 + reach
        return self.cells_around(mx, my, half, lambda p: segment_distance(p, line.a, line.b) < reach
                                 and (not front_only or front_side(line, p)))

    def _teleports(self) -> dict[Cell, Cell]:
        """Cell in front of a teleport line -> destination cell."""
        destinations = {}
        for thing in self.data.things:
            cell = self.cell(thing.x, thing.y)
            if thing.kind == wad.TELEPORT_DEST and self.inside(cell) and self.sector[cell] >= 0:
                destinations.setdefault(self.data.sectors[self.sector[cell]][3], cell)
        jumps = {}
        for line in self.data.lines:
            if line.special in wad.TELEPORTS and line.tag in destinations:
                for src in self.cells_near_line(line, TELEPORT_REACH, front_only=True):
                    jumps[src] = destinations[line.tag]
        return jumps

    # ---------- line of sight and cover ----------

    def _sight_blockers(self) -> np.ndarray:
        """Lines that block shots and sight right now: one-sided walls and closed
        two-sided openings (a lowered door, ceiling touching the floor)."""
        one_sided = self._seg_back == -1
        back = np.where(one_sided, 0, self._seg_back)
        gap = (np.minimum(self.sector_ceil[self._seg_front], self.sector_ceil[back])
               - np.maximum(self.sector_floor[self._seg_front], self.sector_floor[back]))
        return self._seg[one_sided | (gap <= 0)]

    def visible(self, p, targets: np.ndarray) -> np.ndarray:
        """For each point in `targets` (N×2), whether the segment from `p` to it crosses no wall."""
        walls = self._sight_blockers()
        ax, ay, bx, by = (walls[:, i][None, :] for i in range(4))
        px, py = p
        qx, qy = targets[:, 0][:, None], targets[:, 1][:, None]
        d1x, d1y, d2x, d2y = qx - px, qy - py, bx - ax, by - ay
        denom = d1x * d2y - d1y * d2x
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((ax - px) * d2y - (ay - py) * d2x) / denom
            u = ((ax - px) * d1y - (ay - py) * d1x) / denom
        hit = (denom != 0) & (t > 0.001) & (t < 0.999) & (u >= 0) & (u <= 1)
        return ~hit.any(axis=1)

    def _nearby_points(self, x: float, y: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Free points on the same floor within `radius` that he can reach in a straight line."""
        r0, c0 = self.cell(x, y)
        k = int(radius // CELL)
        rows, cols = np.mgrid[max(r0 - k, 0): r0 + k + 1, max(c0 - k, 0): c0 + k + 1]
        rows, cols = rows.ravel(), cols.ravel()
        keep = (rows < self.shape[0]) & (cols < self.shape[1])
        rows, cols = rows[keep], cols[keep]
        keep = ~self.blocked[rows, cols] & (np.abs(self.floor[rows, cols] - self.floor[r0, c0]) <= MAX_STEP)
        rows, cols = rows[keep], cols[keep]
        points = np.stack([self.origin[0] + CELL * cols + CELL / 2, self.origin[1] + CELL * rows + CELL / 2], 1)
        near = np.hypot(points[:, 0] - x, points[:, 1] - y)
        points, near = points[near <= radius], near[near <= radius]
        if len(points):
            reachable = self.visible((x, y), points)
            points, near = points[reachable], near[reachable]
        return points, near

    def cover(self, x: float, y: float, threat: tuple[float, float], radius: float = COVER_RADIUS):
        """The nearest point within `radius` where the threat cannot see the player —
        the door frame, the pillar."""
        points, near = self._nearby_points(x, y, radius)
        if not len(points):
            return None
        hidden = ~self.visible(threat, points)
        if not hidden.any():
            return None
        best = np.argmin(np.where(hidden, near, np.inf))
        return float(points[best, 0]), float(points[best, 1])

    def escape(self, x: float, y: float, threat: tuple[float, float], radius: float = COVER_RADIUS):
        """Where to back off from a melee attacker: the reachable point that most
        increases the distance to the threat (not the wall right behind)."""
        points, _ = self._nearby_points(x, y, radius)
        if not len(points):
            return None
        gain = np.hypot(points[:, 0] - threat[0], points[:, 1] - threat[1])
        best = int(np.argmax(gain))
        if gain[best] <= math.dist((x, y), threat) + CELL:
            return None
        return float(points[best, 0]), float(points[best, 1])

    # ---------- live heights ----------

    def update_heights(self, live: list[tuple[float, float]] | None, keys: frozenset[str]) -> bool:
        """Recomputes walkability from the current sector heights (opened remote
        doors, lifts, raised stairs) and the keys the player holds. Manual doors count
        as open if he can open them. Returns True if anything changed — then the
        distance fields must be rebuilt."""
        source = live if live is not None else [(s[0], s[1]) for s in self.data.sectors]
        floors = np.array([f for f, _ in source], float)
        ceils = np.array([c for _, c in source], float)
        self.sector_floor, self.sector_ceil = floors, ceils.copy()  # real ones, for line of sight
        for s, key in self.door_keys.items():
            if key is None or key in keys:
                ceils[s] = max(ceils[s], floors[s] + OPEN_DOOR_GAP)
        if live is None:
            for s in self.assumed_open:
                ceils[s] = max(ceils[s], floors[s] + OPEN_DOOR_GAP)
        signature = (floors.tobytes(), ceils.tobytes())
        if signature == self._heights:
            return False
        self._heights = signature

        self.blocked = self.static_blocked | self.learned
        valid = self.sector >= 0
        # a lift floor keeps moving: it must not leak into the effective floor around it
        solid_floor = np.where(valid & ~self._lift_core, floors[self.sector], -1e4)
        # floor reach of half a radius: a 32-wide body fits a 32-wide pit (falls in and
        # stays there); with the whole radius the pit "vanished" and the route said climb
        self.floor = np.where(self._lift_core, floors[self.sector],
                              _neighbourhood(solid_floor, np.max, FLOOR_REACH))
        self.ceil = _neighbourhood(np.where(valid, ceils[self.sector], 1e4), np.min)
        self._build_moves()
        return True

    def _build_moves(self) -> None:
        """moves[d][r, c]: whether one can go from (r, c) in direction d. Also kept as
        flat bytes so the pure-Python Dijkstra indexes them fast."""
        ok = []
        for dr, dc, _ in _STEPS:
            dest_floor = _shift(self.floor, dr, dc, 1e4)
            dest_ceil = _shift(self.ceil, dr, dc, -1e4)
            # height wildcard only for stepping on or off the lift itself: two cells next
            # to it (pit and platform) do not connect without going through it
            lift = self._lift_core | _shift(self._lift_core, dr, dc, False)
            step_ok = (dest_floor - self.floor <= MAX_STEP) | lift
            gap_ok = np.minimum(self.ceil, dest_ceil) - np.maximum(self.floor, dest_floor) >= PLAYER_HEIGHT
            ok.append(~self.blocked & ~_shift(self.blocked, dr, dc, True) & step_ok & (gap_ok | lift))
        for d, (dr, dc, _) in enumerate(_STEPS):
            if dr and dc:  # diagonal only if both orthogonal moves pass (no corner cutting)
                ok[d] = ok[d] & ok[_STEPS.index((dr, 0, 1.0))] & ok[_STEPS.index((0, dc, 1.0))]
        self.moves = np.stack(ok)
        self._flat_moves = [bytes(m.ravel()) for m in ok]

    # ---------- fields and route ----------

    def field(self, goals: dict[int, set[Cell]], penalty: dict[int, float] | None = None) -> Field:
        """Reverse Dijkstra from all targets at once: every cell knows the distance to
        the nearest target and which one it is. `penalty` (in cells) makes a target
        less attractive than an equally close one — that is priority."""
        h, w = self.shape
        penalty = penalty or {}
        dist = [math.inf] * (h * w)
        owner = [-1] * (h * w)
        queue = []
        for target, cells in goals.items():
            start = float(penalty.get(target, 0.0))
            for r, c in cells:
                i = r * w + c
                if start < dist[i]:
                    dist[i], owner[i] = start, target
                    queue.append((start, i))
        heapq.heapify(queue)
        back: dict[int, list[int]] = {}
        for src, dst in self.teleports.items():
            back.setdefault(dst[0] * w + dst[1], []).append(src[0] * w + src[1])
        offsets = [(dr * w + dc, step, self._flat_moves[k]) for k, (dr, dc, step) in enumerate(_STEPS)]
        blocked = bytes(self.blocked.ravel())
        size = h * w
        while queue:
            d, v = heapq.heappop(queue)
            if d > dist[v]:
                continue
            for off, step, moves in offsets:
                u = v - off
                if 0 <= u < size and moves[u] and d + step < dist[u]:
                    dist[u], owner[u] = d + step, owner[v]
                    heapq.heappush(queue, (d + step, u))
            for u in back.get(v, ()):
                if d + 1 < dist[u] and not blocked[u]:
                    dist[u], owner[u] = d + 1, owner[v]
                    heapq.heappush(queue, (d + 1, u))
        return Field(np.array(dist).reshape(h, w), np.array(owner).reshape(h, w), dict(penalty))

    def _floor_under(self, x: float, y: float) -> float:
        """Real floor of the sector under the player. In a pit, the edge cells already
        belong to the level above — they cannot be a starting point."""
        cell = self.cell(x, y)
        if self.inside(cell) and self.sector[cell] >= 0:
            return float(self.sector_floor[self.sector[cell]])
        return math.inf

    def reachable_from(self, x: float, y: float) -> np.ndarray:
        """Every cell the player can get to from here (forward search, with drops and
        teleports). One search answers "is this target reachable?" for every target —
        much cheaper than one field per attempt."""
        h, w = self.shape
        seen = bytearray(h * w)
        r0, c0 = self.cell(x, y)
        here = self._floor_under(x, y)
        starts = [(r, c) for r in range(r0 - 3, r0 + 4) for c in range(c0 - 3, c0 + 4)
                  if self.inside((r, c)) and not self.blocked[r, c] and self.floor[r, c] - here <= MAX_STEP]
        queue = deque(r * w + c for r, c in starts)
        for i in queue:
            seen[i] = 1
        offsets = [(dr * w + dc, self._flat_moves[k]) for k, (dr, dc, _) in enumerate(_STEPS)]
        jumps = {src[0] * w + src[1]: dst[0] * w + dst[1] for src, dst in self.teleports.items()}
        while queue:
            u = queue.popleft()
            for off, moves in offsets:
                if moves[u]:
                    v = u + off
                    if not seen[v]:
                        seen[v] = 1
                        queue.append(v)
            v = jumps.get(u)
            if v is not None and not seen[v]:
                seen[v] = 1
                queue.append(v)
        return np.frombuffer(bytes(seen), dtype=np.uint8).reshape(h, w).astype(bool)

    def nearest_reachable(self, field: Field, x: float, y: float) -> Cell | None:
        """The field cell the player starts from: the nearest one he can see in a
        straight line — otherwise the search "goes through" a closed door and the
        route starts on its other side."""
        cell = self.cell(x, y)
        here = self._floor_under(x, y)
        for radius in range(12):
            options = [(r, c) for r in range(cell[0] - radius, cell[0] + radius + 1)
                       for c in range(cell[1] - radius, cell[1] + radius + 1)
                       if self.inside((r, c)) and np.isfinite(field.dist[r, c])
                       and self.floor[r, c] - here <= MAX_STEP]
            if radius > 1 and options:
                points = np.array([self.center(o) for o in options])
                options = [o for o, ok in zip(options, self.visible((x, y), points)) if ok]
            if options:
                return min(options, key=lambda rc: field.dist[rc])
        return None

    def downhill(self, field: Field, cell: Cell, steps: int) -> list[Cell]:
        path = [cell]
        for _ in range(steps):
            u = path[-1]
            if field.dist[u] <= self._goal_floor(field, u) or u in self.teleports:
                break
            options = [(u[0] + dr, u[1] + dc) for k, (dr, dc, _) in enumerate(_STEPS) if self.moves[k][u]]
            best = min(options, key=lambda v: field.dist[v], default=None)
            if best is None or field.dist[best] >= field.dist[u]:
                break
            path.append(best)
        return path

    @staticmethod
    def _goal_floor(field: Field, cell: Cell) -> float:
        """Field distance of a target cell (0, or the target's penalty)."""
        return (field.base or {}).get(int(field.owner[cell]), 0.0)

    def route(self, field: Field, x: float, y: float, aim=None) -> Route | None:
        """Next route point in the field. `aim(target)` gives the point to face on
        arrival (the switch, the key); in front of a teleporter the point is on the
        other side of the line so he walks through it."""
        start = self.nearest_reachable(field, x, y)
        if start is None:
            return None
        target = int(field.owner[start])
        path = self.downhill(field, start, LOOKAHEAD_CELLS)
        end = path[-1]
        remaining = field.remaining(start)
        aimed = aim(target) if aim is not None and remaining <= LOOKAHEAD_CELLS else None
        lift = self._lift_ahead(path, start)
        if lift is not None:  # lift on the way, up top: walk to it and call it
            return Route(self.center(lift), CELL * remaining, True, target)
        if aimed is not None:
            waypoint = aimed
        elif end in self.teleports:
            waypoint = self._through_teleport(end)
        else:
            waypoint = self.center(end)
        use_ahead = any(segment_distance((x, y), line.a, line.b) < USE_REACH for line in self.use_lines)
        return Route(waypoint, CELL * remaining, use_ahead, target)

    def _lift_ahead(self, path: list[Cell], start: Cell) -> Cell | None:
        """First lift cell on the route, if the lift is too high to step on right now."""
        for cell in path:
            if self._lift_core[cell]:
                lift_floor = self.sector_floor[self.sector[cell]]
                return cell if lift_floor - self.floor[start] > MAX_STEP else None
        return None

    def _through_teleport(self, cell: Cell) -> tuple[float, float]:
        """A point on the far side of the teleport line, away from its ends (near them
        the wall touches and a 32-wide body does not fit), perpendicular to the line."""
        p = self.center(cell)
        line = min((l for l in self.data.lines if l.special in wad.TELEPORTS),
                   key=lambda l: segment_distance(p, l.a, l.b))
        (ax, ay), (bx, by) = line.a, line.b
        length = math.dist(line.a, line.b) or 1.0
        ux, uy = (bx - ax) / length, (by - ay) / length
        inset = min(PLAYER_RADIUS + CELL, length / 2)
        t = max(inset, min(length - inset, (p[0] - ax) * ux + (p[1] - ay) * uy))
        qx, qy = ax + t * ux, ay + t * uy
        nx, ny = -uy, ux  # left of a→b = the back side (the front is the right)
        return qx + 32 * nx, qy + 32 * ny

    def mark_stuck(self, field: Field, x: float, y: float) -> None:
        """Got stuck following the route: the floor-plan model was wrong there (a step,
        an object, bars). Blocks the stretch just ahead — except doors and lifts, which
        are only waiting to open — and forces a recompute. Learns from the collision."""
        start = self.nearest_reachable(field, x, y)
        if start is None:
            return
        for r, c in self.downhill(field, start, 2 * LOOKAHEAD_CELLS)[2:]:
            box = (slice(max(r - 1, 0), r + 2), slice(max(c - 1, 0), c + 2))
            is_goal = np.array([[field.remaining((i, j)) <= 0 for j in range(box[1].start, box[1].stop)]
                                for i in range(box[0].start, box[0].stop)])
            self.learned[box] |= ~self.special_cells[box] & ~is_goal
        self.learned[start] = False
        self._heights = None  # forces update_heights to rebuild the grid

    def assume_open(self, tag: int) -> None:
        """Without live heights: after triggering the tag's switch/line, assume its
        doors opened (which is what a switch usually does)."""
        opened = {i for i, sec in enumerate(self.data.sectors) if tag and sec[3] == tag}
        if opened - self.assumed_open:
            self.assumed_open |= opened
            self._heights = None

    def doubt_open(self, x: float, y: float, radius: float = 96) -> bool:
        """Stuck next to a door we assumed open: it did not open. Treat it as closed
        again (until another switch with the same tag is triggered)."""
        r0, c0 = self.cell(x, y)
        k = int(radius // CELL)
        box = self.sector[max(r0 - k, 0): r0 + k + 1, max(c0 - k, 0): c0 + k + 1]
        wrong = self.assumed_open & {int(s) for s in np.unique(box) if s >= 0}
        if wrong:
            self.assumed_open -= wrong
            self._heights = None
        return bool(wrong)

    def reset(self) -> None:
        """Start of an episode: forget learned blocks and assumed doors."""
        self.assumed_open.clear()
        self.forget()

    def forget(self) -> None:
        """Forget what was learned by getting stuck (when that cut off every target)."""
        self.learned[:] = False
        self._heights = None
