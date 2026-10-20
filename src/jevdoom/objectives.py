"""What to do now to finish the level, like a player who knows the map: grab the
weapon that is right there, get health when hurt, go to the exit if it is reachable;
otherwise whatever unlocks the way — a key, a switch, a line that triggers something.
On reaching an objective it is marked done; the live sector heights (a door that
opened, a lift that came down, stairs that rose), the keys and the collected items
change the floor plan and the distance fields are rebuilt. With nothing reachable,
it explores what it has not stepped on yet."""

import math
from dataclasses import dataclass

import numpy as np

from jevdoom import config
from jevdoom import level as lv
from jevdoom import wad

KEY_REACH = 24
ITEM_REACH = 20
WALK_REACH = 16
EXPLORE_BLOCK = 128
USE_PRESSES = 4
RECOMPUTE_EVERY = 12
REFRESH_EVERY = 40
GIVE_UP_STUCK = 3
# only detour for items within this straight-line distance: armor across the map
# cost more health on the way than it gave
ITEM_RADIUS = {"weapon": 700, "armor": 500, "health": 900, "ammo": 600}
LOW_AMMO = {2: 20, 3: 8}  # bullets, shells
HOT_ITEM = 400
PRESENT_TOLERANCE = 16

# Priority in cells (8 units): how much a target "costs extra" over an equally close one.
PENALTY = {"exit": 0, "key": 0, "weapon": 25, "health": 0, "armor": 25, "ammo": 20, "switch": 40,
           "trigger": 40, "explore": 0}

ITEM_KINDS = ("weapon", "health", "armor", "ammo")
# THING type -> (ViZDoom name, category, weapon slot)
ITEMS = {
    2001: ("Shotgun", "weapon", 3), 82: ("SuperShotgun", "weapon", None), 2002: ("Chaingun", "weapon", 4),
    2003: ("RocketLauncher", "weapon", 5), 2004: ("PlasmaRifle", "weapon", 6), 2006: ("BFG9000", "weapon", 7),
    2012: ("Medikit", "health", None), 2011: ("Stimpack", "health", None), 2013: ("Soulsphere", "health", None),
    2018: ("GreenArmor", "armor", None), 2019: ("BlueArmor", "armor", None),
    2008: ("Shell", "ammo", 3), 2049: ("ShellBox", "ammo", 3),
    2007: ("Clip", "ammo", 2), 2048: ("ClipBox", "ammo", 2),
}
ITEM_LABELS = {"Shotgun": "shotgun", "SuperShotgun": "super shotgun", "Chaingun": "chaingun",
               "RocketLauncher": "rocket launcher", "PlasmaRifle": "plasma rifle", "BFG9000": "BFG",
               "Medikit": "medikit", "Stimpack": "stimpack", "Soulsphere": "soulsphere",
               "GreenArmor": "green armor", "BlueArmor": "blue armor",
               "Shell": "shells", "ShellBox": "box of shells", "Clip": "bullets", "ClipBox": "box of bullets"}


@dataclass(frozen=True)
class World:
    """What the player knows about the world this step, read from ViZDoom by the session."""
    heights: list | None = None            # live (floor, ceiling) of each sector
    keys: frozenset = frozenset()          # key colors he holds
    objects: tuple = ()                    # (name, x, y) of what is still on the map
    health: float = 100.0
    armor: float = 0.0
    weapon_slots: frozenset = frozenset()  # weapon slots he owns
    ammo: dict = None                      # slot -> ammo (2 = bullets, 3 = shells)
    threats: tuple = ()                    # (x, y) of hitscan shooters in view


@dataclass
class Objective:
    id: int
    kind: str  # exit | key | switch | trigger | weapon | health | armor | ammo | explore
    label: str
    cells: set
    aim: tuple[float, float] | None = None
    line: wad.Line | None = None
    key: str | None = None   # key required (locked switch) or provided (key)
    item: str | None = None  # the item's ViZDoom name
    slot: int | None = None
    repeatable: bool = False
    done: bool = False
    gave_up: bool = False  # gave up after getting stuck: counts again when nothing else is left
    visits: int = 0
    stuck: int = 0

    @property
    def point(self) -> tuple[float, float] | None:
        if self.aim is not None:
            return self.aim
        if self.line is not None:
            return (self.line.a[0] + self.line.b[0]) / 2, (self.line.a[1] + self.line.b[1]) / 2
        return None


@dataclass(frozen=True)
class Plan:
    route: lv.Route | None
    objective: Objective | None


class Planner:
    def __init__(self, level: lv.Level) -> None:
        self.level = level
        self.world = World()
        self.objectives = self._build()
        self.visited: set[tuple[int, int]] = set()
        self.current: Objective | None = None
        self._field: lv.Field | None = None
        self._dirty = True
        self._since_recompute = 0
        self._explore_objective: Objective | None = None
        self._pos = (0.0, 0.0)

    # ---------- setup ----------

    def _build(self) -> list[Objective]:
        level, data = self.level, self.level.data
        objectives: list[Objective] = []

        def add(**kwargs) -> None:
            if kwargs["cells"]:
                objectives.append(Objective(id=len(objectives), **kwargs))

        for line in data.lines:
            if line.special in wad.EXITS_USE:
                add(kind="exit", label="exit switch", line=line,
                    cells=level.cells_near_line(line, lv.USE_RANGE, front_only=True))
            elif line.special in wad.EXITS_WALK:
                add(kind="exit", label="exit line", line=line,
                    cells=level.cells_near_line(line, WALK_REACH, front_only=False))
            elif line.special in wad.SWITCHES and line.special not in wad.MANUAL_DOORS:
                key = wad.LOCKED_SWITCHES.get(line.special)
                add(kind="switch", label=f"switch ({key})" if key else "switch",
                    line=line, key=key, repeatable=line.special not in _ONCE,
                    cells=level.cells_near_line(line, lv.USE_RANGE, front_only=True))
            elif line.is_walk_trigger:
                add(kind="trigger", label="walk-over trigger", line=line,
                    repeatable=line.special not in _ONCE,
                    cells=level.cells_near_line(line, WALK_REACH, front_only=False))
        for thing in data.things:
            if thing.kind in wad.KEY_THINGS:
                color = wad.KEY_THINGS[thing.kind]
                add(kind="key", label=f"{color} key", key=color, aim=(thing.x, thing.y),
                    cells=level.cells_around(thing.x, thing.y, KEY_REACH))
            elif thing.kind in ITEMS:
                name, kind, slot = ITEMS[thing.kind]
                add(kind=kind, label=f"grab {ITEM_LABELS[name]}", item=name, slot=slot, aim=(thing.x, thing.y),
                    cells=level.cells_around(thing.x, thing.y, ITEM_REACH))
        return objectives

    # ---------- every step ----------

    def update(self, x: float, y: float, world: World) -> Plan:
        previous, self.world, self._pos = self.world, world, (x, y)
        if world.keys != previous.keys or self._mark_collected():
            self._dirty = True
        if (world.health < config.LOW_HEALTH) != (previous.health < config.LOW_HEALTH):
            self._dirty = True
        if bool(world.threats) != bool(previous.threats):
            self._dirty = True
        self.visited.add((int(x // EXPLORE_BLOCK), int(y // EXPLORE_BLOCK)))
        self._since_recompute += 1
        changed = self.level.update_heights(world.heights, world.keys)
        if (changed and self._since_recompute >= RECOMPUTE_EVERY) or self._since_recompute >= REFRESH_EVERY:
            self._dirty = True
        if self._dirty or self._field is None:
            self._recompute(x, y)

        route = self._route(x, y)
        if route is None and self._since_recompute > 0:
            self._recompute(x, y)
            route = self._route(x, y)
        objective = self._by_id(route.target) if route else None
        if objective is not None:
            self._arrive(objective, route)
        self.current = objective
        return Plan(route, objective)

    def stuck(self, x: float, y: float) -> None:
        """Got stuck heading to the current objective: the floor plan learns the block
        and, if the same objective keeps getting stuck, gives up on it for now."""
        if self._field is None:
            return
        if self.level.doubt_open(x, y):  # the assumed-open door did not open: replan without it
            self._dirty = True
            return
        self.level.mark_stuck(self._field, x, y)
        if self.current is not None and self.current.kind == "explore":
            self._give_up_block(x, y)
        elif self.current is not None:
            self.current.stuck += 1
            if self.current.stuck >= GIVE_UP_STUCK and self.current.kind != "exit":
                self.current.done = self.current.gave_up = True
        self._dirty = True

    def _give_up_block(self, x: float, y: float) -> None:
        """Stuck while exploring: the block it was heading to counts as seen (unreachable)."""
        start = self.level.nearest_reachable(self._field, x, y)
        if start is None:
            return
        path = self.level.downhill(self._field, start, 4000)
        gx, gy = self.level.center(path[-1])
        self.visited.add((int(gx // EXPLORE_BLOCK), int(gy // EXPLORE_BLOCK)))

    def summary(self) -> dict:
        """For the Jev and the UI: what is at stake in the level."""
        countable = [o for o in self.objectives if o.kind in ("key", "switch", "trigger")]
        return {
            "objective": self.current.label if self.current else None,
            "keys": sorted(self.world.keys),
            "objectives_done": sum(o.done for o in countable),
            "objectives_total": len(countable),
        }

    def objectives_view(self) -> list[dict]:
        """The list drawn on the UI's minimap."""
        return [{"id": o.id, "kind": o.kind, "label": o.label, "x": round(o.point[0]), "y": round(o.point[1]),
                 "done": o.done} for o in self.objectives if o.point is not None]

    # ---------- internals ----------

    def _route(self, x: float, y: float) -> lv.Route | None:
        return self.level.route(self._field, x, y, self._aim) if self._field is not None else None

    def _by_id(self, target: int) -> Objective | None:
        if 0 <= target < len(self.objectives):
            return self.objectives[target]
        return self._explore_objective

    def _aim(self, target: int) -> tuple[float, float] | None:
        """Point to face on arrival: the point of the switch nearest the player (facing
        the wall — aiming at its middle from the side, "use" does not reach), the key,
        the item."""
        o = self._by_id(target)
        if o is None or o.kind == "trigger":
            return None
        if o.line is not None and o.line.is_use:
            (ax, ay), (bx, by) = o.line.a, o.line.b
            inset = 0.2  # away from the ends, where the "use" ray slips off
            a = (ax + inset * (bx - ax), ay + inset * (by - ay))
            b = (bx - inset * (bx - ax), by - inset * (by - ay))
            return lv.closest_point(self._pos, a, b)
        return o.point

    def _present(self, o: Objective) -> bool:
        """Item or key still on the map (ViZDoom only lists what exists at this difficulty)."""
        if o.kind == "key":
            return any(wad.KEY_NAMES.get(n) == o.key for n, _, _ in self.world.objects)
        return any(n == o.item and abs(x - o.aim[0]) <= PRESENT_TOLERANCE and abs(y - o.aim[1]) <= PRESENT_TOLERANCE
                   for n, x, y in self.world.objects)

    def _mark_collected(self) -> bool:
        if not self.world.objects:
            return False
        changed = False
        for o in self.objectives:
            if (o.kind == "key" or o.kind in ITEM_KINDS) and not o.done and not self._present(o):
                o.done = changed = True
        return changed

    def _arrive(self, objective: Objective, route: lv.Route) -> None:
        """Switches and trigger lines count as done when the sector they control (same
        tag) moves — a door opened, a lift came down. Without that signal, after
        insisting for a while, it gives up (it may be a light, a trap)."""
        if route.distance > 2 * lv.CELL or objective.kind == "exit":
            return
        objective.visits += 1
        acted = objective.line is not None and self._tag_moved(objective.line.tag)
        needed = 1 if objective.kind not in ("switch", "trigger") else (1 if acted else USE_PRESSES * 3)
        if objective.visits >= needed:
            objective.done = True
            # insisted without seeing an effect: maybe it never fired — counts again if nothing else is left
            objective.gave_up = objective.kind in ("switch", "trigger") and not acted and bool(self.world.heights)
            self._dirty = True
            if self.world.heights is None and objective.line is not None:
                self.level.assume_open(objective.line.tag)

    def _tag_moved(self, tag: int) -> bool:
        if not tag or not self.world.heights:
            return False
        sectors = self.level.data.sectors
        return any(sectors[i][3] == tag and (abs(f - sectors[i][0]) > 1 or abs(c - sectors[i][1]) > 1)
                   for i, (f, c) in enumerate(self.world.heights))

    def _available(self, o: Objective) -> bool:
        w = self.world
        if o.done:
            return False
        if o.kind in ITEM_KINDS:
            if math.dist(o.aim, self._pos) > ITEM_RADIUS[o.kind]:
                return False
            if any(math.dist(o.aim, t) < HOT_ITEM for t in w.threats):
                return False  # item in the middle of the shooters: clear the room first, from afar
            if o.kind == "weapon":
                return o.slot is None or o.slot not in w.weapon_slots
            if o.kind == "ammo":
                return o.slot in w.weapon_slots and (w.ammo or {}).get(o.slot, 0) < LOW_AMMO[o.slot]
            return (w.health if o.kind == "health" else w.armor) < config.LOW_HEALTH
        if o.kind == "key":
            return o.key not in w.keys
        return o.kind == "trigger" or (o.kind == "switch" and (o.key is None or o.key in w.keys))

    def _recompute(self, x: float, y: float) -> None:
        """Picks the first stage with some reachable target and computes ONE field,
        only for that stage's reachable targets."""
        self._dirty = False
        self._since_recompute = 0
        self._explore_objective = None
        reach = self.level.reachable_from(x, y)
        items = {o.id for o in self.objectives if o.kind in ITEM_KINDS and self._available(o)}
        exits = {o.id for o in self.objectives if o.kind == "exit"}
        open_goals = lambda: {o.id for o in self.objectives if o.kind != "exit" and self._available(o)}
        for stage in (lambda: exits | items, open_goals, self._forget_learned, lambda: exits | items, open_goals,
                      self._reset_repeatables):
            ids = stage()
            if stage == self._forget_learned:
                reach = self.level.reachable_from(x, y)
            goals = {i: cells for i in ids if (cells := {c for c in self.objectives[i].cells if reach[c]})}
            if goals:
                self._use(goals)
                return
        for _ in range(2):  # nothing new: forget what it has seen and explore again
            explore = {i: {c for c in cells if reach[c]} for i, cells in self._explore_goals().items()}
            explore = {i: cells for i, cells in explore.items() if cells}
            if explore:
                self._use(explore)
                return
            self.visited = {(int(x // EXPLORE_BLOCK), int(y // EXPLORE_BLOCK))}
        self._field = None

    def _use(self, goals: dict[int, set]) -> None:
        penalty = {i: PENALTY.get(self._by_id(i).kind, 0) for i in goals}
        self._field = self.level.field(goals, penalty)

    def _forget_learned(self) -> set[int]:
        if self.level.learned.any():
            self.level.forget()
            self.level.update_heights(self.world.heights, self.world.keys)
        return set()

    def _reset_repeatables(self) -> set[int]:
        """Everything done and nothing reachable: redo repeatable switches and lines and
        retry the objectives it gave up on after getting stuck."""
        again = set()
        for o in self.objectives:
            if o.repeatable or o.gave_up:
                o.done = o.gave_up = False
                o.stuck = 0
                again.add(o.id)
        return again

    def _explore_goals(self) -> dict[int, set]:
        """128-unit blocks not stepped on yet become targets: go see what is there.
        Any free cell of the block will do (its center may fall inside a wall)."""
        level = self.level
        rows, cols = np.nonzero(~level.blocked)
        x = level.origin[0] + lv.CELL * cols + lv.CELL / 2
        y = level.origin[1] + lv.CELL * rows + lv.CELL / 2
        bx, by = np.floor_divide(x, EXPLORE_BLOCK), np.floor_divide(y, EXPLORE_BLOCK)
        unseen = np.array([(int(a), int(b)) not in self.visited for a, b in zip(bx, by)], dtype=bool)
        cells = set(zip(rows[unseen].tolist(), cols[unseen].tolist()))
        if not cells:
            return {}
        self._explore_objective = Objective(id=len(self.objectives), kind="explore",
                                            label="explore new area", cells=cells)
        return {self._explore_objective.id: cells}


# Specials that only work once (S1 and W1); the rest are repeatable (SR/WR).
_ONCE = frozenset({
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 29, 30,
    35, 36, 37, 38, 39, 40, 41, 44, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 71, 101, 102, 103,
    104, 108, 109, 110, 111, 112, 113, 121, 122, 124, 127, 130, 131, 133, 135, 137, 140, 141,
})
