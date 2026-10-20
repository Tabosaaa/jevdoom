"""The agent's memory between steps: where it has been, where the nearest unexplored
area is, which walls it has not tried to open yet, whether the last move failed and
whether it is taking damage. Reads POSITION_X/Y/ANGLE and HEALTH from ViZDoom and
complements perception.py, which only sees the screen. The legacy loop in main.py
merges both into the state sent to the Jev; session.py uses it for damage and
stationary tracking."""

import math

import vizdoom as vzd

from jevdoom import config

MOVE_ACTIONS = {"move_forward": 0.0, "move_back": 180.0}
TURN_ACTIONS = frozenset({"turn_left", "turn_right"})


class Navigator:
    def __init__(self) -> None:
        self.visited: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.use_tried: set[tuple[int, int]] = set()
        self.unreachable: set[tuple[int, int]] = set()
        self.position: tuple[float, float] = (0.0, 0.0)
        self.angle: float = 0.0
        self.failed_moves: int = 0
        self.used_since_move: bool = False
        self.took_damage: bool = False
        self.stationary_steps: int = 0
        self.target: tuple[int, int] | None = None
        self.steps_since_discovery: int = 0
        self._health: float | None = None
        self._target_best = float("inf")
        self._target_age = 0
        self._updates = 0

    def update(self, game: vzd.DoomGame, last_action: str | None = None) -> None:
        x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        previous = self.position
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        self.took_damage = self._health is not None and health < self._health
        self._health = health
        self.angle = game.get_game_variable(vzd.GameVariable.ANGLE)
        self.position = (x, y)

        here = self._cell(x, y)
        known = len(self.visited)
        self.visited.add(here)
        self.walls.discard(here)
        self.steps_since_discovery = 0 if len(self.visited) > known else self.steps_since_discovery + 1

        self._updates += 1
        if self._updates <= config.WARMUP_STEPS:
            self.failed_moves = 0
            self.used_since_move = False
            return

        moved = math.dist((x, y), previous) >= config.STUCK_DIST
        self.stationary_steps = 0 if moved else self.stationary_steps + 1

        if moved:
            self.failed_moves = 0
            self.used_since_move = False
        elif last_action in MOVE_ACTIONS:
            self.failed_moves += 1
            self.walls.add(self._ahead(MOVE_ACTIONS[last_action]))
        elif last_action == "use":
            self.used_since_move = True
            self.use_tried.add(self._ahead(0))
        elif last_action in TURN_ACTIONS:
            self.failed_moves = 0
            self.used_since_move = False

        self._track_target()

    def facts(self) -> dict:
        untried = self._untried_walls()
        return {
            "blocked": self.failed_moves > 0,
            "blocked_attempts": self.failed_moves,
            "already_tried_use": self.used_since_move,
            "under_attack": self.took_damage,
            "stationary_for": self.stationary_steps,
            "path_ahead_new": self._is_new(0),
            "unexplored_bearing": self._bearing_to(self.target),
            "explored_areas": len(self.visited),
            "steps_without_discovery": self.steps_since_discovery,
            "untested_walls": len(untried),
            "untested_wall_bearing": self._bearing_to(self._nearest(untried)),
        }

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // config.CELL_SIZE), int(y // config.CELL_SIZE))

    def _ahead(self, offset_degrees: float) -> tuple[int, int]:
        radians = math.radians(self.angle + offset_degrees)
        return self._cell(
            self.position[0] + config.LOOK_AHEAD * math.cos(radians),
            self.position[1] + config.LOOK_AHEAD * math.sin(radians),
        )

    def _is_new(self, offset_degrees: float) -> bool:
        cell = self._ahead(offset_degrees)
        return cell not in self.visited and cell not in self.walls

    def _untried_walls(self) -> list[tuple[int, int]]:
        return [w for w in self.walls if w not in self.use_tried]

    def _nearest(self, cells: list[tuple[int, int]]) -> tuple[int, int] | None:
        here = self._cell(*self.position)
        options = [c for c in cells if c != here]
        return min(options, key=lambda c: math.dist(c, here)) if options else None

    def _frontier(self) -> list[tuple[int, int]]:
        """Unknown cells bordering already visited ground."""
        edges = []
        for cx, cy in self.visited:
            for neighbour in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if neighbour not in self.visited and neighbour not in self.walls:
                    edges.append(neighbour)
        return edges

    def _track_target(self) -> None:
        """Chases the nearest frontier and gives up on one that does not get closer."""
        here = self._cell(*self.position)
        if self.target is not None:
            if self.target in self.visited or self.target in self.walls:
                self.target = None
            else:
                distance = math.dist(self.target, here)
                if distance < self._target_best - 0.5:
                    self._target_best, self._target_age = distance, 0
                else:
                    self._target_age += 1
                    if self._target_age >= config.TARGET_PATIENCE:
                        self.unreachable.add(self.target)
                        self.target = None

        if self.target is None:
            options = [c for c in self._frontier() if c not in self.unreachable]
            if not options and self.unreachable:
                self.unreachable.clear()
                options = self._frontier()
            self.target = self._nearest(options)
            if self.target is not None:
                self._target_best = math.dist(self.target, here)
                self._target_age = 0

    def _bearing_to(self, cell: tuple[int, int] | None) -> str | None:
        if cell is None:
            return None
        return self.bearing_to_point(
            (cell[0] + 0.5) * config.CELL_SIZE, (cell[1] + 0.5) * config.CELL_SIZE
        )

    def bearing_to_point(self, x: float, y: float) -> str | None:
        dx = x - self.position[0]
        dy = y - self.position[1]
        relative = (math.degrees(math.atan2(dy, dx)) - self.angle + 180) % 360 - 180
        if abs(relative) <= 45:
            return "ahead"
        if relative > 135 or relative < -135:
            return "behind"
        return "left" if relative > 0 else "right"
