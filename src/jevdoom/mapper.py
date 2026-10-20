"""EXPERIMENT, MEASURED AND REMOVED FROM THE LOOP — see "Phase 8" in PLAN.md.

Reads ViZDoom's automap (WHOLE mode, which draws the level's entire floor plan) and
points to where there is level geometry not yet visited. It works out the
pixel<->world transform by itself by watching the player marker. It complemented
navigator.py, whose frontier only knows what has already been touched: here we see
rooms we never visited.

Known limit: the automap is a line drawing and its colors do not separate
walkable borders from solid walls, so you cannot flood the reachable area or plan a
route. It tells where the level is, not how to get there — and without reachability
the target aimed at rooms behind walls, which explored LESS than navigator.py's
frontier (26 areas vs 42-47). Nothing in the game loop uses this module; level.py
reads the real floor plan from the WAD instead.

What is solid and reusable here: the pixel<->world calibration by Theil-Sen, which
finds the real scale (~11.3 units/px) and validates itself by requiring both axis
scales to agree."""

import numpy as np
import vizdoom as vzd

from jevdoom import config

MIN_SAMPLES = 20
MAX_SAMPLES = 80
MIN_PAIR_PX = 8.0
MIN_PAIRS = 20
MAX_SCALE_DISAGREEMENT = 0.15


class Mapper:
    def __init__(self) -> None:
        self.samples: list[tuple[float, float, float, float]] = []
        self.fit_x: tuple[float, float] | None = None
        self.fit_y: tuple[float, float] | None = None
        self.target_world: tuple[float, float] | None = None
        self.rejected: set[tuple[int, int]] = set()
        self._target_cell: tuple[int, int] | None = None
        self._best_distance = float("inf")
        self._age = 0

    @property
    def calibrated(self) -> bool:
        return self.fit_x is not None and self.fit_y is not None

    def update(self, game: vzd.DoomGame, visited: set[tuple[int, int]]) -> None:
        state = game.get_state()
        if state is None or state.automap_buffer is None:
            return
        buffer = state.automap_buffer
        marker = self._marker(buffer)
        if marker is None:
            return

        self.samples.append((
            game.get_game_variable(vzd.GameVariable.POSITION_X),
            game.get_game_variable(vzd.GameVariable.POSITION_Y),
            marker[0],
            marker[1],
        ))
        if not self.calibrated:
            self._calibrate()
        if self.calibrated:
            self.target_world = self._pick_target(
                buffer, visited, marker, self.samples[-1][0], self.samples[-1][1]
            )

    def facts(self) -> dict:
        return {
            "mapa_calibrado": self.calibrated,
            "tem_sala_nova_no_mapa": self.target_world is not None,
        }

    @staticmethod
    def _marker(buffer: np.ndarray) -> tuple[float, float] | None:
        image = buffer.astype(np.int16)
        mask = (image[:, :, 2] > 200) & (image[:, :, 0] < 60) & (image[:, :, 1] < 60)
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        return (float(xs.mean()), float(ys.mean()))

    @staticmethod
    def _structure(buffer: np.ndarray) -> np.ndarray:
        """A band around everything the automap drew, i.e. where the level is."""
        image = buffer.astype(np.int32)
        sample = image[::4, ::4].reshape(-1, 3)
        packed = (sample[:, 0] << 16) | (sample[:, 1] << 8) | sample[:, 2]
        background = np.bincount(packed).argmax()
        floor = np.array(
            [(background >> 16) & 255, (background >> 8) & 255, background & 255]
        )
        lines = np.abs(image - floor).sum(axis=2) > config.MAP_WALL_TOLERANCE
        zone = lines
        for _ in range(config.MAP_STRUCTURE_BAND):
            grown = zone.copy()
            grown[1:, :] |= zone[:-1, :]
            grown[:-1, :] |= zone[1:, :]
            grown[:, 1:] |= zone[:, :-1]
            grown[:, :-1] |= zone[:, 1:]
            zone = grown
        return zone

    def _calibrate(self) -> None:
        """Fits pixel->world by Theil-Sen (median of slopes between well separated
        pairs). The player marker is a rotating arrow, so its center carries a bias
        that follows the direction of motion: a least-squares fit absorbs that bias,
        the median does not. Since the automap has a square aspect, requiring both
        scales to agree discards bad fits."""
        if len(self.samples) < MIN_SAMPLES:
            return
        data = np.array(self.samples[-MAX_SAMPLES:])
        fit_x = self._robust_fit(data[:, 0], data[:, 2])
        fit_y = self._robust_fit(data[:, 1], data[:, 3])
        if fit_x is None or fit_y is None:
            return

        scales = (abs(fit_x[0]), abs(fit_y[0]))
        if abs(scales[0] - scales[1]) / max(scales) > MAX_SCALE_DISAGREEMENT:
            return
        self.fit_x, self.fit_y = fit_x, fit_y

    @staticmethod
    def _robust_fit(world: np.ndarray, pixel: np.ndarray) -> tuple[float, float] | None:
        left, right = np.triu_indices(len(world), 1)
        span = pixel[right] - pixel[left]
        usable = np.abs(span) >= MIN_PAIR_PX
        if usable.sum() < MIN_PAIRS:
            return None
        slope = float(np.median((world[right] - world[left])[usable] / span[usable]))
        return slope, float(np.median(world - slope * pixel))

    def to_world(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ax, bx = self.fit_x
        ay, by = self.fit_y
        return ax * px + bx, ay * py + by

    def _pick_target(
        self,
        buffer: np.ndarray,
        visited: set[tuple[int, int]],
        marker: tuple[float, float],
        px_world: float,
        py_world: float,
    ) -> tuple[float, float] | None:
        ys, xs = np.nonzero(self._structure(buffer))
        if len(xs) == 0:
            return None
        step = max(1, len(xs) // config.MAP_MAX_CANDIDATES)
        xs, ys = xs[::step], ys[::step]
        wx, wy = self.to_world(xs.astype(float), ys.astype(float))
        cells = np.stack([wx // config.CELL_SIZE, wy // config.CELL_SIZE], axis=1).astype(int)

        here = (int(px_world // config.CELL_SIZE), int(py_world // config.CELL_SIZE))
        if self._target_cell is not None:
            if self._target_cell in visited:
                self._target_cell = None
            else:
                distance = abs(here[0] - self._target_cell[0]) + abs(here[1] - self._target_cell[1])
                if distance < self._best_distance:
                    self._best_distance, self._age = distance, 0
                else:
                    self._age += 1
                    if self._age >= config.TARGET_PATIENCE:
                        self.rejected.add(self._target_cell)
                        self._target_cell = None

        if self._target_cell is None:
            fresh = [
                i for i, cell in enumerate(cells)
                if tuple(cell) not in visited and tuple(cell) not in self.rejected
            ]
            if not fresh:
                return None
            distances = (xs[fresh] - marker[0]) ** 2 + (ys[fresh] - marker[1]) ** 2
            best = fresh[int(np.argmin(distances))]
            self._target_cell = tuple(cells[best])
            self._best_distance = abs(here[0] - self._target_cell[0]) + abs(here[1] - self._target_cell[1])
            self._age = 0

        cx, cy = self._target_cell
        return ((cx + 0.5) * config.CELL_SIZE, (cy + 0.5) * config.CELL_SIZE)
