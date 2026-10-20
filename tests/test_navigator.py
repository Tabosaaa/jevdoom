"""Tests for the spatial memory. Uses a fake DoomGame, without opening the game."""

from jevdoom import config
from jevdoom.navigator import Navigator


def warmup(game, nav):
    """Consumes the first updates, during which stuck detection is off."""
    for _ in range(config.WARMUP_STEPS + 1):
        nav.update(game)


class FakeGame:
    def __init__(self, x: float = 0.0, y: float = 0.0, angle: float = 0.0) -> None:
        self.values = {"POSITION_X": x, "POSITION_Y": y, "ANGLE": angle, "HEALTH": 100.0}

    def move(self, x: float, y: float) -> None:
        self.values["POSITION_X"] = x
        self.values["POSITION_Y"] = y

    def get_game_variable(self, variable):
        return self.values[variable.name]


def test_turning_in_place_does_not_count_as_blocked():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    nav.update(game, "turn_left")
    nav.update(game, "turn_right")
    assert nav.facts()["blocked"] is False


def test_trying_to_walk_without_moving_blocks():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    assert nav.facts() == nav.facts() and nav.failed_moves == 1
    nav.update(game, "move_forward")
    assert nav.facts()["blocked_attempts"] == 2
    assert nav.facts()["blocked"] is True


def test_real_movement_clears_the_block():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    game.move(config.STUCK_DIST * 10, 0)
    nav.update(game, "move_forward")
    assert nav.facts()["blocked"] is False
    assert nav.facts()["already_tried_use"] is False


def test_use_marks_an_attempt_without_unblocking():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    nav.update(game, "use")
    facts = nav.facts()
    assert facts["already_tried_use"] is True
    assert facts["blocked"] is True


def test_a_discovered_wall_no_longer_looks_new():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    assert nav.facts()["path_ahead_new"] is True
    nav.update(game, "move_forward")
    assert nav.facts()["path_ahead_new"] is False


def test_compass_points_to_the_only_frontier():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    for x in range(1, 4):
        game.move(x * config.CELL_SIZE, 0)
        nav.update(game, "move_forward")

    for cx, cy in list(nav.visited):
        nav.walls.update({(cx, cy + 1), (cx, cy - 1)})
    nav.walls.add((-1, 0))

    assert nav.facts()["unexplored_bearing"] == "ahead"
    game.values["ANGLE"] = 180
    nav.update(game, "turn_left")
    assert nav.facts()["unexplored_bearing"] == "behind"


def test_counts_explored_areas():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    game.move(config.CELL_SIZE * 5, 0)
    nav.update(game, "move_forward")
    assert nav.facts()["explored_areas"] == 2


def test_turning_clears_the_block_because_it_changes_direction():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    nav.update(game, "use")
    assert nav.facts()["blocked"] is True
    nav.update(game, "turn_right")
    facts = nav.facts()
    assert facts["blocked"] is False
    assert facts["already_tried_use"] is False


def test_detects_damage_taken():
    game, nav = FakeGame(), Navigator()
    game.values["HEALTH"] = 100
    warmup(game, nav)
    assert nav.facts()["under_attack"] is False
    game.values["HEALTH"] = 70
    nav.update(game, "move_forward")
    assert nav.facts()["under_attack"] is True


def test_gives_up_on_an_unreachable_frontier():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    first = nav.target
    assert first is not None
    for _ in range(config.TARGET_PATIENCE + 1):
        nav.update(game, "move_forward")
    assert first in nav.unreachable
    assert nav.target != first


def test_use_marks_the_wall_as_tested():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    assert nav.facts()["untested_walls"] == 1
    nav.update(game, "use")
    assert nav.facts()["untested_walls"] == 0
    assert nav.facts()["untested_wall_bearing"] is None


def test_points_to_a_wall_not_tested_yet():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    assert nav.facts()["untested_wall_bearing"] == "ahead"


def test_a_visited_cell_stops_being_a_wall():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    nav.update(game, "move_forward")
    wall = next(iter(nav.walls))
    game.move(wall[0] * config.CELL_SIZE + 1, wall[1] * config.CELL_SIZE + 1)
    nav.update(game, "move_forward")
    assert wall not in nav.walls


def test_counts_steps_without_discovering_new_ground():
    game, nav = FakeGame(angle=0), Navigator()
    warmup(game, nav)
    assert nav.facts()["steps_without_discovery"] >= 1
    game.move(config.CELL_SIZE * 4, 0)
    nav.update(game, "move_forward")
    assert nav.facts()["steps_without_discovery"] == 0
    nav.update(game, "move_forward")
    nav.update(game, "move_forward")
    assert nav.facts()["steps_without_discovery"] == 2


def test_counts_stationary_steps_even_while_turning():
    game, nav = FakeGame(), Navigator()
    warmup(game, nav)
    base = nav.facts()["stationary_for"]
    nav.update(game, "turn_right")
    nav.update(game, "use")
    nav.update(game, "move_forward")
    assert nav.facts()["stationary_for"] == base + 3
    game.move(config.STUCK_DIST * 10, 0)
    nav.update(game, "move_forward")
    assert nav.facts()["stationary_for"] == 0


def test_unsticking_cycles_the_manoeuvre_and_starts_with_a_strafe():
    from jevdoom.main import ESCAPE_MOVES, escape_move

    assert escape_move(config.ESCAPE_AFTER) == "strafe_right"
    moves = [escape_move(config.ESCAPE_AFTER + i) for i in range(len(ESCAPE_MOVES))]
    assert moves == list(ESCAPE_MOVES)
    assert escape_move(config.ESCAPE_AFTER + len(ESCAPE_MOVES)) == ESCAPE_MOVES[0]
