"""Tests for the floor plan, the objectives and the hands. They read the real maps
from freedoom2.wad (shipped with vizdoom), without opening the game."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import vizdoom as vzd

from jevdoom import level, objectives, pilot, wad

WAD = os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad")
START = (-192.0, -192.0)


def load(name: str) -> level.Level:
    return level.Level(wad.read_map(WAD, name))


@pytest.fixture(scope="module")
def map01():
    return load("MAP01")


@pytest.fixture(scope="module")
def map02():
    return load("MAP02")


def exit_field(lv: level.Level) -> level.Field:
    planner = objectives.Planner(lv)
    return lv.field({o.id: o.cells for o in planner.objectives if o.kind == "exit"})


def test_reads_the_maps_from_the_wad():
    names = wad.map_names(WAD)
    assert names[0] == "MAP01" and len(names) == 32
    data = wad.read_map(WAD, "MAP01")
    assert any(l.special in wad.EXITS_USE for l in data.lines)
    with pytest.raises(ValueError):
        wad.read_map(WAD, "MAP99")


def test_there_is_a_path_from_the_start_to_the_exit(map01):
    route = map01.route(exit_field(map01), *START)
    assert route is not None and 3000 < route.distance < 6000


def test_the_exit_is_only_reached_from_the_front_of_the_switch(map01):
    """MAP01's exit switch faces south (y < -632); from the north, "use" triggers the
    other switch of the same box."""
    planner = objectives.Planner(map01)
    exit_cells = next(o.cells for o in planner.objectives if o.kind == "exit")
    assert exit_cells and all(map01.center(c)[1] < -632 for c in exit_cells)


def test_going_downhill_always_gets_closer_to_the_target(map01):
    field = exit_field(map01)
    start = map01.nearest_reachable(field, *START)
    distances = [field.dist[c] for c in map01.downhill(field, start, 400)]
    assert all(b < a for a, b in zip(distances, distances[1:]))


def test_a_thin_trench_can_be_walked_over(map01):
    """8-unit trenches (sector 127, floor -480) do not stop a 32-wide player."""
    assert np.isfinite(exit_field(map01).dist[map01.cell(216, -720)])


def test_getting_stuck_blocks_the_stretch_and_forgetting_undoes_it():
    lv = load("MAP01")
    lv.mark_stuck(exit_field(lv), *START)
    lv.update_heights(None, frozenset())
    assert lv.learned.any()
    lv.forget()
    lv.update_heights(None, frozenset())
    assert not lv.learned.any() and lv.route(exit_field(lv), *START) is not None


def test_priority_makes_an_equally_close_target_lose(map01):
    a = map01.cells_around(-192, -128, 16)
    b = map01.cells_around(-192, -256, 16)
    assert a and b
    field = map01.field({0: a, 1: b}, penalty={0: 40})
    assert field.owner[map01.cell(*START)] == 1


def test_a_locked_door_only_opens_with_the_key(map02):
    """MAP02's exit is behind key doors."""
    exit_goal = {o.id: o.cells for o in objectives.Planner(map02).objectives if o.kind == "exit"}
    start = next(t for t in map02.data.things if t.kind == wad.PLAYER_START)
    map02.update_heights(None, frozenset())
    locked = map02.nearest_reachable(map02.field(exit_goal), start.x, start.y)
    map02.update_heights(None, frozenset({"red", "blue", "yellow"}))
    opened = map02.nearest_reachable(map02.field(exit_goal), start.x, start.y)
    map02.update_heights(None, frozenset())
    assert locked is None and opened is not None


def test_a_lift_connects_different_heights(map02):
    """MAP02's red key is 120 units up, past a lift."""
    key = map02.cells_around(856, -320, 24)
    assert np.isfinite(map02.field({0: key}).dist[map02.cell(1232, -230)])


def test_line_of_sight(map01):
    assert map01.visible(START, np.array([[-100.0, -192.0]]))[0]
    assert not map01.visible(START, np.array([[216.0, -720.0]]))[0]


def test_the_planner_starts_with_something_reachable_on_map01(map01):
    plan = objectives.Planner(map01).update(*START, objectives.World())
    assert plan.route is not None and plan.objective.kind in ("exit", "weapon", "armor", "health", "ammo")


def test_the_planner_looks_for_what_unlocks_a_locked_exit(map02):
    start = next(t for t in map02.data.things if t.kind == wad.PLAYER_START)
    armed = objectives.World(weapon_slots=frozenset(range(1, 8)), armor=100)
    plan = objectives.Planner(map02).update(start.x, start.y, armed)
    assert plan.objective is not None and plan.objective.kind != "exit"


def test_ammo_becomes_an_objective_when_it_runs_low(map01):
    planner = objectives.Planner(map01)
    shells = next(o for o in planner.objectives if o.kind == "ammo" and o.slot == 3)
    planner._pos = shells.aim
    planner.world = objectives.World(weapon_slots=frozenset({2, 3}), ammo={2: 50, 3: 2})
    assert planner._available(shells)
    planner.world = objectives.World(weapon_slots=frozenset({2, 3}), ammo={2: 50, 3: 30})
    assert not planner._available(shells)


def test_an_item_among_shooters_waits(map01):
    planner = objectives.Planner(map01)
    shotgun = next(o for o in planner.objectives if o.kind == "weapon")
    planner._pos = shotgun.aim
    planner.world = objectives.World(threats=((shotgun.aim[0] + 100, shotgun.aim[1]),))
    assert not planner._available(shotgun)


def seen(type="Zombieman", distance=300.0, angle=0.0, x=0.0, y=0.0):
    return pilot.Seen(type, x, y, distance, angle)


def test_reflex_attacks_up_close_follows_the_route_otherwise_and_searches_for_shooters():
    assert pilot.reflex(pilot.Context(enemies=(seen(),))) == "attack"
    assert pilot.reflex(pilot.Context()) == "follow_route"
    assert pilot.reflex(pilot.Context(enemies=(seen(distance=900),))) == "follow_route"
    assert pilot.reflex(pilot.Context(enemies=(seen(distance=900),)), under_attack=True) == "attack"
    assert pilot.reflex(pilot.Context(), under_attack=True) == "search"


def test_reflex_hurt_takes_cover_and_backs_off_from_a_demon():
    hurt = pilot.Context(enemies=(seen(distance=300),), cover_angle=90.0)
    assert pilot.reflex(hurt, health=20) == "take_cover"
    assert pilot.reflex(pilot.Context(enemies=(seen(distance=300),)), health=20) == "retreat"
    point_blank = pilot.Context(enemies=(seen(distance=100),), cover_angle=90.0)
    assert pilot.reflex(point_blank, health=20) == "attack"
    demon = pilot.Context(enemies=(seen("Demon", distance=100),), escape_angle=180.0)
    assert pilot.reflex(demon) == "retreat"
    three = tuple(seen("ShotgunGuy", distance=d) for d in (400, 500, 600))
    assert pilot.reflex(pilot.Context(enemies=three, cover_angle=90.0, escape_angle=180.0)) == "take_cover"
    assert pilot.reflex(pilot.Context(enemies=three, escape_angle=180.0)) == "retreat"


def test_shooters_come_first_in_the_kill_order():
    imp = seen("DoomImp", distance=300)
    shotgunner = seen("ShotgunGuy", distance=500)
    assert pilot.prioritize([imp, shotgunner])[0] is shotgunner


def test_only_detours_for_health_when_hurt_and_gives_up_on_the_unreachable():
    medikit = seen("Medikit", x=10, y=20)
    assert pilot.wanted_item([medikit], health=100, ignored=set()) is None
    assert pilot.wanted_item([medikit], health=30, ignored=set()) == medikit
    assert pilot.wanted_item([medikit], health=30, ignored={pilot.item_key(medikit)}) is None


def test_a_new_weapon_is_worth_the_detour_even_at_full_health():
    shotgun = seen("Shotgun")
    assert pilot.wanted_item([seen("Clip"), shotgun], health=100, ignored=set()) == shotgun


def test_attack_shoots_while_strafing_and_cover_moves_keeping_aim():
    cmd = pilot.command("attack", pilot.Context(enemies=(seen(angle=20),)))
    assert cmd.attack and cmd.turn == 20 and (cmd.left or cmd.right)
    cover = pilot.command("take_cover", pilot.Context(enemies=(seen(angle=0),), cover_angle=180.0))
    assert cover.back and not cover.forward and cover.turn == 0


def test_a_corpse_does_not_count_as_an_enemy():
    assert pilot._alive(SimpleNamespace(height=60), distance=200, screen_h=240)
    assert not pilot._alive(SimpleNamespace(height=12), distance=200, screen_h=240)


def test_positive_relative_angle_is_to_the_left():
    assert math.isclose(pilot.relative_angle(0, 0, 0, 0, 100), 90)
    assert math.isclose(pilot.relative_angle(90, 0, 0, 100, 0), -90)
