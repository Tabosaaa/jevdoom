"""Tests for the pure function perception.build. Runs without the game open."""

from jevdoom.perception import build

W, H = 320, 240

OBJECTS = [
    ("GreenArmor", 157, 122, 4, 2),
    ("Zombieman", 207, 106, 25, 67),
    ("ShotgunGuy", 87, 108, 23, 66),
    ("DoomPlayer", 150, 226, 19, 13),
]


def test_splits_enemies_from_items_and_ignores_the_rest():
    p = build(64, 52, W, H, OBJECTS)
    assert (p.health, p.ammo) == (64, 52)
    assert {s.type for s in p.enemies} == {"Zombieman", "ShotgunGuy"}
    assert {s.type for s in p.items} == {"GreenArmor"}


def test_direction_by_screen_band():
    objs = [
        ("Zombieman", 0, 100, 20, 80),
        ("Zombieman", 150, 100, 20, 80),
        ("Zombieman", 300, 100, 20, 80),
    ]
    assert {s.direction for s in build(100, 10, W, H, objs).enemies} == {"left", "ahead", "right"}


def test_distance_by_height():
    objs = [
        ("Demon", 150, 10, 20, 130),
        ("Demon", 150, 10, 20, 60),
        ("Demon", 150, 10, 20, 20),
    ]
    assert {s.distance for s in build(100, 10, W, H, objs).enemies} == {"near", "mid", "far"}


def test_sorts_nearest_first():
    objs = [
        ("Zombieman", 150, 10, 20, 40),
        ("ShotgunGuy", 150, 10, 20, 130),
    ]
    assert build(100, 10, W, H, objs).enemies[0].type == "ShotgunGuy"


def test_ignores_tiny_slivers():
    assert build(100, 10, W, H, [("Zombieman", 141, 122, 0, 1)]).enemies == ()


def test_detects_a_visible_key():
    state = build(100, 10, W, H, [("BlueCard", 150, 120, 10, 14)]).to_state()
    assert state["key_visible"] is True
    assert state["items"][0]["type"] == "BlueCard"


def test_enemy_ahead():
    centered = build(100, 10, W, H, [("Demon", 150, 10, 20, 130)]).to_state()
    aside = build(100, 10, W, H, [("Demon", 0, 10, 20, 130)]).to_state()
    assert centered["enemy_ahead"] is True
    assert aside["enemy_ahead"] is False


def test_points_to_a_health_item_when_health_is_low():
    objs = [("Stimpack", 0, 120, 12, 14), ("Clip", 150, 120, 10, 10)]
    state = build(30, 10, W, H, objs).to_state()
    assert state["low_health"] is True
    assert state["health_item"] == {"direction": "left", "distance": "far"}


def test_no_health_item_in_sight():
    state = build(30, 10, W, H, [("Clip", 150, 120, 10, 10)]).to_state()
    assert state["low_health"] is True
    assert state["health_item"] is None
