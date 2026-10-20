"""Turns raw ViZDoom state into semantic facts: health, ammo, visible enemies and
items, each with a direction and a distance. `build` is pure and testable;
`perceive` reads the DoomGame from engine.py. The legacy loop in main.py merges this
with navigator.py's spatial facts into the state sent to the Jev in brain.py."""

from dataclasses import asdict, dataclass

import vizdoom as vzd

from jevdoom import config

MONSTERS = frozenset({
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre",
    "Cacodemon", "LostSoul", "HellKnight", "BaronOfHell", "Arachnotron",
    "PainElemental", "Revenant", "Fatso", "Mancubus", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS",
})

MONSTER_PREFIXES = ("Marine",)

HEALTH_ITEMS = frozenset({"Medikit", "Stimpack", "HealthBonus"})

KEYS = frozenset({
    "RedCard", "BlueCard", "YellowCard", "RedSkull", "BlueSkull", "YellowSkull",
})

PICKUPS = KEYS | frozenset({
    "GreenArmor", "BlueArmor", "Medikit", "Stimpack", "HealthBonus", "ArmorBonus",
    "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "Cell", "Backpack",
    "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG9000",
})


@dataclass(frozen=True)
class Sighting:
    type: str
    direction: str
    distance: str


@dataclass(frozen=True)
class Perception:
    health: int
    ammo: int
    enemies: tuple[Sighting, ...]
    items: tuple[Sighting, ...]

    def to_state(self) -> dict:
        return {
            "health": self.health,
            "ammo": self.ammo,
            "enemies": [asdict(s) for s in self.enemies],
            "enemy_ahead": any(s.direction == "ahead" for s in self.enemies),
            "items": [asdict(s) for s in self.items],
            "key_visible": any(s.type in KEYS for s in self.items),
            "low_health": self.health < config.LOW_HEALTH,
            "health_item": next(
                ({"direction": s.direction, "distance": s.distance}
                 for s in self.items if s.type in HEALTH_ITEMS),
                None,
            ),
        }


def is_monster(name: str) -> bool:
    return name in MONSTERS or name.startswith(MONSTER_PREFIXES)


def _direction(center_x: float, screen_w: int) -> str:
    """The "ahead" band is narrow on purpose: it means "I can hit it", not "it is
    roughly in front of me". Widening it makes the agent shoot off-axis and run out
    of ammo (measured: 33% of the screen yields 2 kills, 10% yields 15)."""
    half = screen_w * config.CENTER_FRAC / 2
    if center_x < screen_w / 2 - half:
        return "left"
    if center_x > screen_w / 2 + half:
        return "right"
    return "ahead"


def _distance(height: int, screen_h: int) -> str:
    if height >= config.NEAR_H_FRAC * screen_h:
        return "near"
    if height >= config.MID_H_FRAC * screen_h:
        return "mid"
    return "far"


def build(
    health: float,
    ammo: float,
    screen_w: int,
    screen_h: int,
    objects: list[tuple[str, int, int, int, int]],
) -> Perception:
    enemies: list[tuple[int, Sighting]] = []
    items: list[tuple[int, Sighting]] = []
    for name, x, _y, w, h in objects:
        if is_monster(name):
            if w < config.MIN_W or h < config.MIN_H:
                continue
            bucket = enemies
        elif name in PICKUPS:
            bucket = items
        else:
            continue
        sighting = Sighting(name, _direction(x + w / 2, screen_w), _distance(h, screen_h))
        bucket.append((h, sighting))

    enemies.sort(key=lambda pair: pair[0], reverse=True)
    items.sort(key=lambda pair: pair[0], reverse=True)
    return Perception(
        int(health),
        int(ammo),
        tuple(s for _, s in enemies),
        tuple(s for _, s in items),
    )


def perceive(game: vzd.DoomGame) -> Perception:
    state = game.get_state()
    objects = (
        [(l.object_name, l.x, l.y, l.width, l.height) for l in state.labels]
        if state is not None
        else []
    )
    return build(
        game.get_game_variable(vzd.GameVariable.HEALTH),
        game.get_game_variable(vzd.GameVariable.AMMO2),
        game.get_screen_width(),
        game.get_screen_height(),
        objects,
    )
