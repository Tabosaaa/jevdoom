"""Motor control: turns the TACTIC chosen by the Jev into buttons, like a human
player's hands. Aims with a "mouse" (precise turn in degrees, with a speed limit per
tic), turns while running, strafes while keeping aim, steps behind a corner without
taking its eyes off the enemy and presses "use" on reaching a door. The Jev decides
WHAT to do; this module only decides HOW, from what is on screen (labels) and from
level.py's floor plan. It also holds the reflex: the hand-written tactic policy used
by autopilot mode and as the Jev's fallback."""

import math
from dataclasses import dataclass

import vizdoom as vzd

from jevdoom import config, perception

MAX_TURN_PER_TIC = 12.0
AIM_TOLERANCE = 4.0
AIM_REACH = 45.0
WALK_TOLERANCE = 50.0
CORPSE_RATIO = 0.55
ENGAGE_RANGE = 1200
ITEM_DETOUR = 256
RETREAT_RANGE = 400
CLOSE_RANGE = 600
POINT_BLANK = 256
SEARCH_TURN = 60.0
THREAT_MEMORY = 12  # steps it remembers where the last enemy was seen
STRAFE_PERIOD = 4
MELEE = frozenset({"Demon", "Spectre", "LostSoul"})
MELEE_RANGE = 250
HITSCAN_WEIGHT = 0.5
CLAW_RANGE = 200  # an Imp up close claws hard
HITSCAN = frozenset({"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS", "SpiderMastermind"})
WEAPONS = frozenset({"Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle"})


@dataclass(frozen=True)
class Seen:
    type: str
    x: float
    y: float
    distance: float
    angle: float  # degrees relative to the aim; positive = to the left


@dataclass(frozen=True)
class Context:
    """What the hands need to know to carry out any tactic this step."""
    enemies: tuple[Seen, ...] = ()
    item: Seen | None = None
    route_angle: float | None = None
    press_use: bool = False  # standing at a door/switch and has not pressed for a while
    threat_angle: float | None = None  # where an enemy was last seen (if none is seen now)
    cover_angle: float | None = None   # direction of the nearest cover point
    escape_angle: float | None = None  # direction to back off that opens the most distance
    step: int = 0


def relative_angle(angle: float, x: float, y: float, tx: float, ty: float) -> float:
    return (math.degrees(math.atan2(ty - y, tx - x)) - angle + 180) % 360 - 180


def _alive(label, distance: float, screen_h: int) -> bool:
    """Corpses stay in the labels under the same name. A standing monster's height on
    screen is proportional to 1/distance; a body lying down takes much less."""
    expected = screen_h * 56 / max(distance, 1.0)
    return label.height >= CORPSE_RATIO * expected


def look(game: vzd.DoomGame, x: float, y: float, angle: float) -> tuple[list[Seen], list[Seen]]:
    """Live enemies and visible items, nearest first."""
    state = game.get_state()
    if state is None:
        return [], []
    screen_h = game.get_screen_height()
    enemies, items = [], []
    for label in state.labels:
        tx, ty = label.object_position_x, label.object_position_y
        seen = Seen(label.object_name, tx, ty, math.dist((x, y), (tx, ty)), relative_angle(angle, x, y, tx, ty))
        if perception.is_monster(label.object_name):
            if _alive(label, seen.distance, screen_h):
                enemies.append(seen)
        elif label.object_name in perception.PICKUPS:
            items.append(seen)
    return sorted(enemies, key=lambda s: s.distance), sorted(items, key=lambda s: s.distance)


def prioritize(enemies: list[Seen]) -> list[Seen]:
    """A good player's kill order: a hitscan shooter counts as if it were at half the
    distance (it hurts from afar and goes down fast — a ShotgunGuy has 30 health);
    the rest, by distance."""
    return sorted(enemies, key=lambda e: e.distance * (HITSCAN_WEIGHT if e.type in HITSCAN else 1.0))


@dataclass
class Command:
    """One control step: total desired turn (degrees, + = left) and buttons."""
    turn: float = 0.0
    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    attack: bool = False
    use: bool = False


def _move(direction: float) -> dict:
    """Movement buttons to walk toward `direction` (degrees relative to the aim,
    + = left) without turning — forward/back combined with strafing, like WASD."""
    return {
        "forward": abs(direction) < 67.5,
        "back": abs(direction) > 112.5,
        "left": 22.5 < direction < 157.5,
        "right": -157.5 < direction < -22.5,
    }


def command(tactic: str, ctx: Context) -> Command:
    """`press_use` comes from the session: pressing "use" again on a door that is
    opening closes it, so it only presses while standing at it, with a cooldown."""
    target = ctx.enemies[0] if ctx.enemies else None
    turn = target.angle if target else 0.0
    # the turn happens first and the trigger after it (execute): the target only has to be within the turn
    aligned = target is not None and abs(turn) <= AIM_REACH
    if tactic == "attack":
        # shoots while strafing, switching sides every STRAFE_PERIOD steps (circle-strafe):
        # an Imp fireball is a projectile and misses a moving target
        left = ctx.step // STRAFE_PERIOD % 2 == 0
        return Command(turn=turn, attack=aligned, left=left, right=not left)
    if tactic in ("strafe_left", "strafe_right"):
        return Command(turn=turn, attack=aligned, left=tactic == "strafe_left", right=tactic == "strafe_right")
    if tactic == "retreat":
        if ctx.escape_angle is not None:
            return Command(turn=turn, attack=aligned, **_move(ctx.escape_angle - turn))
        return Command(turn=turn, attack=aligned, back=True)
    if tactic == "take_cover" and ctx.cover_angle is not None:
        # heads for cover keeping aim on the enemy: movement relative to the already-turned aim
        return Command(turn=turn, attack=aligned, **_move(ctx.cover_angle - turn))
    if tactic == "search":
        # turns and steps away from the wall: a shooter on top of a ledge only shows up
        # on screen after a few steps back
        return Command(turn=ctx.threat_angle if ctx.threat_angle is not None else SEARCH_TURN, back=True)
    if tactic == "use":
        return Command(use=True)
    if tactic == "grab_item" and ctx.item is not None:
        return _walk_towards(ctx.item.angle, use=False)
    if ctx.route_angle is None:
        return Command(forward=True, use=ctx.press_use)
    return _walk_towards(ctx.route_angle, use=ctx.press_use)


def _walk_towards(relative: float, use: bool) -> Command:
    """Like a human: turns while running, and only stops moving forward in a sharp turn."""
    return Command(turn=relative, forward=abs(relative) <= WALK_TOLERANCE, use=use)


def wanted_item(items: list[Seen], health: float, ignored: set[tuple[int, int]]) -> Seen | None:
    """Only leaves the route for what changes the fight: a health item when hurt and a
    new weapon (the shotgun at the start of MAP01 wins the duels the pistol loses).
    The rest it picks up on the way. `ignored` holds items it tried and failed to
    reach (behind bars, up on a platform), otherwise it stays stuck staring at them."""
    wanted = WEAPONS | (perception.HEALTH_ITEMS if health < config.LOW_HEALTH else frozenset())
    return next((s for s in items if s.type in wanted and item_key(s) not in ignored), None)


def item_key(item: Seen) -> tuple[int, int]:
    return int(item.x), int(item.y)


def reflex(ctx: Context, health: float = 100.0, under_attack: bool = False) -> str:
    """The tactic a player would pick on autopilot. It is the fallback when the Jev is
    unsure and the policy of --autopilot mode (free, no API). Always shoots while
    strafing; a Demon up close means backing off; hurt with an enemy nearby, it gets
    out of the line of fire (cover) or backs off aiming — hitscan enemies' shots
    spread by angle, so distance and corners reduce damage. Shot by someone it cannot
    see, it turns to where it last saw an enemy. Every rule here came from measuring
    batches of games (see PLAN.md)."""
    enemies = ctx.enemies
    shooters = any(e.type in HITSCAN and e.distance < ENGAGE_RANGE for e in enemies)
    # a weapon on the floor is worth the detour, but not running into the shooters' room
    weapon_near = (ctx.item is not None and ctx.item.type in WEAPONS and ctx.item.distance < ITEM_DETOUR
                   and not shooters)
    engage = ENGAGE_RANGE if under_attack else CLOSE_RANGE  # long-range duels only when already being shot
    if enemies and enemies[0].distance < engage and not (weapon_near and enemies[0].distance > MELEE_RANGE):
        nearest = enemies[0]
        if nearest.type in MELEE and nearest.distance < MELEE_RANGE:
            # a Demon has 150 health and bites: back off toward open ground shooting (the
            # player outruns it); with no room, step aside
            if ctx.escape_angle is not None:
                return "retreat"
            return "strafe_left" if ctx.step // STRAFE_PERIOD % 2 else "strafe_right"
        if nearest.type not in HITSCAN and nearest.distance < CLAW_RANGE and ctx.escape_angle is not None:
            return "retreat"  # Imp up close: back off shooting, out of claw range
        if nearest.distance < POINT_BLANK:
            return "attack"  # point blank: turning your back to reach cover means taking the whole blast
        outgunned = sum(e.type in HITSCAN and e.distance < ENGAGE_RANGE for e in enemies) >= 2
        if outgunned and ctx.cover_angle is not None:
            return "take_cover"  # several shooters: fight from the corner, peeking
        if outgunned and ctx.escape_angle is not None:
            return "retreat"  # no corner nearby: back off shooting and let them come one by one
        if health < config.LOW_HEALTH and nearest.distance < RETREAT_RANGE:
            return "take_cover" if ctx.cover_angle is not None else "retreat"
        return "attack"
    if under_attack and not enemies:
        return "search"
    if ctx.item is not None and ctx.item.distance < ITEM_DETOUR:
        return "grab_item"
    return "follow_route"


def buttons_vector(game: vzd.DoomGame, cmd: Command, turn: float, attack: bool) -> list[float]:
    values = {
        vzd.Button.MOVE_FORWARD: cmd.forward, vzd.Button.MOVE_BACKWARD: cmd.back,
        vzd.Button.MOVE_LEFT: cmd.left, vzd.Button.MOVE_RIGHT: cmd.right,
        vzd.Button.SPEED: True, vzd.Button.USE: cmd.use, vzd.Button.ATTACK: attack,
    }
    # in ViZDoom a positive delta turns right
    return [-turn if b == vzd.Button.TURN_LEFT_RIGHT_DELTA else float(values.get(b, False))
            for b in game.get_available_buttons()]


def execute(game: vzd.DoomGame, cmd: Command, tics: int, on_tic=None) -> float:
    """Applies the command tic by tic, like a hand on a mouse: turns at a capped speed
    per tic and only presses the trigger when the weapon is ready, releasing between
    shots. That is a good player's "tap-fire": in Doom only the first shot of each
    click comes out without spread; holding the button sprays the shots and loses
    long-range duels against shotgunners. `on_tic(game)` is called every tic (the UI
    uses it to stream frames)."""
    turn_tics = min(tics, max(1, math.ceil(abs(cmd.turn) / MAX_TURN_PER_TIC)))
    reward = 0.0
    for tic in range(tics):
        if game.is_episode_finished():
            break
        turn = cmd.turn / turn_tics if tic < turn_tics else 0.0
        attack = cmd.attack and tic >= turn_tics - 1 and _weapon_ready(game)
        reward += game.make_action(buttons_vector(game, cmd, turn, attack), 1)
        if on_tic is not None and not game.is_episode_finished():
            on_tic(game)
    return reward


def _weapon_ready(game: vzd.DoomGame) -> bool:
    return game.get_game_variable(vzd.GameVariable.ATTACK_READY) > 0
