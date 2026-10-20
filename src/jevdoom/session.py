"""One game, start to finish, in any mode: the Jev decides, the hand-written reflex
decides (autopilot, free) or a person plays from the browser. Serves the CLI
(main.py) and the UI (server.py): everything that happens goes out as an event via
`emit(kind, data)` — a screen frame, each step's decision, the end of an episode —
and is logged to runs/*.jsonl. Levels with an exit use the floor plan (level +
objectives) and the hands (pilot); arenas without movement use the survival question."""

import base64
import json
import math
import os
import struct
import threading
import time
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import vizdoom as vzd

from jevdoom import actions, brain, config, engine, level, navigator, objectives, perception, pilot, wad

RUNS_DIR = "runs"
TIC_RATE = 35
FRAME_INTERVAL = 1 / 20  # seconds between frames sent to the UI
MODES = ("jev", "autopilot", "human")
HUMAN_TURN = 4.0  # degrees per tic while an arrow key is held
MOUSE_SCALE = 0.15  # degrees per mouse pixel
USE_COOLDOWN = 6  # steps between two "use" presses (a DR door closes if used while opening)


class NoExitError(ValueError):
    """The level has no exit the floor plan recognises (the CLI falls back to the legacy loop)."""


@dataclass(frozen=True)
class SessionConfig:
    scenario: str = "freedoom2"
    doom_map: str | None = "MAP01"
    mode: str = "jev"
    tics: int = config.DECISION_TICS
    episodes: int = 1
    max_steps: int | None = None
    visible: bool = False
    realtime: bool = False  # human game pace (35 tics/s × speed) — the UI turns it on
    speed: float = 1.0
    frames: bool = False
    skill: int = 3

    def validated(self) -> "SessionConfig":
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if not os.path.exists(os.path.join(engine.SCENARIOS_DIR, f"{self.scenario}.cfg")):
            raise ValueError(f"unknown scenario: {self.scenario}")
        if not 1 <= self.tics <= 35 or not 1 <= self.episodes <= 100 or not 0.1 <= self.speed <= 8:
            raise ValueError("tics, episodes or speed out of range")
        if not 1 <= self.skill <= 5:
            raise ValueError("difficulty goes from 1 to 5")
        iwad = engine.iwad_path(self.scenario)
        if iwad and self.doom_map and self.doom_map.upper() not in wad.map_names(iwad):
            raise ValueError(f"unknown map: {self.doom_map}")
        return self


class HumanInput:
    """Pressed keys and mouse movement coming from the browser (server thread)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: frozenset[str] = frozenset()
        self._dx = 0.0

    def set(self, keys, dx: float = 0.0) -> None:
        with self._lock:
            self._keys = frozenset(str(k) for k in keys)
            self._dx += float(dx)

    def take(self) -> tuple[frozenset[str], float]:
        with self._lock:
            dx, self._dx = self._dx, 0.0
            return self._keys, dx


def encode_png(rgb: np.ndarray) -> str:
    """Base64 PNG using only zlib (no imaging dependency)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode("ascii")


def new_log_path() -> str:
    """runs/YYYYMMDD-HHMMSS-mmm.jsonl, unique: two games started in the same millisecond
    (CLI and UI, or parallel benchmarks) must not write to the same file. The file is
    claimed here with an exclusive create, so the name is taken before anyone else checks."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    while True:
        now = time.time()
        path = os.path.join(RUNS_DIR, f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}.jsonl")
        try:
            open(path, "x").close()
            return path
        except FileExistsError:
            time.sleep(0.002)


class JevAdvisor:
    """Asks the Jev without pausing the game. At real-time pace (UI), a thread keeps
    asking, always with the latest state, while the game goes on with the last tactic
    decided — like a person, who does not freeze to think: the answer takes effect
    ~0.3–0.5 s later (reaction time). Without real-time pace (CLI, benchmarks) it waits
    for the answer and the Jev decides every step. Measured: synchronous, the screen
    was frozen 68% of the time (7 frames/s); this way, 10% (17 frames/s), same as
    autopilot."""

    def __init__(self, client, question, realtime: bool) -> None:
        self.client, self.question, self.realtime = client, question, realtime
        self._cond = threading.Condition()
        self._state: tuple[dict, int] | None = None  # latest (state, step)
        self._closed = False
        self.latest = None     # latest answer: (answer, ms)
        self.latest_step = -1  # step whose state produced that answer
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, name="jev", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        asked = -1
        while True:
            with self._cond:
                while not self._closed and (self._state is None or self._state[1] == asked):
                    self._cond.wait()
                if self._closed:
                    return
                state, asked = self._state
            started = time.monotonic()
            try:
                answer = brain.choose_action(self.client, state, self.question)
            except Exception as error:  # the session re-raises it in the game loop
                with self._cond:
                    self._error = error
                    self._cond.notify_all()
                return
            with self._cond:
                self.latest, self.latest_step = (answer, (time.monotonic() - started) * 1000), asked
                self._cond.notify_all()

    def advise(self, state: dict, step: int) -> tuple[object, float] | None:
        """The latest answer (None until the first arrives; the reflex holds until then)."""
        with self._cond:
            self._state = (state, step)
            self._cond.notify_all()
            if not self.realtime:
                while self.latest_step != step and self._error is None:
                    self._cond.wait()
            if self._error is not None:
                raise self._error
            return self.latest

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


_LEVELS: dict[tuple[str, str], level.Level | None] = {}


def load_level(scenario: str, doom_map: str | None) -> level.Level | None:
    """The level's floor plan, if the scenario is an IWAD and the map has an exit."""
    iwad = engine.iwad_path(scenario)
    name = (doom_map or "MAP01").upper()
    if iwad is None:
        return None
    if (scenario, name) not in _LEVELS:
        try:
            data = wad.read_map(iwad, name)
        except (ValueError, KeyError):
            _LEVELS[(scenario, name)] = None
        else:
            has_exit = any(l.special in wad.EXITS_USE | wad.EXITS_WALK for l in data.lines)
            _LEVELS[(scenario, name)] = level.Level(data) if has_exit else None
    return _LEVELS[(scenario, name)]


def minimap(lv: level.Level) -> dict:
    """Static floor plan for the UI: floor in shades of gray, dark walls."""
    floor = lv.floor
    valid = ~lv.static_blocked
    lo, hi = np.percentile(floor[valid], [2, 98]) if valid.any() else (0, 1)
    shade = np.clip((floor - lo) / max(hi - lo, 1) * 120 + 70, 40, 200).astype(np.uint8)
    rgb = np.where(valid[..., None], shade[..., None].repeat(3, 2), np.uint8(18))
    return {"png": encode_png(np.ascontiguousarray(rgb[::-1])), "origin": lv.origin,
            "cell": level.CELL, "shape": lv.shape}


class Session:
    def __init__(self, cfg: SessionConfig, emit=lambda kind, data: None,
                 should_stop=lambda: False, human: HumanInput | None = None) -> None:
        self.cfg = cfg.validated()
        self.emit = emit
        self.should_stop = should_stop
        self.human = human or HumanInput()
        self._last_frame = 0.0
        self._clock = time.monotonic()

    # ---------- running ----------

    def run(self) -> list[dict]:
        cfg = self.cfg
        lv = load_level(cfg.scenario, cfg.doom_map)
        game = engine.make_game(scenario=cfg.scenario, visible=cfg.visible, skill=cfg.skill,
                                doom_map=cfg.doom_map if engine.iwad_path(cfg.scenario) else None)
        survival = not actions.supports_movement(game)
        if lv is None and not survival and cfg.mode != "human":
            game.close()
            raise NoExitError("this level has no exit the floor plan recognises; use human mode")
        client = brain.make_client() if cfg.mode == "jev" else None
        log_path = new_log_path()
        results = []
        self.emit("start", {"config": asdict(cfg), "log": log_path, "survival": survival,
                            "map": minimap(lv) if lv is not None else None})
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                self._log = log
                for ep in range(1, cfg.episodes + 1):
                    if self.should_stop():
                        break
                    game.new_episode()
                    started = time.monotonic()
                    if cfg.mode == "human":
                        steps = self._play_human(game, ep, survival)
                    else:
                        question = brain.SURVIVAL_QUESTION if survival else brain.TACTIC_QUESTION
                        advisor = JevAdvisor(client, question, cfg.realtime) if client is not None else None
                        try:
                            steps = (self._play_arena(game, ep, advisor) if survival
                                     else self._play_route(game, lv, ep, advisor))
                        finally:
                            if advisor is not None:
                                advisor.close()
                    results.append(self._episode_result(game, ep, steps, survival, time.monotonic() - started))
                    self.emit("episode", results[-1])
        finally:
            if client is not None:
                client.close()
            game.close()
        self.emit("end", {"results": results, "log": log_path})
        return results

    def _episode_result(self, game, ep: int, steps: int, survival: bool, seconds: float) -> dict:
        kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        exited = game.get_total_reward() > 0 and not survival and game.is_episode_finished() and not game.is_player_dead()
        outcome = ("level complete" if exited else "died" if game.is_player_dead()
                   else "survived" if survival and game.is_episode_finished() else "stopped")
        return {"episode": ep, "steps": steps, "kills": kills, "outcome": outcome,
                "health": int(game.get_game_variable(vzd.GameVariable.HEALTH)), "seconds": round(seconds, 1)}

    def _on_tic(self, game) -> None:
        """Human pace (if on) and frames for the UI."""
        if self.cfg.realtime:
            self._clock += 1 / (TIC_RATE * self.cfg.speed)
            delay = self._clock - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._clock = time.monotonic()  # running late: do not try to catch up
        if self.cfg.frames and time.monotonic() - self._last_frame >= FRAME_INTERVAL:
            state = game.get_state()
            if state is not None:
                self._last_frame = time.monotonic()
                self.emit("frame", {
                    "png": encode_png(state.screen_buffer),
                    "w": game.get_screen_width(), "h": game.get_screen_height(),
                    "labels": [{"name": l.object_name, "x": l.x, "y": l.y, "w": l.width, "h": l.height,
                                "enemy": perception.is_monster(l.object_name)}
                               for l in state.labels if l.object_name != "DoomPlayer"],
                })

    def _record(self, data: dict) -> None:
        data = {**data, "scenario": self.cfg.scenario, "map": self.cfg.doom_map}
        self._log.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.emit("step", data)

    def _decide(self, advisor: JevAdvisor | None, state: dict, step: int, fallback: str) -> tuple[str, object, float, int]:
        """The Jev's tactic (its latest answer) or the reflex, below the confidence gate
        or while the first answer has not arrived."""
        if advisor is None:
            return fallback, None, 0.0, step
        latest = advisor.advise(state, step)
        if latest is None:
            return fallback, None, 0.0, step
        answer, ms = latest
        choice = answer.choice if answer.confidence >= config.MIN_CONFIDENCE else fallback
        return choice, answer, ms, advisor.latest_step

    # ---------- levels with a floor plan ----------

    def _play_route(self, game, lv: level.Level, ep: int, advisor: JevAdvisor | None) -> int:
        """The Jev (or the reflex) picks the tactic; pilot carries it out. Liveness lives
        in code: with no progress toward the objective, the floor plan learns that the
        way is blocked there."""
        lv.reset()
        planner = objectives.Planner(lv)
        key_colors = {wad.KEY_THINGS[t.kind] for t in lv.data.things if t.kind in wad.KEY_THINGS}
        nav = navigator.Navigator()
        steps, tactic, item = 0, None, None
        no_progress, best = 0, math.inf
        shown = None
        last_use = -USE_COOLDOWN
        ignored_items: set[tuple[int, int]] = set()
        threat: tuple[float, float, int] | None = None
        while not game.is_episode_finished() and not self.should_stop() and \
                (self.cfg.max_steps is None or steps < self.cfg.max_steps):
            nav.update(game, tactic)
            x, y = nav.position
            health = game.get_game_variable(vzd.GameVariable.HEALTH)
            enemies, items = pilot.look(game, x, y, nav.angle)
            enemies = pilot.prioritize(enemies)
            world = self._world(game, key_colors)
            world = objectives.World(**{**world.__dict__, "threats": tuple(
                (e.x, e.y) for e in enemies if e.type in pilot.HITSCAN and e.distance < pilot.ENGAGE_RANGE)})
            plan = planner.update(x, y, world)
            route = plan.route
            view = planner.objectives_view()
            if view != shown:
                shown = view
                self.emit("objectives", {"episode": ep, "objectives": view})

            progress = (route.distance if tactic == "follow_route" and route else
                        math.dist((x, y), (item.x, item.y)) if tactic == "grab_item" and item else None)
            if progress is None or progress < best - config.PROGRESS_MIN:
                no_progress, best = 0, (progress if progress is not None else math.inf)
            else:
                no_progress += 1
            patience = config.ROUTE_STUCK * (4 if lv.waiting_zone(x, y) else 1)
            if no_progress >= patience:
                if tactic == "grab_item":
                    ignored_items.add(pilot.item_key(item))
                else:
                    planner.stuck(x, y)
                no_progress, best = 0, math.inf

            if enemies:
                threat = (enemies[0].x, enemies[0].y, steps)
            remembered = threat is not None and not enemies and steps - threat[2] <= pilot.THREAT_MEMORY
            item = pilot.wanted_item(items, health, ignored_items)
            cover = escape = None
            outgunned = sum(e.type in pilot.HITSCAN for e in enemies) >= 2
            if enemies and (health < config.LOW_HEALTH or outgunned):
                cover = lv.cover(x, y, (enemies[0].x, enemies[0].y))
            if enemies and (enemies[0].distance < pilot.RETREAT_RANGE or outgunned):
                escape = lv.escape(x, y, (enemies[0].x, enemies[0].y))
            ctx = pilot.Context(
                enemies=tuple(enemies), item=item,
                route_angle=pilot.relative_angle(nav.angle, x, y, *route.waypoint) if route else None,
                # a door on the way: only while standing at it (using it again closes it); at a
                # switch objective: press on arrival, moving or not
                press_use=steps - last_use >= USE_COOLDOWN and bool(route) and (
                    (route.use_ahead and nav.stationary_steps >= 1)
                    or (plan.objective is not None and plan.objective.kind in ("switch", "exit")
                        and route.distance <= 2 * level.CELL)),
                threat_angle=pilot.relative_angle(nav.angle, x, y, threat[0], threat[1]) if remembered else None,
                cover_angle=pilot.relative_angle(nav.angle, x, y, *cover) if cover else None,
                escape_angle=pilot.relative_angle(nav.angle, x, y, *escape) if escape else None,
                step=steps,
            )
            state = self._route_state(game, nav, ctx, route, planner)
            reflex = pilot.reflex(ctx, health, nav.took_damage)
            tactic, answer, ms, asked = self._decide(advisor, state, steps, reflex)
            cmd = pilot.command(tactic, ctx)
            if cmd.use:
                last_use = steps
            reward = pilot.execute(game, cmd, self.cfg.tics, self._on_tic)
            self._record({
                "episode": ep, "step": steps, "mode": self.cfg.mode, "action": tactic, "reflex": reflex,
                "confidence": answer.confidence if answer else None,
                "probabilities": dict(answer.probabilities) if answer else None,
                "decision_ms": round(ms), "reaction_steps": steps - asked, "state": state, "reward": reward,
                "pos": [round(x), round(y)], "angle": round(nav.angle),
                "waypoint": [round(v) for v in route.waypoint] if route else None,
                "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                "plan": {**planner.summary(), "target_kind": planner.current.kind if planner.current else None,
                         "target_pos": self._target_point(planner)},
            })
            steps += 1
        return steps

    @staticmethod
    def _world(game, key_colors: set[str]) -> objectives.World:
        """What the player knows about the world: live heights, keys, items on the map, weapons."""
        gstate = game.get_state()
        if gstate is None:
            return objectives.World()
        objects = tuple((o.name, o.position_x, o.position_y) for o in gstate.objects)
        present = {wad.KEY_NAMES.get(name) for name, _, _ in objects}
        slots = frozenset(i for i in range(1, 8)
                          if game.get_game_variable(getattr(vzd.GameVariable, f"WEAPON{i}")) > 0)
        return objectives.World(
            heights=[(s.floor_height, s.ceiling_height) for s in gstate.sectors or ()] or None,  # map without sectors_info
            keys=frozenset(key_colors - present), objects=objects,
            health=game.get_game_variable(vzd.GameVariable.HEALTH),
            armor=game.get_game_variable(vzd.GameVariable.ARMOR), weapon_slots=slots,
            ammo={2: game.get_game_variable(vzd.GameVariable.AMMO2), 3: game.get_game_variable(vzd.GameVariable.AMMO3)},
        )

    @staticmethod
    def _target_point(planner) -> list[int] | None:
        o = planner.current
        if o is None:
            return None
        x, y = o.point or planner.level.center(next(iter(o.cells)))
        return [round(x), round(y)]

    def _route_state(self, game, nav, ctx: pilot.Context, route, planner) -> dict:
        """What the Jev reads: what a player knows from looking at the screen and
        remembering the level. Concrete numbers (distance, angle) instead of labels."""
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        summary = planner.summary()
        seen = lambda s: {"type": s.type, "distance": int(s.distance), "angle": int(s.angle)}
        return {
            "health": int(health),
            "armor": int(game.get_game_variable(vzd.GameVariable.ARMOR)),
            "ammo": int(game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)),
            "low_health": health < config.LOW_HEALTH,
            "under_attack": nav.took_damage,
            "enemies": [seen(e) for e in ctx.enemies[:3]],
            "enemy_in_sights": bool(ctx.enemies) and abs(ctx.enemies[0].angle) <= pilot.AIM_TOLERANCE,
            "remembered_threat": None if ctx.threat_angle is None else int(ctx.threat_angle),
            "cover_nearby": ctx.cover_angle is not None,
            "escape_route": ctx.escape_angle is not None,
            "shooters_in_view": sum(e.type in pilot.HITSCAN for e in ctx.enemies),
            "melee_threat_close": any((e.type in pilot.MELEE and e.distance < pilot.MELEE_RANGE) or
                                      (e.type not in pilot.HITSCAN and e.distance < pilot.CLAW_RANGE)
                                      for e in ctx.enemies),
            "wanted_item": None if ctx.item is None else seen(ctx.item),
            "objective": summary["objective"],
            "keys": summary["keys"],
            "objective_distance": None if route is None else int(route.distance),
            "route_bearing": None if ctx.route_angle is None else int(ctx.route_angle),
        }

    # ---------- arena ----------

    def _play_arena(self, game, ep: int, advisor: JevAdvisor | None) -> int:
        """Arena: only turn and shoot. The Jev uses the survival question; the reflex
        aims with the pilot's hands and sweeps the room when it sees nobody."""
        nav = navigator.Navigator()
        steps, action = 0, None
        while not game.is_episode_finished() and not self.should_stop() and \
                (self.cfg.max_steps is None or steps < self.cfg.max_steps):
            nav.update(game, action)
            enemies, _ = pilot.look(game, *nav.position, nav.angle)
            state = perception.perceive(game).to_state()
            state["under_attack"] = nav.took_damage
            reflex = "shoot" if state["enemy_ahead"] else (
                "turn_left" if enemies and enemies[0].angle > 0 else "turn_right")
            action, answer, ms, asked = self._decide(advisor, state, steps, reflex)
            if advisor is None:  # the reflex uses the pilot's mouse aim
                tactic = "attack" if enemies else "search"
                reward = pilot.execute(game, pilot.command(tactic, pilot.Context(enemies=tuple(enemies), step=steps)),
                                       config.SURVIVAL_TICS, self._on_tic)
                action = tactic
            else:
                reward = 0.0
                for _ in range(config.SURVIVAL_TICS):
                    if game.is_episode_finished():
                        break
                    reward += engine.step(game, action, 1)
                    self._on_tic(game)
            self._record({
                "episode": ep, "step": steps, "mode": self.cfg.mode, "action": action, "reflex": reflex,
                "confidence": answer.confidence if answer else None,
                "probabilities": dict(answer.probabilities) if answer else None,
                "decision_ms": round(ms), "reaction_steps": steps - asked, "state": state, "reward": reward,
                "pos": [round(v) for v in nav.position], "angle": round(nav.angle),
                "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            })
            steps += 1
        return steps

    # ---------- human ----------

    def _play_human(self, game, ep: int, survival: bool) -> int:
        """A person plays from the browser: WASD moves, arrows turn, the mouse aims,
        space/click shoots, E uses. Every 8 tics it records the state as the Jev would
        see it, and what the reflex would have done, for comparison."""
        nav = navigator.Navigator()
        tics = 0
        self.cfg = SessionConfig(**{**asdict(self.cfg), "realtime": True})  # people play at 35 tics/s
        while not game.is_episode_finished() and not self.should_stop():
            keys, dx = self.human.take()
            turn = dx * -MOUSE_SCALE + HUMAN_TURN * (("ArrowLeft" in keys) - ("ArrowRight" in keys))
            cmd = pilot.Command(
                turn=turn, forward="w" in keys or "ArrowUp" in keys, back="s" in keys or "ArrowDown" in keys,
                left="a" in keys, right="d" in keys, attack=" " in keys or "Mouse" in keys, use="e" in keys,
            )
            game.make_action(pilot.buttons_vector(game, cmd, turn, cmd.attack), 1)
            self._on_tic(game)
            tics += 1
            if tics % config.DECISION_TICS == 0 and not game.is_episode_finished():
                nav.update(game)
                enemies, _ = pilot.look(game, *nav.position, nav.angle)
                ctx = pilot.Context(enemies=tuple(enemies), step=tics)
                state = perception.perceive(game).to_state()
                self._record({
                    "episode": ep, "step": tics // config.DECISION_TICS, "mode": "human",
                    "action": "human", "reflex": pilot.reflex(ctx, state["health"], nav.took_damage),
                    "confidence": None, "probabilities": None, "decision_ms": 0, "state": state,
                    "reward": 0.0, "pos": [round(v) for v in nav.position], "angle": round(nav.angle),
                    "kills": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
                })
        return tics // config.DECISION_TICS
