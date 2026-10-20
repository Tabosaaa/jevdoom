# jevDoom

**Jev** — [TypeSafe](https://typesafe.ai)'s System One model — playing Doom, and finishing levels.

The code owns the game loop; the Jev decides. Doom runs for real through
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom). Every step the code reads the game
state (health, ammo, enemies with distance and angle, the level's current objective),
describes it as JSON, asks the Jev "what is the best tactic now?", and the "hands" carry it
out: mouse-like aim, running, strafing, doors, switches, lifts. Same recipe as the Jev
Doom demo: structured text state, several decisions per second, code + AI.

The full lab notebook — every phase, what was measured and what was learned — is in
[PLAN.md](PLAN.md).

## Quick start

Needs **Python 3.10+**. A TypeSafe key (https://console.typesafe.ai) is only needed for
the Jev mode; autopilot and "You" run without it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # paste your TYPESAFE_API_KEY into .env

PYTHONPATH=src python -m jevdoom.server     # opens http://127.0.0.1:8765
```

## The UI: game + Jev + controls on one page

- **Who plays** — *Jev* (the model picks each tactic), *Autopilot* (the hand-written
  reflex, no API) or *You* (keyboard and mouse).
- **Settings** — scenario (Freedoom 1/2 or the survival arenas), map, difficulty 1–5
  (*I'm too young to die* … *Nightmare!*), pace (tics per decision → decisions per
  second), game speed and number of episodes.
- **Live game** — the screen with brackets on what the Jev detects (red = enemy,
  green = item), the HUD, and a ribbon with the tactic of every decision.
- **What the Jev decides** — the current tactic, confidence, API latency, reaction time,
  whether the hand-written reflex agrees, and the probability of every alternative.
- **What the Jev reads** — a radar with enemies by angle and distance, the route heading,
  the remembered threat and the wanted item, plus the full JSON sent to the model.
- **Floor plan** — the map read from the WAD with the player's trail, the next route
  point, the objectives (exit, keys, switches, weapons, health) and the current target;
  level progress and collected keys.
- **You play** — in "You" mode click the screen: WASD moves, arrows or mouse turn,
  space/click shoots, E uses, Esc releases the mouse. The panel shows what the reflex
  would do in your place.
- **Replay** — reopen any recorded `runs/*.jsonl` step by step.

The server listens on 127.0.0.1 only and uses nothing but the standard library
(Server-Sent Events). Options: `--port N`, `--no-browser`.

## How it works

```
ViZDoom ─▶ pilot.look (screen) + objectives/level (floor plan) ─JSON─▶ brain (Jev) ─tactic─▶ pilot (hands) ─▶ ViZDoom
                                                               └─ hand-written reflex when confidence < 0.5
```

In the UI the Jev runs on its own thread (`JevAdvisor`): the game never waits for the
API, it keeps playing the last tactic decided, and each answer lands ~0.45 s after the
state it was based on — about a human reaction time.

| module | role |
| --- | --- |
| `wad.py` | reads a map from the WAD: lines, sectors, things and what every special line does |
| `level.py` | the floor plan: walkability grid with the player's radius, live heights (doors, lifts), key doors, teleporters, line of sight and cover; distance fields (reverse Dijkstra) to any set of targets |
| `objectives.py` | what to do now: a nearby weapon, health when hurt, the exit if reachable, otherwise the key / switch / walkover line that unlocks it; explore when nothing is left |
| `pilot.py` | the hands: aim in degrees (mouse), tap-fire, strafing while shooting, cover, retreat toward open ground, searching for an unseen shooter; and the hand-written reflex |
| `brain.py` | the questions to the Jev: tactic (levels), survival (arenas), and the two legacy goal/action questions for levels without an exit |
| `session.py` | one game in any mode, emitting events for the CLI and the UI |
| `server.py` + `ui/index.html` | the UI |
| `perception.py`, `navigator.py`, `actions.py`, `engine.py` | screen into facts, memory across steps, buttons, and the thin layer over ViZDoom |
| `mapper.py` | automap experiment, measured and kept out of the loop (Phase 8 in PLAN.md) |

**Is the autopilot machine learning?** No. It is a set of hand-written rules
(`pilot.reflex`) on top of classic algorithms (Dijkstra distance fields, line-of-sight
tests), with thresholds calibrated by measured A/B runs. The Jev is the only model in the loop.

## Command line

```bash
# the Jev plays a real Doom level (Freedoom MAP01) to the exit
PYTHONPATH=src python -m jevdoom.main --scenario freedoom2 --map map01

# the same with the hand-written reflex only (free, no API); --realtime plays at a human pace
PYTHONPATH=src python -m jevdoom.main --scenario freedoom2 --map map02 --autopilot

# the Jev defends the arena against waves of monsters (default scenario)
PYTHONPATH=src python -m jevdoom.main --episodes 3

# you play in the Doom window; or watch what the Jev sees, without the API
PYTHONPATH=src python -m jevdoom.main --human
PYTHONPATH=src python -m jevdoom.main --perceive --no-window
```

| option | meaning |
| --- | --- |
| `--scenario NAME` | `freedoom2`, `freedoom1`, `defend_the_center` (default), `defend_the_line` |
| `--map MAPxx` | level inside the scenario's WAD (default MAP01) |
| `--episodes N` | how many games to play |
| `--max-steps N` | stop an episode after N decisions |
| `--tics N` | tics per decision (35 tics = 1 s) |
| `--no-window` | run headless |
| `--autopilot` | hand-written reflex only, no API |
| `--realtime` | play at a person's pace (the Jev answers asynchronously) |
| `--human` | you play in the Doom window |
| `--perceive` | print the state the Jev would read, no API |

Every game writes `runs/*.jsonl` (state, tactic, confidence, probabilities, position,
objective) — that is what the UI's replay reopens. `viewer.html` is a small standalone
viewer for logs of the legacy loop (levels without an exit).

## Results

| map | who decides | difficulty | finished |
| --- | --- | --- | --- |
| MAP01 | reflex (`--autopilot`) | 3 | **8 of 8** (12 of 12 in the previous batch) |
| MAP01 | Jev | 3 | finished (also from the UI), ~0.3 s per decision |
| MAP02 | reflex | 1 | **3 of 4** — the failures are deaths, not getting stuck |
| MAP02 | Jev | 1 | **2 of 2** — 794 and 834 steps, 41–43 kills |
| MAP02 | reflex | 2 | 2 of 6 |
| MAP02 | reflex | 3 | ~3 in 26 — the opening (3 ShotgunGuys) kills early in half of the runs |
| MAP03 | reflex | 1 | grabs the blue and yellow keys; does not finish yet |
| arena | reflex with mouse aim | — | 26 kills from 26 bullets, 3 of 3 |

MAP02 asks for what MAP01 does not: grab the weapon, press the switch that opens a remote
door, call and ride the lift, collect the three keys and open the locked doors. All of it
comes from the WAD and the live sector heights (Phases 12 and 13 in PLAN.md). On some maps
(MAP03, 04, 08, 09) ViZDoom's `sectors_info` hangs the engine: the project detects that
once, opens the map without it and assumes doors open after triggering the switch with
the same tag.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

The tests read the real Freedoom maps shipped with ViZDoom; none of them needs the API.
