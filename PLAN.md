# jevDoom — the Jev plays Doom

Project plan and lab notebook. It started as architecture only ("no code yet"); each
phase below records what was built, what was measured and what was learned.

## The idea in one sentence

The **Jev** (TypeSafe's System One model, the "Jeff" in the name) is the *brain* that
decides the next action; the **code** owns the game loop. Doom runs for real in a window
on your Mac via **ViZDoom**, and at every decision the code describes the game state in
JSON, asks the Jev "what now?" and executes the chosen action.

## The constraint that shapes everything

The Jev **does not generate text and does not see pixels**. It takes *state* (JSON/text)
and returns a *typed judgement*: a `Choice` (one option from a set), a `Noul` (yes/no
probability) or a `Score`. Direct consequences:

1. **No screenshots for the model.** We need *structured* game state. Hence ViZDoom,
   not a regular Doom port + screen capture + OCR.
2. **The Jev picks an action, it does not describe the scene.** The central question is
   a `Choice` over a fixed set of actions (move, turn, shoot, etc.).
3. **Code does rules and geometry; the Jev does "common sense".** Enemy distance and
   angle, "wall ahead", "critical health" → computed by code (deterministic). *What to
   do* about it → judged by the Jev.

## Why ViZDoom

- Runs on macOS (Apple Silicon, via Homebrew + pip) and opens a real Doom window you can watch.
- Its Python API exposes structured state per tic: `GameVariables` (HEALTH, AMMO, etc.)
  and the **labels buffer** (visible objects with position/label) — exactly the "state"
  the Jev needs, for free.
- Accepts programmatic actions (button vector) and controls the pace (`set_ticrate`,
  frame skip). Built for AI playing Doom.
- Ships ready-made scenarios (`basic`, `defend_the_center`, `deadly_corridor`) to start
  simple before the full game.

## Loop architecture (the code is the owner)

```
        ┌──────────────────────────── loop per decision (~2–5 Hz) ───────────────────────────┐
        │                                                                                     │
  ViZDoom ──raw state──▶ perception.py ──state JSON──▶ brain.py (Jev: Choice) ──action──▶ actions.py ──buttons──▶ ViZDoom
  (window)  (vars+labels)  (semantic facts)            (+ confidence)     (gating)       (vector)     (runs N tics)
```

1. **Read** the ViZDoom `GameState`: variables + labels of the visible objects.
2. **Perceive** (`perception.py`): turn the raw data into compact facts — e.g.
   `{"health": 60, "ammo": 12, "weapon": "pistol", "enemies": [{"direction": "ahead", "distance": "near"}], "wall_ahead": false}`.
   This is the most important engineering piece: the quality of the Jev's decision
   depends on the quality of this summary.
3. **Decide** (`brain.py`): a `Choice` — `instructions: "The player is playing Doom.
   Given the state, what is the best action now?"`, `criteria` = the action set. A few
   safety `Noul`s ("in immediate danger?") can run in parallel if useful.
4. **Confidence gate**: >0.9 execute; 0.5–0.9 execute with caution (e.g. a conservative
   action); <0.5 fall back to a safe default (sweep/hold) instead of gambling.
5. **Act** (`actions.py`): map the chosen action to a button vector and run it for N tics
   (the decision *throttle*).

## Known ceiling (deliberate cut)

Doom runs at 35 tics/s. **The Jev cannot be called every frame** — network latency plus
per-request cost rule it out. The loop decides at ~2–5 Hz and *holds the action* during
the tics in between (frame skip). The Jev is fast for a model, but it is still one API
call per decision. Frame-level reaction would need a different kind of model (local RL),
out of scope.
`ponytail: fixed decision rate; make it adaptive only if gameplay demands it.`

## Folder layout (original plan)

```
jevDoom/
├── README.md                 # how to install and run
├── PLAN.md                   # this file
├── requirements.txt          # deps: vizdoom, typesafe-sdk, python-dotenv
├── .env.example              # TYPESAFE_API_KEY=  (the key stays local, never committed)
├── scenarios/                # .cfg + .wad (start with the ones ViZDoom ships)
├── src/jevdoom/
│   ├── __init__.py
│   ├── main.py               # entry point: builds the engine and runs the loop
│   ├── engine.py             # ViZDoom setup/teardown, read_state(), step(action, tics)
│   ├── perception.py         # raw GameState -> dict of semantic facts
│   ├── brain.py              # Jev wrapper: (state, action_set) -> (action, confidence)
│   ├── actions.py            # action set + mapping to ViZDoom buttons
│   └── config.py             # decision rate, thresholds, scenario path
└── tests/
    └── test_perception.py    # given a known GameState, the facts come out right
```

Size rule: small focused files; each module has one responsibility.

## Execution phases (when coding starts)

1. **Doom on screen.** Install ViZDoom, open the `basic` scenario, a **random** agent
   just to confirm the window and inputs work on the Mac.
2. **Perception.** Log the semantic state at each decision and check it matches the screen.
3. **Brain.** Plug in the Jev (`typesafe-sdk`), a `Choice` over the action set, log
   decision + probabilities. No gating yet.
4. **Integration.** Loop + throttle + confidence gate together. The Jev plays the simple scenario.
5. **Raise the difficulty.** `defend_the_center` → `deadly_corridor` → a full map. Tune
   the perception summary, the actions and the thresholds based on what works and what fails.

Every phase leaves **one runnable check** (e.g. `test_perception.py`, a decision log)
before moving on.

### Phase 4 — Navigation (goal: reach the end of the corridor)

`defend_the_center` is a survival arena → you can only spin in circles. Focus moved to
**`deadly_corridor`**, which has a start and an end. Changes:
- Navigation actions: `move_forward`, `move_back`, `turn_left`, `turn_right`, `shoot`.
  Mapped by *name* to the ViZDoom button (the engine builds the right vector for the scenario).
- Perception now includes the **objective** (the green armor at the end: direction + distance).
- The Jev's question rewritten for navigation + combat.
- With low confidence the default becomes `move_forward` (advance instead of stalling).
- The loop writes a **decision log** to `runs/*.jsonl` (state, action, confidence,
  probabilities) — it feeds the Phase 5 UI without spending Jev calls again.

Honest limit: without map memory, doors/keys and real mazes are out of reach. The goal
is to cross the corridor to the objective, not to beat the original Doom.

### Phase 5 — Editorial decision-tree UI

A clean, minimal panel that replays the log: the Jev's decision at every step (chosen
action, probability bars, confidence) next to the game state. Built against the recorded
`runs/*.jsonl` — zero API cost to iterate on the visuals. (Today: `viewer.html`.)

## Dependencies and environment

- macOS Apple Silicon + Homebrew.
- **Python 3.11+ recommended** — the machine had 3.9; a venv with a newer version avoids
  trouble with wheels.
- ViZDoom needs a few build libraries from Homebrew (cmake, sdl2, etc.) — sorted out in Phase 1.
- TypeSafe key in `.env` (`console.typesafe.ai`), read via `python-dotenv`.

## Open questions (decided before Phase 3)

1. **Action set**: minimum viable (forward, turn left, turn right, shoot) or include
   strafe/back? Start small.
2. **A single `Choice`, or `Choice` + safety `Noul`s in parallel?** Start with one
   `Choice` and add a `Noul` ("immediate danger?") if it struggles.
3. **First scenario**: `defend_the_center` (aim + survival, no complex navigation) as the
   first "real" target after `basic`.

### Phase 6 — A real Doom level (Freedoom MAP01)

ViZDoom already ships the full Freedoom IWADs, so the target became a **real level**,
with doors, items, enemies and an exit (`map_exit_reward = 1`). What changed:

- **`use`** action (USE button): in Doom, doors and switches look like walls.
- New **`navigator.py`** — memory across steps, all computed in code:
  - visited cells and discovered walls;
  - **frontier compass**: points toward the nearest unknown area;
  - **target patience**: if it stops getting closer, mark it unreachable and switch;
  - blocking detection tied to the *movement attempt* and to **damage taken**.
- Perception now separates **enemies** from **items** (including keys and health kits).

Three real deadlocks were found and fixed by measurement, not guesswork:
1. turning counted as "stuck" → infinite `use`;
2. turning did not clear the block → infinite spinning;
3. the first step of the level barely moves the player → false block (fixed with a warm-up).

**Measured result (MAP01):** survives 1500 steps with health going up (picks up kits),
correct combat (shoots in 100% of steps with an enemy ahead) and explores ~45 areas.
**It still does not finish the level:** exploration saturates because reactive decisions
with grid memory do not solve a level's *topology* (finding the right switch, the order
of doors). That is the honest ceiling of the current architecture.

Next levers, if we insist: use ViZDoom's `automap_buffer` as a real map, remember doors
already opened, or a second, higher-level question to the Jev ("which goal now?") that
only runs every N steps.

### Phase 7 — Goal hierarchy (measured experiment)

Hypothesis: deciding from scratch every step prevents committing to a plan; a goal that
lasts several steps should unlock the level. Implemented as two questions to the Jev —
`GOAL_QUESTION` every 10 steps and `ACTION_QUESTION` every step, with the goal inside the
state. Cost: ~10% more calls.

Three versions, all measured on MAP01:

| version | survived | areas | what it taught |
| --- | --- | --- | --- |
| flat (Phase 6) | 1500 steps | 45 | baseline |
| hierarchy with combat as a goal | 125 and 137 steps | 20 / 23 | **worse**: committing to "fight" for 10 steps is fatal |
| navigation-only goal | 800 steps | 42 | tied with the baseline; combat went back to being a reflex |
| + stagnation signal | 215 steps | 25 | it did switch goals, but with no gain |

**Honest conclusions:**

1. **Time scales do not mix.** Combat and health must be a reflex every step; only the
   level *heading* tolerates commitment. Mixing the two killed the agent.
2. **A goal is only useful if the signal that triggers it exists.** `open_path` was dead
   because nothing in the state said "exploration stalled" — the code had to compute
   `steps_without_discovery`.
3. **The hierarchy did not break the ceiling.** Every version saturates at ~42–47 areas.
   The bottleneck is not the decision structure, it is the **spatial representation**: a
   96-unit grid with walls discovered by collision does not represent the topology of a
   Doom level (lifts, distant switches, key doors in unreachable areas).

What the hierarchy did deliver: the UI became a **two-level decision tree**
(heading → action), which was the visual goal of the project.

The real next lever, if it ever pays off: ViZDoom's `automap_buffer`, which gives the
level's real map instead of a grid inferred from collisions.

### Phase 8 — Automap (measured and reverted)

Hypothesis: the bottleneck is the spatial representation, so reading ViZDoom's
`automap_buffer` (`WHOLE` mode, which draws the whole floor plan) would give the real
topology and unlock the level.

**What worked very well:**
- The automap delivers the full floor plan at step 0, including rooms never visited.
- The pixel↔world calibration solves itself. The player marker is a rotating arrow and its
  centre carries a bias that follows the movement direction: least squares absorbs that
  bias and is off by up to 50%, while **Theil-Sen** (median of slopes between well-separated
  pairs) finds the true scale (~11.3 units/px) in 6 of 6 seeds. Requiring both axis scales
  to agree rejects bad fits for free.
- Cost: 5–8 ms per step, negligible next to ~360 ms of API latency.

**What sank the idea:**
The automap is a **line drawing, not a filled map**, and colour does not separate a
walkable boundary from a solid wall — the starting room is sealed by a mix of colours that
also appear as walls elsewhere. Measured consequence:
- blocking every line, the flood fill gets stuck at 644 px (one room);
- blocking only the bright ones, it leaks into the void (86–98% of the screen);
- dilating to close gaps does not help.

Without reachability, targets taken from the map aim at rooms behind walls:

| version | steps | areas | behaviour |
| --- | --- | --- | --- |
| baseline (navigator only) | 1500 | **45** | reference |
| automap without target patience | 900 | 21 | spins in place (847 turns / 30 moves) |
| automap with target patience | 900 | 26 | walks, but explores less than the baseline |

**Decision:** `mapper.py` was taken out of the decision loop and the baseline restored.
The module and its tests stay in the repository as a documented experiment — the
Theil-Sen calibration is solid and reusable if a walkability source ever shows up.

**Lesson:** the shape of the data matters more than the amount of code. The "~100 lines"
estimate was right; the mistake was assuming the automap encodes walkability. A floor plan
that cannot tell a door from a wall does not solve navigation, however cheap it is to read.

### Phase 9 — Survival arena (the right fit)

Everything that defeated Phases 6–8 was *navigation and topology*, exactly the weak spot
of a reactive model. `defend_the_center` flips that: the player is stuck in the centre,
can only turn and shoot, and monsters spawn in 360° and advance. Pure reflex — the Jev's
strength.

**The discovery that decided everything was in perception, not in the model.** The
"ahead" band covered 1/3 of the screen (~30°), so the agent shot with the enemy 15° off
axis, missed and emptied its ammo. Narrowing the band to 10% of the screen, the *same*
test heuristic jumped from 2.2 to 14 kills:

| "ahead" window | kills (heuristic) |
| --- | --- |
| 33% of the screen | 2.2 |
| 16% | 12.8 |
| 10% | 14.0 |

Now "ahead" means "can be hit", not "more or less in front of me".

**Results (defend_the_center, 4 tics per decision):**

| agent | kills (mean) |
| --- | --- |
| random | 1.5 |
| hand-written heuristic | 14.0 |
| **Jev** | **15.7** (18 / 17 / 12) |

Jev and heuristic are tied within variance — what impresses is not beating the rule, it
is **getting there from a plain-language description**, with no coded rule. And ammo
discipline was perfect: **zero wasted shots** in 3 episodes (it never fired without a
lined-up target), firing at 80% of the opportunities and using the rest to turn toward a
closer threat.

Other changes: `MarineChainsawVzd` (the scenario's custom monster) was not in the enemy
list and was invisible to perception; the loop now detects an arena by the absence of the
move button and switches to a single reactive question, with no goal hierarchy.

### Phase 10 — Liveness: the character no longer gets stuck

Report: in the level, the character kept bumping into a wall. A diagnosis with the free
heuristic confirmed the freeze (the last positions were identical) — and revealed that
`blocked_attempts` **never went above 1**. A side effect of the Phase 6 fix ("turning
clears the block"): the cycle `move(fail) → blocked → use → turn → reset` ran forever
without accumulating. The honest signal was missing: *how long have I not moved*,
regardless of the action chosen.

Three changes:
1. **`stationary_for`** in navigator.py — steps without real displacement, whatever the action.
2. **Strafes** (`strafe_left` / `strafe_right`): `freedoom2` always had
   `MOVE_LEFT`/`MOVE_RIGHT` and the action set never used them. The quick move that was missing.
3. **Deterministic unsticking** in main.py: past the threshold, the *code* takes over and
   runs a manoeuvre. Liveness is a guarantee, not a judgement — it does not depend on the model.

Manoeuvre and threshold were calibrated with the free heuristic, 4–5 runs per configuration:

| manoeuvre / threshold | areas (mean) | longest stationary | stuck at the end |
| --- | --- | --- | --- |
| no unsticking | 57.5 | **36 steps** | 0/4 (but freezes midway) |
| turn first, threshold 10 | 38.8 | 14 | 0/4 |
| **strafe first, threshold 14** | **52.0** | 18 | **0/5** |

Strafing first frees the character without throwing away the direction it was heading;
turning first destroys the alignment and costs 13 areas of exploration.

**Healing:** the agent saw Stimpacks and Medikits and ignored them. The cause was not the
prompt, it was a **missing capability** — the `recover_health` goal had been removed in
Phase 7. It came back as a *navigation* goal (go to the item), with the `low_health` and
`health_item` facts computed in code.

**Status:** in the level it explores, shoots with perfect aim (10/10 and 6/6 in the runs),
looks for health when hurt, strafes and **does not get stuck** (0 unstickings triggered in
a real run, worst case 13 steps stationary). It still dies on MAP01 — a pistol against
hitscan enemies shooting from outside the field of view is a hard fight, and that remains open.

### Phase 11 — Finishing the level: WAD floor plan, human hands and the Jev on tactics

Inspiration: the demo video of the Jev playing Doom ("Jev AI playing Doom", TechSpot).
According to TypeSafe's announcement it decides ~10 times per second over **structured
text state**, and the message is "code + AI": the model follows instructions, the code
takes care of the rest. Phases 6–8 showed the bottleneck was the spatial representation,
so the code got what it lacked: a real map.

**level.py — the real floor plan.** The WAD already has everything: lines, sectors with
floor and ceiling heights, doors and the exit switch (special 11). It becomes an 8-unit grid with:
- one-sided walls and impassable lines dilated by the player's radius;
- solid decorations (trees, barrels, columns) as obstacles;
- effective floor = the highest under the 32-unit body. Without it the 8-unit trench of
  sector 127 (floor -480) looked like a pit and the exit was unreachable; with 16-unit
  cells, a 32-unit gutter looked climbable and it got stuck in it;
- manual doors treated as open (the loop presses "use" on arrival);
- target = only the cells IN FRONT of the switch. The exit box has two buttons, one on each
  face; from the north "use" triggered the other switch and it stood there.

The route is a **distance field** (reverse Dijkstra from the exit), computed once in ~1 s.
Following the route is walking down the gradient, 0.3 ms per step, from anywhere — no
replanning when combat pushes it around. The first version with A* replanned at every
detour (0.5 s each). If it gets stuck without progress along the route, the plan learns:
it blocks the stretch ahead and recomputes; if that isolates the exit, it forgets what it learned.

**pilot.py — the hands.** ViZDoom accepts `TURN_LEFT_RIGHT_DELTA`: turning in degrees per
tic, like a mouse. Aim comes from the real position of the monsters visible on screen
(labels), and corpses are discarded by their on-screen height relative to distance. It
turns while running, strafes without losing aim and fires with **tap-fire**: it only pulls
the trigger with the weapon ready (`ATTACK_READY`) and releases between shots, because in
Doom only the first shot of each pistol click comes out without spread. (A fixed 1-tic tap
lost shots: the weapon was still mid-cycle and the click was dropped — 3/10.)

**The Jev picks the tactic** (`TACTIC_QUESTION`): follow_route, attack, strafe_*, retreat,
grab_item, use — over health, ammo, enemies with distance and angle in degrees, wanted
item, distance to the exit and whether a door is nearby. Below the confidence gate the
hand-written reflex rules, which is also the `--autopilot` mode (free).

**Results on MAP01 (reflex, no API):**

| version | finished the level |
| --- | --- |
| before (Phases 6–10) | 0 — explored ~45 areas and died |
| floor plan + route + hands | 7/10 |
| + tap-fire and target in front of the switch | 23/30 |
| + detour for a new weapon | 21/30 (noise; the shotgun was already picked up in 18/20) |
| **+ retreat when hurt with an enemy close** | **28/33** |

Each level takes ~160 steps (~40 s of game time, ~1.5 s of CPU). The remaining failures are
all combat deaths: 60% of the damage comes from ShotgunGuys in a single room (~1792,-256),
with the player trading shots standing still. No navigation failure in the last 60 runs.

**Honesty about "human":** the floor plan is the map of someone who has already played the
level; aiming only uses what is on screen. What a human still does and this does not: use
a door as cover (open, step aside, wait for the enemy to come), and know where an unseen
shot comes from.

### Phase 12 — Other levels, human combat and the integrated UI

**The Jev on the whole level.** Two full MAP01 runs with the Jev deciding every tactic: one
finished the level (135 steps, 10 kills), the other died. The first attempt exposed a
prompt loop: the criterion "use when stationary near a door" fed itself (use does not move,
so `stationary_for` grows). `use` was removed from the Jev's options — the hands already
open doors. After that the Jev also finished MAP01 from the UI, at ~0.3 s per decision.

**Progression (MAP02+): `wad.py` + `objectives.py`.** The floor plan became generic:
- distance fields to any set of targets, with **priority** (per-target starting cost in
  the Dijkstra) and the "owner" of each cell — the planner knows where the route leads;
- **live sector heights** (`sectors_info`): a remote door that opened, a lift that came
  down, stairs that rose change walkability and the fields are rebuilt;
- **key doors** closed until it has the key (keys disappear from `objects_info`);
- **lifts** as "any height" (including their surroundings, by the player's radius) and
  **teleporters** as extra graph edges;
- objectives: exit, keys, switches (the right S1/SR, from the front), walkover lines that
  trigger something, new weapons, health when hurt, armor; exploration as a last resort.

Model errors that only showed up while playing, each with its measured cause:

| symptom | cause | fix |
| --- | --- | --- |
| switch "done" with no effect | counted 4 steps nearby | done when the sector with the same tag moves |
| explores forever | the exploration target was recreated and never gave up | an unreachable block counts as seen |
| lift loop | getting stuck waiting for the lift blocked the path past it | patience ×4 and no learned block near doors/lifts |
| loop between repeatables | learned blocks isolated the key | forget what was learned before reopening repeatables |
| red switch never again | giving up was permanent for S1 | temporary give-up |
| door opens and closes | "use" every 2 steps closes the DR door that is opening | use only when stationary in front of it, with an interval |
| stuck in a 32×32 pit | floor dilation by the full radius "covered" the pit | effective floor with an 8-unit reach |
| "no reachable objective" | wall clearance 18 > radius 16 erased narrow strips | clearance = radius; reachable-cell search up to 96 units |
| dies at the foot of the lift | shooters on the upper edge are out of frame | `search` turns and backs off |
| reaches rooms without health | crossed the map for an armor | detour radius per item type (weapon 700, armor 500, health 900) |

**Human combat.** Measured by damage taken per (enemy, distance, tactic):
- always shoot while moving sideways (circle-strafe): an Imp fireball is a projectile;
- do not duel beyond 600 units if nobody is shooting at it; new weapon first;
- **bug**: it barely shot while retreating — the trigger required the target to be lined up
  *before* the turn; now it is enough for the step's turn to reach it;
- Demon up close: retreat toward open ground (`level.escape`), not into the wall;
- several hitscan shooters: fight from the corner (`level.cover`, vectorised line of sight
  against the walls), but point blank (< 256) it shoots — turning your back means taking the whole blast;
- got shot without seeing anyone: turn to where an enemy was last seen (`search`).

Arena with the new hands: **26 kills in 3 of 3 episodes** — all 26 bullets of the
scenario, no misses (before: heuristic 14.0, Jev 15.7).

**Integrated UI (`server.py` + `ui/index.html`).** One page: live game with what the Jev
detects, the decision with probabilities and disagreement with the reflex, radar, floor
plan with route and objectives, mode/scenario/map/difficulty/pace/speed controls, a "You"
mode with keyboard and mouse, and replay of any `runs/*.jsonl`. Standard library,
Server-Sent Events, 127.0.0.1 only. Verified with real Chrome via the DevTools Protocol.

**Environment:** the process is suspended by the OS for 500–1000 s from time to time
(measured even inside pure numpy); it is neither the game nor the code. Long background
benchmarks must account for it.

### Phase 13 — More robustness: MAP02 on higher difficulties, the Jev on MAP02, MAP03

**The Jev on MAP02.** With the old criteria the Jev chose "attack" where the reflex
(calibrated by measurement) would retreat or take cover — 42 disagreements against Demons
and Imps up close in a 185-step game. The question now describes the measured tactics and
the state gained the facts that trigger them, computed in code (`shooters_in_view`,
`melee_threat_close`, `escape_route`). Result: **the Jev finished MAP02 in both runs**
(difficulty 1; 794 and 834 steps, 41–43 kills). Following instructions is the model's
strength — but it only helps if the instructions say what is already known to work.

**Combat, by measurement (difficulties 2 and 3):**
- target priority: a hitscan shooter counts as if it were at half the distance
  (a ShotgunGuy has 30 health and hurts from afar);
- Imp up close: retreat while shooting (its claw is strong);
- a weapon on the floor is not worth running into a room of shooters;
- outgunned (2+ shooters): cover within a 320 radius (the door it came through), or
  retreat while shooting if there is no corner;
- ammo becomes a wanted item when it runs out;
- **Automatic weapon switching was tested and removed**: A/B of 6+6 episodes on
  difficulty 3 — with switching, median 115 steps and 8.5 kills; without, 507 and 20.8.

MAP02 on difficulty 3: from 0 to ~1 in 4–8 episodes per batch (the opening, with three
ShotgunGuys, still kills early in half of the runs); on 2, 2 in 6; on 1, 3 in 4 (the
failures are deaths). MAP01: 8 of 8 with the new aim.

**MAP03** (no live heights): aiming at the switch from the front, it grabs the blue and
yellow keys; dies in combat around step 500.

**Model bugs found while playing:**

| symptom | cause | fix |
| --- | --- | --- |
| Jev log mixed with another game | filename with seconds only | milliseconds + exclusive open |
| jumps from the pit onto the platform | height wildcard on the whole lift surroundings | wildcard only entering/leaving the lift |
| waits under a raised lift | — | route points to the lift and calls it (use) when it is up |
| pushes a closed door forever | start-cell search "crossed" doors | only cells in line of sight |
| route starts on the upper level while in the pit | an edge cell already belongs to the upper level | start only on floor reachable from the real floor |
| explore finds nothing | the target was the block centre, sometimes inside a wall | any free cell of the block; if none, explore again |
| MAP03/04/08/09 do not open | ViZDoom's `sectors_info` hangs the engine on those maps | probe in a subprocess with a timeout, cached in `.cache/` |
| stuck in a channel, remote door never opens | switch "done" by insistence without being pressed | press use on the target switch on arrival; done without effect counts again |
| leaning on the switch without triggering it | aimed at the middle of the line from the side; the use range (64) does not reach | aim at the nearest point of the line, facing the wall |
| runs into the shooters' room after the shotgun | the item was a target even with shooters around | an item with a shooter < 400 away waits |
| pushes the teleporter without crossing | crossed the line 12 units from its end, where the wall touches | cross away from the ends, perpendicular |

**Maps without live heights** (where `sectors_info` hangs): the floor plan uses the WAD
heights; when a switch/walkover line is triggered, it assumes the doors with the same tag
are open, and undoes the assumption if it gets stuck in front of them.

**Performance:** instead of one distance field per planner stage (5 per recompute, 0.14 s
each on MAP03), a forward search says what is reachable and a single field is computed.
MAP03: 150 steps in 5 s (before: 36 s).

### Phase 14 — The game does not stop for the Jev to think

In the UI, Jev mode looked choppy. Measured in the browser: 7.3 frames/s and the screen
frozen 68% of the time, against 16.6 frames/s and 10% on autopilot. Cause: the API call
(~0.3 s) was synchronous — the game froze at every decision. Now (`JevAdvisor` in
session.py) a thread asks non-stop, always with the latest state, and the game keeps going
with the last tactic decided; the answer takes effect ~0.45 s after what it saw — a human
reaction time, shown in the UI ("reaction"). Result: 16.4 frames/s, the same as autopilot,
~4 decisions per second. In the CLI without `--realtime` (benchmarks) the call stays
synchronous and the Jev decides every step.

### Phase 15 — Everything in English

Code identifiers, strings, comments, docstrings, the Jev's prompts, the UI, the legacy
viewer and the docs were translated to English. The state keys the Jev reads changed
accordingly (`health`, `enemies`, `objective`, `follow_route`, …), so logs recorded before
this phase use the old keys and replay only partially in the UI.
