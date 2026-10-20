"""Thin layer over ViZDoom: creates the game (training scenario or a full Freedoom
IWAD), enables the labels buffer that perception.py and pilot.py read, the automap
mapper.py reads, sectors and objects for level/objectives, the position variables
navigator.py uses and the degree-based turn (mouse) of the pilot, and runs actions
by name. Used by session.py and main.py."""

import json
import os
import signal
import subprocess
import sys

import vizdoom as vzd

from jevdoom import actions, config

PKG_DIR = os.path.dirname(vzd.__file__)
SCENARIOS_DIR = os.path.join(PKG_DIR, "scenarios")

TRACKED_VARIABLES = (
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.ANGLE,
)

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          ".cache", "sectors_info.json")
PROBE_TIMEOUT = 15
_PROBE = """
import vizdoom as vzd, sys
g = vzd.DoomGame(); g.load_config(sys.argv[1]); g.set_doom_game_path(sys.argv[2]); g.set_doom_map(sys.argv[3])
g.set_window_visible(False); g.set_audio_buffer_enabled(False); g.set_sectors_info_enabled(True)
g.init(); g.new_episode(); g.get_state(); g.close(); print("ok")
"""


def sectors_info_ok(scenario: str, doom_map: str | None) -> bool:
    """ViZDoom's `sectors_info` hangs the engine on some maps (Freedoom 2: MAP03,
    MAP04, MAP08, MAP09 — it is not size, MAP10 is bigger and works). Probes once in a
    separate process with a timeout and caches the answer in .cache/."""
    iwad = iwad_path(scenario)
    if iwad is None or not doom_map:
        return True
    key = f"{scenario}/{doom_map.upper()}"
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        cfg = os.path.join(SCENARIOS_DIR, f"{scenario}.cfg")
        proc = subprocess.Popen([sys.executable, "-c", _PROBE, cfg, iwad, doom_map], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, start_new_session=True, text=True)
        try:
            out, _ = proc.communicate(timeout=PROBE_TIMEOUT)
            cache[key] = "ok" in out
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # takes the hung vizdoom process down too
            proc.wait()
            cache[key] = False
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
    return cache[key]


def iwad_path(scenario: str) -> str | None:
    """The WAD holding a full scenario's levels (freedoom1/2), if it exists."""
    path = os.path.join(PKG_DIR, f"{scenario}.wad")
    return path if os.path.exists(path) else None


def make_game(
    scenario: str = "freedoom2",
    visible: bool = True,
    mode: vzd.Mode = vzd.Mode.PLAYER,
    doom_map: str | None = None,
    skill: int | None = None,
) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(SCENARIOS_DIR, f"{scenario}.cfg"))

    iwad = iwad_path(scenario)
    if iwad:
        game.set_doom_game_path(iwad)
    if doom_map:
        game.set_doom_map(doom_map)
    if skill is not None:
        game.set_doom_skill(skill)  # 1 = baby, 3 = default, 5 = nightmare

    game.set_window_visible(visible)
    # freedoom2.cfg turns on the audio buffer, which nothing here reads
    game.set_audio_buffer_enabled(False)
    game.set_mode(mode)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
    game.set_labels_buffer_enabled(True)
    # live heights of doors and lifts (level.py), where ViZDoom can handle it
    game.set_sectors_info_enabled(sectors_info_ok(scenario, doom_map))
    game.set_objects_info_enabled(True)  # keys still on the map (objectives.py)
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.WHOLE)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    for variable in TRACKED_VARIABLES:
        game.add_available_game_variable(variable)
    game.init()
    return game


def step(game: vzd.DoomGame, action_name: str, tics: int = config.DECISION_TICS) -> float:
    pressed = actions.ACTIONS[action_name]
    vector = [int(button in pressed) for button in game.get_available_buttons()]
    return game.make_action(vector, tics)
