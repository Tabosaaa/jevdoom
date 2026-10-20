"""Tests for the session and the UI server, without opening the game or the API."""

import base64
import struct
import zlib

import numpy as np
import pytest

from jevdoom import server, session


def test_encoded_png_has_the_dimensions_and_pixels():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[1, 2] = (255, 10, 20)
    png = base64.b64decode(session.encode_png(rgb))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    idat = png[png.index(b"IDAT") + 4: png.index(b"IEND") - 8]
    rows = zlib.decompress(idat)
    assert (w, h) == (3, 2) and rows[1 + 3 * 3 + 1 + 2 * 3: 1 + 3 * 3 + 1 + 3 * 3] == bytes((255, 10, 20))


def test_config_validates_mode_map_and_ranges():
    session.SessionConfig(scenario="freedoom2", doom_map="MAP05").validated()
    for bad in ({"mode": "cheat"}, {"doom_map": "MAP99"}, {"skill": 9}, {"tics": 0}, {"scenario": "../x"}):
        with pytest.raises(ValueError):
            session.SessionConfig(**{"scenario": "freedoom2", "doom_map": "MAP01", **bad}).validated()


def test_server_only_accepts_known_fields_with_the_right_type():
    cfg = server.config_from({"mode": "autopilot", "doom_map": "MAP02", "skill": "4", "evil": "x"})
    assert cfg.mode == "autopilot" and cfg.skill == 4 and cfg.frames
    with pytest.raises(ValueError):
        server.config_from({"tics": "lots"})


def test_options_list_scenarios_and_maps():
    opts = server.options()
    freedoom2 = next(s for s in opts["scenarios"] if s["id"] == "freedoom2")
    assert "MAP01" in freedoom2["maps"] and set(opts["modes"]) == {"jev", "autopilot", "human"}


def test_hub_never_drops_a_decision_when_a_client_lags():
    hub = server.Hub()
    q = hub.subscribe()
    for i in range(server.QUEUE_SIZE + 10):
        hub.publish("frame", {"i": i})
    hub.publish("step", {"step": 1})
    kinds = [q.get_nowait()[0] for _ in range(q.qsize())]
    assert kinds[-1] == "step"


def test_human_input_accumulates_mouse_and_hands_it_over_once():
    human = session.HumanInput()
    human.set(["w"], 10)
    human.set(["w", "a"], 5)
    assert human.take() == (frozenset({"w", "a"}), 15.0)
    assert human.take() == (frozenset({"w", "a"}), 0.0)


def test_log_name_is_unique_even_within_the_same_second(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "RUNS_DIR", str(tmp_path))
    first = session.new_log_path()
    second = session.new_log_path()
    assert first != second and server.RUN_NAME.match(second.rsplit("/", 1)[-1])
