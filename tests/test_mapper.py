"""Tests for reading the automap. They use synthetic buffers, without opening the game."""

import numpy as np

from jevdoom.mapper import MAX_SCALE_DISAGREEMENT, Mapper

BG = (111, 87, 67)


def buffer_with(marker_xy=None, lines=()):
    image = np.full((240, 320, 3), BG, dtype=np.uint8)
    for x, y in lines:
        image[y, x] = (139, 139, 139)
    if marker_xy:
        x, y = marker_xy
        image[y - 1:y + 2, x - 1:x + 2] = (0, 0, 255)
    return image


def test_finds_the_blue_player_marker():
    assert Mapper._marker(buffer_with(marker_xy=(100, 50))) == (100.0, 50.0)


def test_no_marker_returns_none():
    assert Mapper._marker(buffer_with()) is None


def test_structure_marks_a_band_around_the_lines():
    zone = Mapper._structure(buffer_with(lines=[(160, 120)]))
    assert zone[120, 160]
    assert zone[120, 163]
    assert not zone[120, 200]


def test_robust_fit_recovers_the_scale_despite_outliers():
    pixel = np.arange(40, dtype=float)
    world = -11.3 * pixel + 2000
    world[5] += 900
    world[17] -= 700
    slope, offset = Mapper._robust_fit(world, pixel)
    assert abs(slope + 11.3) < 0.2
    assert abs(offset - 2000) < 30


def test_robust_fit_requires_well_separated_pairs():
    pixel = np.linspace(0, 2, 30)
    assert Mapper._robust_fit(-11.3 * pixel, pixel) is None


def test_calibration_rejects_disagreeing_scales():
    mapper = Mapper()
    for i in range(40):
        mapper.samples.append((-11.3 * i, 2.0 * i, float(i), float(i)))
    mapper._calibrate()
    assert not mapper.calibrated


def test_calibration_accepts_consistent_scales():
    mapper = Mapper()
    for i in range(40):
        mapper.samples.append((-11.3 * i, 11.3 * i, float(i), float(i)))
    mapper._calibrate()
    assert mapper.calibrated
    assert abs(abs(mapper.fit_x[0]) - abs(mapper.fit_y[0])) / 11.3 < MAX_SCALE_DISAGREEMENT
