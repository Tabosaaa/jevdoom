"""Tunable constants: decision pace, perception thresholds, exploration memory and
the confidence gate. Imported by engine, perception, navigator, brain, session and main."""

DECISION_TICS = 8
SURVIVAL_TICS = 4

CENTER_FRAC = 0.10
NEAR_H_FRAC = 0.50
MID_H_FRAC = 0.22
MIN_W = 3
MIN_H = 6

CELL_SIZE = 96
LOOK_AHEAD = 192
STUCK_DIST = 12
WARMUP_STEPS = 2
TARGET_PATIENCE = 12
ESCAPE_AFTER = 14
GOAL_INTERVAL = 10
ROUTE_STUCK = 8
PROGRESS_MIN = 16

MAP_WALL_TOLERANCE = 30
MAP_STRUCTURE_BAND = 3
MAP_CALIBRATION_STEPS = 70
MAP_MAX_CANDIDATES = 3000

LOW_HEALTH = 50

MIN_CONFIDENCE = 0.5
DEFAULT_ACTION = "move_forward"
SURVIVAL_DEFAULT_ACTION = "turn_right"
