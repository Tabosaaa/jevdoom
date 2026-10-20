"""Player actions mapped to ViZDoom buttons. brain.py picks one by name and
engine.step builds the right button vector for the current scenario."""

from vizdoom import Button

ACTIONS = {
    "move_forward": (Button.MOVE_FORWARD, Button.SPEED),
    "move_back": (Button.MOVE_BACKWARD,),
    "strafe_left": (Button.MOVE_LEFT, Button.SPEED),
    "strafe_right": (Button.MOVE_RIGHT, Button.SPEED),
    "turn_left": (Button.TURN_LEFT,),
    "turn_right": (Button.TURN_RIGHT,),
    "shoot": (Button.ATTACK,),
    "use": (Button.USE,),
}

ACTION_NAMES = list(ACTIONS)


def supports_movement(game) -> bool:
    """Survival arenas only allow turning and shooting; levels allow walking."""
    return Button.MOVE_FORWARD in game.get_available_buttons()
