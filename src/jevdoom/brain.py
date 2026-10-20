"""The brain: the questions asked to the Jev. In a survival arena there is one
reactive question (SURVIVAL_QUESTION): turn to aim or shoot. In a level with a floor
plan and an exit (level.py) there is also a single question, the TACTIC for this
moment (TACTIC_QUESTION), which pilot.py carries out with aim and route. Without a
floor plan (legacy loop in main.py) there are two levels: every few steps the Jev
picks a GOAL that holds for the next steps (GOAL_QUESTION) and, every step, the
ACTION that serves it (ACTION_QUESTION). Takes the state dict built by session.py or
main.py and returns typed answers with confidence and probabilities."""

from dotenv import load_dotenv
from typesafe_sdk import Choice, ChoiceAnswer, TypeSafeClient

load_dotenv()

GOAL_QUESTION = Choice(
    instructions=(
        "You command a player in a Doom level whose final aim is to FIND THE EXIT. "
        "Decide only WHERE to go over the next steps: combat is a reflex and does not "
        "belong here. Going after an item, however, is a heading — if health is low and "
        "a health item is in sight, getting it comes before exploring."
    ),
    criteria={
        "explore": "Follow 'unexplored_bearing' to discover new ground. The default heading while there is somewhere to go and 'steps_without_discovery' is low.",
        "open_path": "Go to a wall not tried yet and try to open it, because in Doom doors and switches look like walls. Choose when 'unexplored_bearing' is null or exploration has stalled and 'untested_walls' is greater than zero.",
        "collect_item": "Go pick up a visible key or weapon, which usually unlocks progress in the level.",
        "recover_health": "Go to the nearest health item. Choose whenever 'low_health' is true and 'health_item' is not null — dying ends every other plan.",
    },
)

ACTION_QUESTION = Choice(
    instructions=(
        "You control a player in a Doom level. SURVIVING always comes before the heading: "
        "if 'enemy_ahead' is true, shoot; if 'under_attack' is true and you see nobody, "
        "turn to find the shooter; with health below 40, prioritise health items. "
        "Otherwise carry out 'current_goal': the fields 'unexplored_bearing', "
        "'untested_wall_bearing' and the 'direction' inside 'health_item' or 'items' tell "
        "which side each target is on, and you only really move forward when it is "
        "ahead. If your goal's heading is 'left', 'right' or 'behind', turn to line up "
        "before walking. What is the best action now?"
    ),
    criteria={
        "move_forward": "Move forward when not blocked and the current goal's heading is already ahead, or to reach an item ahead.",
        "use": "Open a door or press a switch ahead. Choose when 'blocked' is true and 'already_tried_use' is false, and whenever the goal is open_path and you face the wall.",
        "turn_left": "Turn left to line up with the goal's heading, aim at an enemy on the left, or look for the shooter when 'under_attack'.",
        "turn_right": "Turn right to line up with the goal's heading, aim at an enemy on the right, or look for the shooter when 'under_attack'.",
        "shoot": "Fire at any enemy ahead, near or far, whenever 'enemy_ahead' is true.",
        "move_back": "Back off to get unstuck when 'blocked_attempts' is above three, or to get away from a nearby enemy with low health.",
        "strafe_left": "Step sideways to the left without turning: goes around an obstacle ahead or dodges a shooter.",
        "strafe_right": "Step sideways to the right without turning: goes around an obstacle ahead or dodges a shooter.",
    },
)


TACTIC_QUESTION = Choice(
    instructions=(
        "You play a Doom level like a good human player and want to reach the EXIT "
        "alive. Your hands already know how to aim, run, open doors and follow the path "
        "to the current 'objective' (the exit, a key, a switch); you only decide the "
        "TACTIC for this moment. 'enemies' is sorted from the most to the least urgent, "
        "with 'distance' in map units and 'angle' in degrees (positive = to the left, 0 "
        "= in your sights). A live enemy in view is almost always dealt with before "
        "moving on: running with your back to it is how you get shot. What is the best "
        "tactic now?"
    ),
    criteria={
        "follow_route": "Run along the path to the 'objective' (the hands open doors and switches on the way by themselves). Choose when no enemy is closer than 600, or there are only far enemies and 'under_attack' is false.",
        "attack": "Turn to the priority enemy and shoot while strafing. The default with an enemy closer than 600 (or farther if 'under_attack'), and whenever a shooter is point blank (under 256): turning your back on it means taking the whole blast.",
        "strafe_left": "Run sideways to the left keeping your aim. Use when 'melee_threat_close' is true but 'escape_route' is false.",
        "strafe_right": "Run sideways to the right keeping your aim. Use when 'melee_threat_close' is true but 'escape_route' is false.",
        "take_cover": "Move behind a corner the shooters cannot see, keeping your aim. Choose when 'shooters_in_view' is 2 or more and 'cover_nearby' is true and no enemy is closer than 256; or when 'low_health' is true, an enemy is closer than 400 and 'cover_nearby' is true.",
        "retreat": "Back off toward open ground while shooting. Choose whenever 'melee_threat_close' is true and 'escape_route' is true (a Demon bites, an Imp claws: the player outruns them); or with 'low_health', an enemy closer than 400 and no 'cover_nearby'.",
        "search": "Turn and step away from the wall to find the shooter. Choose when 'under_attack' is true and no enemy is in view; 'remembered_threat' is the angle where one was last seen.",
        "grab_item": "Go to 'wanted_item'. Choose when it exists, is closer than 300 and no enemy is near: a health item with 'low_health', or a new weapon.",
    },
)


SURVIVAL_QUESTION = Choice(
    instructions=(
        "You are stuck in the middle of a Doom arena and CANNOT move: only turn and "
        "shoot. Monsters spawn around you and close in; survive as long as possible. "
        "The 'enemies' list is sorted from nearest to farthest, so treat the first one "
        "as the main threat. Ammo is limited and running out means certain death: do "
        "not shoot at nothing. What is the best action now?"
    ),
    criteria={
        "shoot": "Fire at the threat already lined up. Choose whenever 'enemy_ahead' is true, with top priority if the nearest enemy is near.",
        "turn_left": "Turn left to line up the nearest enemy when it is on the left, or to sweep the arena when no enemy is in view.",
        "turn_right": "Turn right to line up the nearest enemy when it is on the right, or to sweep the arena when no enemy is in view.",
    },
)


def make_client() -> TypeSafeClient:
    return TypeSafeClient()


def choose_goal(client: TypeSafeClient, state: dict) -> ChoiceAnswer:
    return client.system_one(state=state, questions={"goal": GOAL_QUESTION}).answers["goal"]


def choose_action(client: TypeSafeClient, state: dict, question: Choice) -> ChoiceAnswer:
    return client.system_one(state=state, questions={"action": question}).answers["action"]
