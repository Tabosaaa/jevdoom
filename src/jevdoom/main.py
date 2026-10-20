"""Command line. Levels with an exit and arenas run in session.py (the Jev or the
--autopilot reflex decides); levels without a recognisable exit fall back to the
legacy exploration loop (navigator + goal/action, Phases 6–10). Also: you play in
the Doom window (--human) or inspect the state (--perceive). For the full UI:
server.py."""

import argparse
import json

import vizdoom as vzd

from jevdoom import actions, brain, config, engine, navigator, perception, session

ESCAPE_MOVES = (
    "strafe_right", "move_forward",
    "strafe_left", "move_forward",
    "move_back", "turn_right",
)


def escape_move(stationary_steps: int) -> str:
    """Unsticking manoeuvre, decided in code. Being stuck against a wall is a matter of
    liveness, not judgement, so it does not depend on the model. It starts with
    strafes, which free the character without throwing away the direction it was
    going; it only escalates to backing off and turning if still stuck. Measured: this
    order explores 52 areas vs 39 for a manoeuvre that turns first."""
    return ESCAPE_MOVES[(stationary_steps - config.ESCAPE_AFTER) % len(ESCAPE_MOVES)]


def kills(game: vzd.DoomGame) -> float:
    return game.get_game_variable(vzd.GameVariable.KILLCOUNT)


def read_state(
    game: vzd.DoomGame,
    nav: navigator.Navigator,
    last_action: str | None,
    survival: bool,
) -> dict:
    nav.update(game, last_action)
    state = perception.perceive(game).to_state()
    if survival:
        state["under_attack"] = nav.facts()["under_attack"]
        return state
    return {**state, **nav.facts()}


def play_session(args) -> bool:
    """A level with an exit or an arena: the whole game runs in session.py. Returns
    False if the scenario does not fit (no recognisable exit) and the legacy loop takes over."""
    cfg = session.SessionConfig(
        scenario=args.scenario, doom_map=args.doom_map or ("MAP01" if engine.iwad_path(args.scenario) else None),
        mode="autopilot" if args.autopilot else "jev", tics=args.tics, episodes=args.episodes,
        max_steps=args.max_steps, visible=not args.no_window, realtime=args.realtime,
    )

    def emit(kind: str, data: dict) -> None:
        if kind == "episode":
            print(f"episode {data['episode']}: {data['steps']} steps | {data['kills']} kills | "
                  f"health {data['health']} | {data['outcome'].upper()}")
        elif kind == "end":
            print(f"log saved to {data['log']}")

    try:
        session.Session(cfg, emit).run()
    except session.NoExitError:
        return False
    return True


def play_jev(scenario: str, doom_map: str, episodes: int, visible: bool, max_steps: int | None) -> None:
    """Legacy loop (Phases 6–10): the Jev picks a goal every few steps and an action every step."""
    log_path = session.new_log_path()
    with brain.make_client() as client, open(log_path, "w", encoding="utf-8") as log:
        game = engine.make_game(scenario=scenario, visible=visible, doom_map=doom_map)
        survival = not actions.supports_movement(game)
        question = brain.SURVIVAL_QUESTION if survival else brain.ACTION_QUESTION
        tics = config.SURVIVAL_TICS if survival else config.DECISION_TICS
        fallback = config.SURVIVAL_DEFAULT_ACTION if survival else config.DEFAULT_ACTION
        try:
            for ep in range(1, episodes + 1):
                game.new_episode()
                nav = navigator.Navigator()
                steps = 0
                action = None
                goal = None
                goal_answer = None
                while not game.is_episode_finished():
                    state = read_state(game, nav, action, survival)
                    if not survival and (goal is None or steps % config.GOAL_INTERVAL == 0):
                        goal_answer = brain.choose_goal(client, state)
                        goal = goal_answer.choice
                    if goal is not None:
                        state["current_goal"] = goal
                    answer = brain.choose_action(client, state, question)
                    action = answer.choice if answer.confidence >= config.MIN_CONFIDENCE else fallback
                    stationary = state.get("stationary_for", 0)
                    if stationary >= config.ESCAPE_AFTER:
                        action = escape_move(stationary)
                    reward = engine.step(game, action, tics)
                    log.write(json.dumps({
                        "episode": ep,
                        "step": steps,
                        "action": action,
                        "confidence": answer.confidence,
                        "probabilities": dict(answer.probabilities),
                        "goal": goal,
                        "goal_confidence": goal_answer.confidence if goal_answer else None,
                        "goal_probabilities": dict(goal_answer.probabilities) if goal_answer else None,
                        "state": state,
                        "reward": reward,
                        "unsticking": stationary >= config.ESCAPE_AFTER,
                    }, ensure_ascii=False) + "\n")
                    steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                if survival:
                    print(f"episode {ep}: survived {steps} steps | "
                          f"{int(kills(game))} kills | reward {game.get_total_reward():.1f}")
                else:
                    print(f"episode {ep}: {steps} steps | reward {game.get_total_reward():.1f} | "
                          f"explored areas {len(nav.visited)} | "
                          f"untested walls {len(nav.walls - nav.use_tried)} | "
                          f"{'LEVEL COMPLETE' if game.is_episode_finished() else 'stopped'}")
        finally:
            game.close()
    print(f"log saved to {log_path}")


def play_human(scenario: str, doom_map: str, episodes: int) -> None:
    game = engine.make_game(scenario=scenario, visible=True, mode=vzd.Mode.SPECTATOR, doom_map=doom_map)
    print("Click the Doom window to focus it and play with the keyboard.")
    try:
        for ep in range(1, episodes + 1):
            game.new_episode()
            while not game.is_episode_finished():
                game.advance_action()
            print(f"episode {ep}: reward = {game.get_total_reward():.1f}")
    except vzd.ViZDoomUnexpectedExitException:
        print("Doom window closed — exiting.")
    finally:
        game.close()


def perceive_demo(scenario: str, doom_map: str, episodes: int, visible: bool, max_steps: int | None) -> None:
    game = engine.make_game(scenario=scenario, visible=visible, doom_map=doom_map)
    survival = not actions.supports_movement(game)
    scan = config.SURVIVAL_DEFAULT_ACTION if survival else config.DEFAULT_ACTION
    try:
        for ep in range(1, episodes + 1):
            game.new_episode()
            nav = navigator.Navigator()
            steps = 0
            while not game.is_episode_finished():
                print(json.dumps(read_state(game, nav, scan, survival), ensure_ascii=False))
                engine.step(game, scan)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
            print(f"--- end of episode {ep} ---")
    finally:
        game.close()


def main() -> None:
    p = argparse.ArgumentParser(description="jevDoom — the Jev plays Doom")
    p.add_argument("--scenario", default="defend_the_center")
    p.add_argument("--map", dest="doom_map", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--human", action="store_true")
    p.add_argument("--perceive", action="store_true")
    p.add_argument("--autopilot", action="store_true", help="play with the reflex only, no API")
    p.add_argument("--tics", type=int, default=config.DECISION_TICS, help="tics per decision (35 = 1 s)")
    p.add_argument("--realtime", action="store_true", help="play at a person's pace")
    args = p.parse_args()

    if args.human:
        play_human(args.scenario, args.doom_map, args.episodes)
    elif args.perceive:
        perceive_demo(args.scenario, args.doom_map, args.episodes, not args.no_window, args.max_steps)
    elif not play_session(args):
        play_jev(args.scenario, args.doom_map, args.episodes, not args.no_window, args.max_steps)


if __name__ == "__main__":
    main()
