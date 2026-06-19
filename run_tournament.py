"""Run realpong against a selected opponent using the code in new_v3."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOURNAMENT_DIR = Path(__file__).resolve().parent
if str(TOURNAMENT_DIR) not in sys.path:
    sys.path.insert(0, str(TOURNAMENT_DIR))


POINTS_TO_WIN = 21
MAX_CYCLES = 8000
DEFAULT_GAMES = 6
DEFAULT_OPPONENT = "ball_follower"


class BallFollowerAgent:
    """Scripted agent that follows the ball from its own mirrored view."""

    def __init__(self) -> None:
        self.last_ball_y = None
        from agent_ale import ball_follower_action

        self.ball_follower_action = ball_follower_action

    def reset(self) -> None:
        self.last_ball_y = None

    def act(self, frame) -> int:
        action, self.last_ball_y = self.ball_follower_action(
            frame, side="right", last_ball_y=self.last_ball_y
        )
        return action


@dataclass
class GameResult:
    game: int
    realpong_side: str
    realpong_points: int
    opponent_points: int


@dataclass
class TournamentResult:
    opponent: str
    games_requested: int
    games_played: int
    realpong_total: int
    opponent_total: int
    winner: str
    games: list[GameResult]


def play_game(env, right_agent, left_agent) -> tuple[int, int]:
    """Play one Atari Pong game.

    Returns:
        A tuple of (right_score, left_score). The right paddle is `first_0`.
    """
    obs, _ = env.reset()
    right_agent.reset()
    left_agent.reset()
    right_score = 0
    left_score = 0

    while env.agents and max(right_score, left_score) < POINTS_TO_WIN:
        frame = obs["first_0"]
        right_action = right_agent.act(frame)
        left_action = left_agent.act(frame[:, ::-1, :])
        obs, rewards, _, _, _ = env.step(
            {"first_0": right_action, "second_0": left_action}
        )

        reward = rewards.get("first_0", 0.0)
        if reward > 0:
            right_score += 1
        elif reward < 0:
            left_score += 1

    return right_score, left_score


def make_opponent(name: str):
    if name == "ball_follower":
        return BallFollowerAgent()
    if name == "karpathy_pong":
        from agent_ale import Agent

        return Agent(str(TOURNAMENT_DIR / "karpathy_pong.pt"))
    raise ValueError(f"unknown opponent: {name}")


def run_tournament(games: int, opponent_name: str) -> TournamentResult:
    from pettingzoo.atari import pong_v3
    from agent_ale import Agent

    games_requested = games
    if games < 2:
        raise ValueError("games must be at least 2")
    if games % 2:
        games += 1

    realpong = Agent(str(TOURNAMENT_DIR / "realpong.pt"))
    opponent = make_opponent(opponent_name)
    env = pong_v3.parallel_env(max_cycles=MAX_CYCLES)

    realpong_total = 0
    opponent_total = 0
    game_results: list[GameResult] = []

    try:
        for game_number in range(1, games + 1):
            if game_number % 2:
                right_score, left_score = play_game(env, realpong, opponent)
                realpong_points = right_score
                opponent_points = left_score
                realpong_side = "right"
            else:
                right_score, left_score = play_game(env, opponent, realpong)
                realpong_points = left_score
                opponent_points = right_score
                realpong_side = "left"

            realpong_total += realpong_points
            opponent_total += opponent_points
            game_results.append(
                GameResult(
                    game=game_number,
                    realpong_side=realpong_side,
                    realpong_points=realpong_points,
                    opponent_points=opponent_points,
                )
            )
    finally:
        env.close()

    if realpong_total > opponent_total:
        winner = "realpong"
    elif opponent_total > realpong_total:
        winner = opponent_name
    else:
        winner = "tie"

    return TournamentResult(
        opponent=opponent_name,
        games_requested=games_requested,
        games_played=games,
        realpong_total=realpong_total,
        opponent_total=opponent_total,
        winner=winner,
        games=game_results,
    )


def write_result(result: TournamentResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_summary(result: TournamentResult, output_path: Path | None) -> None:
    print("=" * 64)
    print(f"realpong vs {result.opponent} ({result.games_played} games)")
    print("=" * 64)
    for game in result.games:
        print(
            f"game {game.game:02d} | realpong {game.realpong_side:>5} | "
            f"realpong {game.realpong_points:2d} - "
            f"{game.opponent_points:2d} {result.opponent}"
        )
    print("-" * 64)
    print(
        f"total    | realpong {result.realpong_total} - "
        f"{result.opponent_total} {result.opponent}"
    )
    print(f"winner   | {result.winner}")
    if output_path is not None:
        print(f"saved    | {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a realpong tournament."
    )
    parser.add_argument(
        "--opponent",
        choices=["ball_follower", "karpathy_pong"],
        default=DEFAULT_OPPONENT,
        help=f"Opponent to play against realpong. Default: {DEFAULT_OPPONENT}",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help=f"Number of games to play. Odd numbers are rounded up. Default: {DEFAULT_GAMES}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tournament" / "results" / "latest.json",
        help="Where to save the JSON result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tournament(args.games, args.opponent)
    write_result(result, args.output)
    print_summary(result, args.output)


if __name__ == "__main__":
    main()
