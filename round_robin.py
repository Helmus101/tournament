"""round_robin.py  --  3-model round-robin tournament on Atari Pong (pong_v3).

Plays every pair of models against each other and ranks them.

Models: realpong, karpathy_pong, ppo   (all loaded from this folder).

Fairness
--------
pong_v3 ignores the seed and structurally favours the LEFT paddle, so a single
fixed-side game is meaningless. Each "fair game" is therefore played as TWO
Atari legs with the sides SWAPPED; a model's score for the fair game is its
TOTAL points across both legs. Each pairing plays N fair games (default 21,
each leg to 21 points). Models are ranked by match wins, then by total point
differential.

Run:
    python tournament/round_robin.py              # 21 fair games per pairing
    python tournament/round_robin.py --games 5
    python tournament/round_robin.py --output tournament/results/round_robin.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pettingzoo.atari import pong_v3
from agent_ale import Agent as MLPAgent

POINTS_TO_WIN = 21
MAX_CYCLES = 8000
DEFAULT_GAMES = 21


# ── model registry: name -> factory that builds a fresh agent ─────────────────
def load_ppo():
    from ppo_agent import Agent as PPOAgent
    return PPOAgent(str(HERE / "ppo.pt"))

MODELS: dict[str, callable] = {
    "realpong":      lambda: MLPAgent(str(HERE / "realpong.pt")),
    "karpathy_pong": lambda: MLPAgent(str(HERE / "karpathy_pong.pt")),
    "ppo":           load_ppo,
}


# ── one Atari leg: right = first_0 (normal), left = second_0 (mirrored) ────────
def play_leg(env, right_agent, left_agent) -> tuple[int, int]:
    obs, _ = env.reset()
    right_agent.reset(); left_agent.reset()
    sr = sl = 0
    while env.agents and max(sr, sl) < POINTS_TO_WIN:
        frame = obs["first_0"]
        a_r = right_agent.act(frame)
        a_l = left_agent.act(frame[:, ::-1, :])
        obs, rew, _, _, _ = env.step({"first_0": a_r, "second_0": a_l})
        r = rew.get("first_0", 0.0)
        if r > 0:   sr += 1
        elif r < 0: sl += 1
    return sr, sl


# ── one fair game = two legs with sides swapped ───────────────────────────────
def play_fair_game(env, agent_a, agent_b) -> tuple[int, int]:
    r1, l1 = play_leg(env, agent_a, agent_b)   # A on the right
    r2, l2 = play_leg(env, agent_b, agent_a)   # B on the right
    a_pts = r1 + l2
    b_pts = l1 + r2
    return a_pts, b_pts


@dataclass
class Standing:
    name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: int = 0
    points_against: int = 0

    @property
    def diff(self) -> int:
        return self.points_for - self.points_against


def run(games: int) -> dict:
    names = list(MODELS)
    agents = {n: MODELS[n]() for n in names}            # build each once
    table = {n: Standing(n) for n in names}
    env = pong_v3.parallel_env(max_cycles=MAX_CYCLES)

    matches = []
    for na, nb in itertools.combinations(names, 2):
        print("=" * 60)
        print(f"  {na}  vs  {nb}   ({games} fair games, sides swap each game)")
        print("=" * 60)
        pa = pb = 0
        for g in range(1, games + 1):
            ga, gb = play_fair_game(env, agents[na], agents[nb])
            pa += ga; pb += gb
            print(f"  game {g:2d}/{games}  {na} {ga:2d} - {gb:2d} {nb}"
                  f"   (total {pa}-{pb})")

        table[na].points_for += pa; table[na].points_against += pb
        table[nb].points_for += pb; table[nb].points_against += pa
        if pa > pb:
            table[na].wins += 1; table[nb].losses += 1; winner = na
        elif pb > pa:
            table[nb].wins += 1; table[na].losses += 1; winner = nb
        else:
            table[na].ties += 1; table[nb].ties += 1; winner = "tie"
        print("-" * 60)
        print(f"  result: {na} {pa} - {pb} {nb}  ->  {winner}\n")
        matches.append({"a": na, "b": nb, "a_points": pa, "b_points": pb, "winner": winner})

    env.close()

    # rank: wins desc, then point differential desc
    ranking = sorted(names, key=lambda n: (table[n].wins, table[n].diff), reverse=True)

    print("=" * 60)
    print("  FINAL STANDINGS")
    print("=" * 60)
    print(f"  {'#':<3}{'model':<16}{'W':>3}{'L':>3}{'T':>3}{'pts+':>7}{'pts-':>7}{'diff':>7}")
    print("  " + "-" * 54)
    for i, n in enumerate(ranking, 1):
        s = table[n]
        print(f"  {i:<3}{n:<16}{s.wins:>3}{s.losses:>3}{s.ties:>3}"
              f"{s.points_for:>7}{s.points_against:>7}{s.diff:>+7}")
    print("=" * 60)
    print(f"  CHAMPION: {ranking[0]}")
    print("=" * 60)

    return {
        "games_per_pairing": games,
        "matches": matches,
        "standings": [asdict(table[n]) | {"diff": table[n].diff} for n in ranking],
        "champion": ranking[0],
    }


def main():
    ap = argparse.ArgumentParser(description="3-model round-robin on pong_v3")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES,
                    help="fair games per pairing (default 21)")
    ap.add_argument("--output", type=str, default=str(HERE / "results" / "round_robin.json"))
    args = ap.parse_args()

    result = run(args.games)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"  saved    | {args.output}")


if __name__ == "__main__":
    main()
