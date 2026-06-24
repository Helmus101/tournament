"""arena_chaos.py  --  a HARDER variant of the Pong arena.

Same field, paddles, scoring, and Agent contract as arena.py, but the BALL
SPEED is UNPREDICTABLE and NOT physics-based:

  * each serve picks a RANDOM speed (not the fixed 2.0),
  * on every paddle hit the ball is re-launched at a FRESH RANDOM speed
    (not the standard env's deterministic 1.08x-per-hit acceleration),
  * mid-flight the speed randomly JOLTS now and then (a "gust"): the magnitude
    suddenly jumps while the travel direction is preserved, so you cannot
    extrapolate the ball's arrival from a constant velocity.

The horizontal direction still flips on a paddle hit and the ball still bounces
off the top/bottom walls, so it is recognisably Pong -- but an agent that learned
the standard env's predictable acceleration cannot linearly predict where the
ball will land. This is the "harder arena".

Any existing submission plugs in UNCHANGED (the observation is the identical
80x80 canonical frame). Run a tournament exactly like the normal arena:

    python arena_chaos.py realpong.py:realpong.pt bf random
    python arena_chaos.py realpong.py:realpong.pt pong.py:pong_best.pt --headless
    python arena_chaos.py --best-of 5

Implementation note: this reuses arena.py's entire tournament/visualizer/agent
machinery and only swaps the ENVIRONMENT class, so the two arenas can never
drift apart.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import arena
from arena import (PongSym, SIZE, PADDLE_H, BALL_VY_MAX, PADDLE_KICK, MAX_STEPS)

# ── chaos parameters ─────────────────────────────────────────────────────────
CHAOS_SPEED_MIN = 1.5     # ball-speed floor (px/step)
CHAOS_SPEED_MAX = 6.0     # ball-speed ceiling -- harder than the standard 5.0 cap, still
                          # well under the field width so collisions can't tunnel
JOLT_PROB       = 0.04    # per-step chance the ball's speed suddenly changes mid-flight
SWAP_EVERY      = 2       # agents trade physical paddles every SWAP_EVERY rallies (points);
                          # 0 = never. The env is symmetric (canonical per-side frames) so this
                          # is fair-by-design and invisible to the agents -- existing weights
                          # need no retraining; it only re-binds who plays which side.


class ChaosPong(PongSym):
    """PongSym with random, non-physical ball speed (see module docstring)."""

    def _rand_speed(self):
        return float(self.rng.uniform(CHAOS_SPEED_MIN, CHAOS_SPEED_MAX))

    def _serve(self):
        super()._serve()                          # sets positions + a default velocity/direction
        # replace the fixed serve speed with a random one (keep the served direction)
        self.bvx = float(np.sign(self.bvx) * self._rand_speed())
        self.bvy = float(self.rng.uniform(-BALL_VY_MAX, BALL_VY_MAX))

    def _bounce(self, paddle_top, paddle_vel=0.0):
        # re-launch at a FRESH RANDOM speed (NOT deterministic acceleration); angle still
        # depends on where the ball met the paddle, so placement skill still matters.
        speed = self._rand_speed()
        offset = (self.by - paddle_top) / PADDLE_H - 0.5
        self.bvx = float(np.sign(-self.bvx) * speed)
        self.bvy = float(np.clip(offset * 2.0 * speed + paddle_vel * PADDLE_KICK, -speed, speed))

    def step(self, a_right, a_left):
        # random mid-flight "gust": rescale the velocity to a new random total speed,
        # preserving direction -> the arrival point can't be predicted from constant motion.
        if self.rng.random() < JOLT_PROB:
            cur = (self.bvx ** 2 + self.bvy ** 2) ** 0.5
            if cur > 1e-6:
                scale = self._rand_speed() / cur
                self.bvx *= scale
                self.bvy *= scale
        return super().step(a_right, a_left)


def _chaos_play_game(agent_r, agent_l, seed, viewer=None, nr="R", nl="L"):
    """Like arena.play_game, but the two agents TRADE physical paddles every SWAP_EVERY
    rallies. Scores are tracked PER AGENT (not per physical side), so the result is correct
    across swaps. agent_r is 'A', agent_l is 'B'; returns (A_score, B_score). SWAP_EVERY==0
    reproduces the standard right/left game exactly. Swaps happen only at rally boundaries
    (the ball is freshly served then), so neither player is interrupted mid-defence."""
    env = arena.PongSym(seed=seed)                 # arena.PongSym is ChaosPong (patched below)
    obs = env.reset(seed=seed)
    agent_r.reset(); agent_l.reset()
    a_score = b_score = rallies = 0
    a_on_right = True                              # A currently controls the right physical paddle
    done = False
    while not done:
        if viewer and not viewer.pump():
            raise KeyboardInterrupt
        ar, al = (agent_r, agent_l) if a_on_right else (agent_l, agent_r)
        obs, rew, _, _ = env.step(ar.act(obs["right"]), al.act(obs["left"]))
        if rew["right"] != 0.0:                    # a point ended this rally (env re-served)
            a_won = (a_on_right == (rew["right"] > 0))   # A scored iff A held the side that scored
            if a_won: a_score += 1
            else:     b_score += 1
            rallies += 1
            if SWAP_EVERY and rallies % SWAP_EVERY == 0:
                a_on_right = not a_on_right
        done = max(a_score, b_score) >= env.points or env.steps >= MAX_STEPS
        if viewer:
            # each agent's name + running total follows it onto whichever side it now holds
            rn, rs = (nr, a_score) if a_on_right else (nl, b_score)
            ln, ls = (nl, b_score) if a_on_right else (nr, a_score)
            viewer.draw(env, rn, ln, rs, ls)
    return a_score, b_score


# Route arena's tournament machinery through the chaos env + side-switching game:
# play_game()/_one_game() resolve these names from arena's globals, so reassigning them
# here makes every chaos-arena game use ChaosPong AND trade sides every SWAP_EVERY rallies.
arena.PongSym = ChaosPong
arena.play_game = _chaos_play_game


def main():
    global SWAP_EVERY
    # pull --swap-every out before delegating the rest of the CLI to arena.main()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--swap-every", type=int, default=SWAP_EVERY, dest="swap_every")
    known, rest = pre.parse_known_args()
    SWAP_EVERY = known.swap_every
    sys.argv = [sys.argv[0]] + rest

    print("=" * 56)
    print("  CHAOS ARENA  --  ball speed is random & non-physical")
    print(f"  speed in [{CHAOS_SPEED_MIN}, {CHAOS_SPEED_MAX}] px/step, re-rolled on serve,")
    print(f"  on every hit, and ~{JOLT_PROB*100:.0f}% of steps mid-flight")
    if SWAP_EVERY:
        print(f"  side-switching: agents trade paddles every {SWAP_EVERY} rallies")
    print("=" * 56)
    arena.main()        # reuse arena's CLI / round-robin / standings, now on ChaosPong


if __name__ == "__main__":
    main()
