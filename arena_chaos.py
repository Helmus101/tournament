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

import numpy as np

import arena
from arena import (PongSym, SIZE, PADDLE_H, BALL_VY_MAX, PADDLE_KICK)

# ── chaos parameters ─────────────────────────────────────────────────────────
CHAOS_SPEED_MIN = 1.5     # ball-speed floor (px/step)
CHAOS_SPEED_MAX = 6.0     # ball-speed ceiling -- harder than the standard 5.0 cap, still
                          # well under the field width so collisions can't tunnel
JOLT_PROB       = 0.04    # per-step chance the ball's speed suddenly changes mid-flight


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


# Swap the env into arena's tournament machinery: play_game() builds its env from
# the module-global `arena.PongSym`, so reassigning it routes every game through ChaosPong.
arena.PongSym = ChaosPong


def main():
    print("=" * 56)
    print("  CHAOS ARENA  --  ball speed is random & non-physical")
    print(f"  speed in [{CHAOS_SPEED_MIN}, {CHAOS_SPEED_MAX}] px/step, re-rolled on serve,")
    print(f"  on every hit, and ~{JOLT_PROB*100:.0f}% of steps mid-flight")
    print("=" * 56)
    arena.main()        # reuse arena's CLI / round-robin / standings, now on ChaosPong


if __name__ == "__main__":
    main()
