"""TEMPLATE for a tournament submission  —  copy this file, fill it in, rename it.

A submission is TWO files:
    1) this .py  (your model's code: architecture + how it picks a move)
    2) a .pt     (your trained weights)

You enter the tournament with:
    python arena.py yourname.py:yourname.pt realpong.py:realpong.pt

THE ENVIRONMENT (what your agent plays in)
------------------------------------------
  * Field is 80 x 80. You are handed an 80x80 float frame each step:
        1.0 = paddle or ball pixel, 0.0 = empty.
  * Your OWN paddle is ALWAYS on the RIGHT of the frame, the opponent on the
    left, the ball in between. (The env is symmetric and mirrors the view for
    you, so you never have to care which physical side you're on.)
  * You return an action each step:  2 = UP, 3 = DOWN  (0 = stay still).
  * A match is first to 21 points.

THE CONTRACT
------------
Your file MUST define a class named exactly `Agent` with these three methods.
That's the only requirement — the network inside can be anything (MLP, CNN, ...).
"""
import numpy as np
# import torch          # uncomment if your model is a neural net

UP, DOWN, STAY = 2, 3, 0


class Agent:
    def __init__(self, weights_path=None):
        """Build your model and load weights_path (a .pt) if provided."""
        # e.g. self.net = MyNet(); self.net.load_state_dict(torch.load(weights_path))
        pass

    def reset(self):
        """Called once at the start of every game. Clear any per-game state."""
        pass

    def act(self, frame) -> int:
        """frame: 80x80 float array, YOUR paddle on the right. Return 2 / 3 / 0."""
        # --- replace this simple ball-tracker with your model ---
        ys, xs = np.nonzero(frame)
        if len(xs) == 0:
            return STAY
        own_col, opp_col = xs.max(), xs.min()           # own paddle = rightmost column
        paddle_y = ys[xs == own_col].mean()
        ball = (xs > opp_col + 1) & (xs < own_col - 1)   # ball = pixels between paddles
        if not ball.any():
            return STAY
        ball_y = ys[ball].mean()
        if ball_y < paddle_y - 2: return UP
        if ball_y > paddle_y + 2: return DOWN
        return STAY
