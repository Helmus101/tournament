"""arena.py  --  ONE self-contained file: the environment + the network + the
agent contract + the tournament. Everything you need to run a fair Pong
tournament between any number of models lives here.

    python arena.py                          # tournament: realpong.pt vs the built-in tracker
    python arena.py realpong.pt alice.pt     # any number of model files
    python arena.py realpong.pt bf random    # bf = scripted tracker, random = baseline
    python arena.py realpong.pt alice.py:alice.pt   # a custom-architecture submission
    python arena.py --games 11               # games per pairing

ADD YOUR OWN MODEL (two ways)
-----------------------------
1. Standard network  -> just drop a `.pt` file (a state_dict for `Net`) in this
   folder and pass its name:   python arena.py realpong.pt yourmodel.pt
2. Custom architecture -> put an `Agent` class in your own `.py` file and pass
   `yourfile.py:yourweights.pt`. The contract is:

       class Agent:
           def __init__(self, weights_path=None): ...
           def reset(self): ...                 # called at the start of each game
           def act(self, frame) -> int:         # frame: 80x80 float (own paddle on RIGHT)
               ...                               # return 2 (UP) or 3 (DOWN)

The environment is SYMMETRIC, so side never matters; see PongSym below for the
full specification.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ── actions ────────────────────────────────────────────────────────────────────
UP, DOWN, NOOP = 2, 3, 0

# ══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT  —  a clean, SYMMETRIC 2-player Pong (full spec in the docstring)
# ══════════════════════════════════════════════════════════════════════════════
SIZE         = 80          # field is SIZE x SIZE; a frame flattens to SIZE*SIZE = 6400
PADDLE_H     = 16          # paddle height in pixels
PADDLE_SPEED = 3.0         # pixels per step
PADDLE_X_L   = 3           # left paddle column
PADDLE_X_R   = SIZE - 4    # right paddle column (76)
BALL_SPEED   = 2.0         # horizontal pixels per step
BALL_VY_MAX  = 2.0         # max vertical pixels per step
POINTS       = 21          # a game is first to POINTS
MAX_STEPS    = 8000        # truncate if neither side reaches POINTS


class PongSym:
    """Symmetric 2-player Pong. Both sides are mirror-identical, so swapping
    sides is fair by construction. Each player's observation is CANONICAL: its
    own paddle is always on the RIGHT of an 80x80 binary frame (opponent on the
    left, ball in between)."""

    def __init__(self, seed=0, points=POINTS):
        self.rng = np.random.default_rng(seed)
        self.points = points
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.score_r = self.score_l = self.steps = 0
        self._serve()
        return self._obs()

    def _serve(self):
        self.pad_l = self.pad_r = (SIZE - PADDLE_H) / 2.0
        self.bx = SIZE / 2.0
        self.by = self.rng.uniform(SIZE * 0.3, SIZE * 0.7)
        self.bvy = float(self.rng.uniform(-BALL_VY_MAX, BALL_VY_MAX))
        self.bvx = float(BALL_SPEED * (1.0 if self.rng.random() < 0.5 else -1.0))

    def step(self, a_right, a_left):
        self._move("r", a_right)
        self._move("l", a_left)
        self.bx += self.bvx
        self.by += self.bvy
        if self.by <= 0:           self.by = 0.0;          self.bvy = abs(self.bvy)
        elif self.by >= SIZE - 1:  self.by = SIZE - 1.0;   self.bvy = -abs(self.bvy)

        reward_r = 0.0
        hit_r = hit_l = miss_r = miss_l = False
        if self.bx <= PADDLE_X_L + 1:
            if self.pad_l <= self.by <= self.pad_l + PADDLE_H:
                self.bx = PADDLE_X_L + 1; self._bounce(self.pad_l); hit_l = True
            elif self.bx < 0:
                reward_r = 1.0; miss_l = True
        elif self.bx >= PADDLE_X_R - 1:
            if self.pad_r <= self.by <= self.pad_r + PADDLE_H:
                self.bx = PADDLE_X_R - 1; self._bounce(self.pad_r); hit_r = True
            elif self.bx > SIZE - 1:
                reward_r = -1.0; miss_r = True

        if reward_r > 0:   self.score_r += 1; self._serve()
        elif reward_r < 0: self.score_l += 1; self._serve()

        self.steps += 1
        done = max(self.score_r, self.score_l) >= self.points or self.steps >= MAX_STEPS
        info = {"hit_r": hit_r, "hit_l": hit_l, "miss_r": miss_r, "miss_l": miss_l}
        return self._obs(), {"right": reward_r, "left": -reward_r}, done, info

    def _move(self, side, action):
        d = -PADDLE_SPEED if action == UP else PADDLE_SPEED if action == DOWN else 0.0
        if side == "r": self.pad_r = float(np.clip(self.pad_r + d, 0, SIZE - PADDLE_H))
        else:           self.pad_l = float(np.clip(self.pad_l + d, 0, SIZE - PADDLE_H))

    def _bounce(self, paddle_top):
        offset = (self.by - paddle_top) / PADDLE_H - 0.5
        self.bvx = -self.bvx
        self.bvy = float(np.clip(offset * 2.0 * BALL_VY_MAX, -BALL_VY_MAX, BALL_VY_MAX))

    def _render(self):
        f = np.zeros((SIZE, SIZE), np.float32)
        yl, yr = int(round(self.pad_l)), int(round(self.pad_r))
        f[yl:yl + PADDLE_H, PADDLE_X_L] = 1.0
        f[yr:yr + PADDLE_H, PADDLE_X_R] = 1.0
        f[int(round(np.clip(self.by, 0, SIZE - 1))), int(round(np.clip(self.bx, 0, SIZE - 1)))] = 1.0
        return f

    def _obs(self):
        true = self._render()
        return {"right": true, "left": true[:, ::-1].copy()}   # left sees mirror -> own paddle on right


# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK + AGENTS
# ══════════════════════════════════════════════════════════════════════════════
D = SIZE * SIZE


class Net(nn.Module):
    """Karpathy-style policy net: input = difference of two 80x80 frames (6400)."""
    def __init__(self, hidden=200):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.policy_head(h)).squeeze(-1), self.value_head(h).squeeze(-1)


class Agent:
    """Standard agent: a trained Net that acts on the env's 80x80 frame.
    Samples UP at P(UP) (stochastic) so seeded games vary naturally."""
    def __init__(self, weights_path=None, stochastic=True, seed=0):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        self.net.eval()
        self.prev = None
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

    def reset(self): self.prev = None

    @torch.no_grad()
    def act(self, frame):
        cur = frame.astype(np.float32).ravel()
        diff = cur - self.prev if self.prev is not None else np.zeros(D, np.float32)
        self.prev = cur
        prob, _ = self.net(torch.from_numpy(diff).unsqueeze(0))
        p = float(prob.item())
        up = self.rng.random() < p if self.stochastic else p > 0.5
        return UP if up else DOWN


class TrackerAgent:
    """Scripted reference ('bf'): move toward the ball. Strong but beatable."""
    def reset(self): pass
    def act(self, frame):
        ys, xs = np.nonzero(frame)
        if len(xs) == 0: return NOOP
        own, opp = xs.max(), xs.min()
        pad_c = ys[xs == own].mean()
        ball = (xs > opp + 1) & (xs < own - 1)
        if not ball.any(): return NOOP
        by = ys[ball].mean()
        return UP if by < pad_c - 2 else DOWN if by > pad_c + 2 else NOOP


class RandomAgent:
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def reset(self): pass
    def act(self, frame): return int(self.rng.choice([UP, DOWN]))


def load_custom(py_path, pt_path):
    spec = importlib.util.spec_from_file_location("submission", py_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Agent(pt_path)


def make_agent(spec, seed):
    if spec == "bf":     return TrackerAgent()
    if spec == "random": return RandomAgent(seed)
    if ":" in spec and spec.split(":")[0].endswith(".py"):
        py, pt = spec.split(":", 1); return load_custom(py, pt)
    path = spec if os.path.exists(spec) else os.path.join(os.path.dirname(os.path.abspath(__file__)), spec)
    return Agent(path, seed=seed)


def short_name(spec):
    if spec in ("bf", "random"): return spec
    base = spec.split(":")[0] if ":" in spec else spec
    return os.path.splitext(os.path.basename(base))[0]


# ══════════════════════════════════════════════════════════════════════════════
#  TOURNAMENT
# ══════════════════════════════════════════════════════════════════════════════
class Viewer:
    """Optional pygame window that shows the games live. Up/Down = speed, Esc = quit."""
    def __init__(self):
        import pygame
        self.pg = pygame
        pygame.init()
        self.scale = 7
        self.ox, self.oy = 50, 70
        self.W = SIZE * self.scale + 2 * self.ox
        self.H = SIZE * self.scale + self.oy + 40
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Pong arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Courier New", 24, bold=True)
        self.small = pygame.font.SysFont("Courier New", 14)
        self.fps = 90

    def pump(self):
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT: return False
            if e.type == self.pg.KEYDOWN:
                if e.key == self.pg.K_ESCAPE: return False
                if e.key == self.pg.K_UP:   self.fps = min(self.fps + 30, 480)
                if e.key == self.pg.K_DOWN: self.fps = max(self.fps - 30, 30)
        return True

    def draw(self, env, nr, nl, sr, sl):
        pg, s, sc = self.pg, self.screen, self.scale
        gx, gy, gw, gh = self.ox, self.oy, SIZE * sc, SIZE * sc
        s.fill((12, 12, 20))
        pg.draw.rect(s, (20, 24, 36), (gx, gy, gw, gh))
        pg.draw.rect(s, (50, 55, 75), (gx, gy, gw, gh), 2)
        pg.draw.rect(s, (80, 200, 90), (gx + PADDLE_X_R * sc, gy + int(env.pad_r) * sc, sc, PADDLE_H * sc))
        pg.draw.rect(s, (220, 120, 50), (gx + PADDLE_X_L * sc, gy + int(env.pad_l) * sc, sc, PADDLE_H * sc))
        pg.draw.rect(s, (255, 230, 80), (gx + int(env.bx) * sc, gy + int(env.by) * sc, sc, sc))
        s.blit(self.font.render(f"{nl} {sl}", True, (220, 120, 50)), (gx, 22))
        t = self.font.render(f"{sr} {nr}", True, (80, 200, 90)); s.blit(t, (gx + gw - t.get_width(), 22))
        s.blit(self.small.render(f"speed {self.fps}fps   Up/Down   Esc quit", True, (120, 120, 140)),
               (gx, gy + gh + 8))
        pg.display.flip()
        self.clock.tick(self.fps)

    def close(self): self.pg.quit()


def play_game(agent_r, agent_l, seed, viewer=None, nr="R", nl="L"):
    env = PongSym(seed=seed)
    obs = env.reset(seed=seed)
    agent_r.reset(); agent_l.reset()
    done = False
    while not done:
        if viewer and not viewer.pump():
            raise KeyboardInterrupt
        obs, _, done, _ = env.step(agent_r.act(obs["right"]), agent_l.act(obs["left"]))
        if viewer:
            viewer.draw(env, nr, nl, env.score_r, env.score_l)
    return env.score_r, env.score_l


def run(specs, games, watch=False):
    names = [short_name(s) for s in specs]
    agents = {s: make_agent(s, seed=i + 1) for i, s in enumerate(specs)}
    pts = {s: 0 for s in specs}
    wins = {s: 0 for s in specs}               # match wins
    gwins = {s: 0 for s in specs}              # individual-game wins
    gplayed = {s: 0 for s in specs}
    viewer = Viewer() if watch else None

    try:
        for sa, sb in itertools.combinations(specs, 2):
            na, nb = short_name(sa), short_name(sb)
            print("=" * 56)
            print(f"  {na}  vs  {nb}   ({games} games)")
            print("=" * 56)
            pa = pb = 0
            for g in range(1, games + 1):
                # symmetric env, but alternate sides anyway as belt-and-braces fairness
                if g % 2 == 1:
                    sr, sl = play_game(agents[sa], agents[sb], 1000 + g, viewer, na, nb); ga, gb = sr, sl
                else:
                    sr, sl = play_game(agents[sb], agents[sa], 1000 + g, viewer, nb, na); ga, gb = sl, sr
                pa += ga; pb += gb
                gplayed[sa] += 1; gplayed[sb] += 1
                if ga > gb:   gwins[sa] += 1
                elif gb > ga: gwins[sb] += 1
                print(f"  game {g:2d}/{games}  {na} {ga:2d} - {gb:2d} {nb}   (total {pa}-{pb})")
            pts[sa] += pa; pts[sb] += pb
            if pa > pb: wins[sa] += 1
            elif pb > pa: wins[sb] += 1
            print("-" * 56)
            print(f"  {na} {pa} - {pb} {nb}\n")
    except KeyboardInterrupt:
        print("\n[interface closed early]")
    finally:
        if viewer: viewer.close()

    ranking = sorted(specs, key=lambda s: (wins[s], pts[s]), reverse=True)
    print("=" * 56)
    print("  FINAL STANDINGS")
    print("=" * 56)
    print(f"  {'#':<3}{'model':<18}{'wins':>6}{'points':>8}{'winrate':>9}")
    print("  " + "-" * 52)
    for i, s in enumerate(ranking, 1):
        wr = (gwins[s] / gplayed[s] * 100) if gplayed[s] else 0.0
        print(f"  {i:<3}{short_name(s):<18}{wins[s]:>6}{pts[s]:>8}{wr:>8.0f}%")
    print("=" * 56)
    print(f"  CHAMPION: {short_name(ranking[0])}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="Symmetric Pong tournament.")
    ap.add_argument("models", nargs="*", help="model.pt | bf | random | file.py:weights.pt")
    ap.add_argument("--games", type=int, default=11, help="games per pairing (default 11)")
    ap.add_argument("--watch", action="store_true", help="open a window and watch the games live")
    args = ap.parse_args()

    specs = args.models or ["realpong.pt", "bf"]
    if len(specs) < 2:
        specs = specs + ["bf"]
    run(specs, args.games, watch=args.watch)


if __name__ == "__main__":
    main()
