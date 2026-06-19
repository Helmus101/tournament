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
#  AGENTS  —  the arena is architecture-agnostic: every model brings its OWN code
#  (a .py with an `Agent` class) plus its weights (.pt). Nothing here assumes a
#  particular network. The only built-ins are two reference opponents.
# ══════════════════════════════════════════════════════════════════════════════
D = SIZE * SIZE


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


def _resolve(path):
    # Resolve relative to the tournament folder FIRST (where models live), so a
    # bare name like "realpong.py" never accidentally matches a same-named file
    # in the directory you happen to run from.
    if not path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, path)
    if os.path.exists(cand):
        return cand
    return path        # fall back to the path as given (absolute or cwd-relative)


def load_submission(py_path, pt_path):
    """Import a competitor's module and build its Agent(weights). The module
    carries the model's ORIGINAL code (architecture), so any design works."""
    py_path = _resolve(py_path)
    pt_path = _resolve(pt_path)
    modname = "submission_" + os.path.basename(py_path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(modname, py_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.Agent(pt_path)


def make_agent(spec, seed):
    if spec == "bf":     return TrackerAgent()
    if spec == "random": return RandomAgent(seed)
    if ":" in spec:                       # model.py:weights.pt  (code + weights)
        py, pt = spec.split(":", 1); return load_submission(py, pt)
    if spec.endswith(".py"):              # code only, the module finds its own weights
        return load_submission(spec, None)
    raise SystemExit(
        f"'{spec}': a model must bring its own code. Pass it as 'model.py:weights.pt' "
        f"(see realpong_agent.py for the contract). Built-in opponents: 'bf', 'random'.")


def short_name(spec):
    if spec in ("bf", "random"): return spec
    if ":" in spec:                       # name by the weights file -> e.g. realpong
        return os.path.splitext(os.path.basename(spec.split(":", 1)[1]))[0]
    return os.path.splitext(os.path.basename(spec))[0]


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


def _labels(specs):
    """Display label per entrant; duplicate models get #1, #2, ... so two
    entries of the same file are still counted as separate competitors."""
    base = [short_name(s) for s in specs]
    seen, out = {}, {}
    for i, b in enumerate(base):
        if base.count(b) > 1:
            seen[b] = seen.get(b, 0) + 1
            out[i] = f"{b}#{seen[b]}"
        else:
            out[i] = b
    return out


def _one_game(agents, label, ia, ib, seed, viewer, flip):
    """Single game to 21 between entrants ia, ib. `flip` swaps who is the right
    paddle (alternation); result is always returned as (score_a, score_b)."""
    na, nb = label[ia], label[ib]
    if not flip:
        sr, sl = play_game(agents[ia], agents[ib], seed, viewer, na, nb)
        return sr, sl
    sr, sl = play_game(agents[ib], agents[ia], seed, viewer, nb, na)
    return sl, sr


def play_set(agents, label, ia, ib, seedbox, viewer, best_of):
    """Best-of-N set, each game to 21. Returns the winning entrant index."""
    na, nb = label[ia], label[ib]
    need = best_of // 2 + 1                     # 2 game-wins for best-of-3
    wa = wb = g = 0
    print(f"  --- {na} vs {nb}  (best of {best_of}, games to 21) ---")
    while wa < need and wb < need:
        g += 1; seedbox[0] += 1
        a, b = _one_game(agents, label, ia, ib, 1000 + seedbox[0], viewer, flip=(g % 2 == 0))
        if a > b: wa += 1
        else:     wb += 1
        print(f"    game {g}: {na} {a:2d} - {b:2d} {nb}   (set {wa}-{wb})")
    winner = ia if wa > wb else ib
    print(f"    -> {label[winner]} wins the set\n")
    return winner


def tiebreak(group, agents, label, seedbox, viewer):
    """Order entrants tied on points via single games to 21 among them."""
    if len(group) <= 1:
        return group
    print(f"  *** TIEBREAK among {', '.join(label[i] for i in group)} (single game to 21) ***")
    w = {i: 0 for i in group}
    for ia, ib in itertools.combinations(group, 2):
        seedbox[0] += 1
        a, b = _one_game(agents, label, ia, ib, 7000 + seedbox[0], viewer, flip=False)
        print(f"    tiebreak: {label[ia]} {a:2d} - {b:2d} {label[ib]}")
        if a > b: w[ia] += 1
        else:     w[ib] += 1
    print()
    return sorted(group, key=lambda i: w[i], reverse=True)


def run(specs, best_of=3, watch=True):
    ids = list(range(len(specs)))
    agents = {i: make_agent(specs[i], seed=i + 1) for i in ids}   # keyed per ENTRANT
    label = _labels(specs)
    points = {i: 0 for i in ids}                # tournament points = sets won
    seedbox = [0]
    viewer = None
    if watch:
        try:
            viewer = Viewer()
        except Exception as e:
            print(f"[no display available, running headless: {e}]")

    order = list(ids)
    try:
        print("=" * 56)
        print(f"  ROUND-ROBIN  ({len(ids)} agents, best of {best_of} to 21, set win = 1 pt)")
        print("=" * 56)
        for ia, ib in itertools.combinations(ids, 2):
            points[play_set(agents, label, ia, ib, seedbox, viewer, best_of)] += 1

        # rank by points; break any ties with single games to 21
        order = []
        for p in sorted(set(points.values()), reverse=True):
            group = [i for i in ids if points[i] == p]
            order.extend(tiebreak(group, agents, label, seedbox, viewer))
    except KeyboardInterrupt:
        print("\n[interface closed early]")
        order = sorted(ids, key=lambda i: points[i], reverse=True)
    finally:
        if viewer: viewer.close()

    print("=" * 56)
    print("  FINAL STANDINGS")
    print("=" * 56)
    print(f"  {'#':<3}{'model':<20}{'points':>8}")
    print("  " + "-" * 44)
    for rank, i in enumerate(order, 1):
        print(f"  {rank:<3}{label[i]:<20}{points[i]:>8}")
    print("=" * 56)
    print(f"  CHAMPION: {label[order[0]]}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="Symmetric Pong tournament.")
    ap.add_argument("models", nargs="*", help="model.pt | bf | random | file.py:weights.pt")
    ap.add_argument("--best-of", type=int, default=3, dest="best_of",
                    help="games per set, each to 21 (default 3)")
    ap.add_argument("--headless", action="store_true", help="run without the visual window (text only)")
    ap.add_argument("--watch", action="store_true", help=argparse.SUPPRESS)  # legacy no-op (visual is default)
    args = ap.parse_args()

    specs = args.models or ["realpong.py:realpong.pt", "bf"]
    if len(specs) < 2:
        specs = specs + ["bf"]
    run(specs, best_of=args.best_of, watch=not args.headless)   # visual ON by default


if __name__ == "__main__":
    main()
