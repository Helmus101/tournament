"""realpong.py  --  the realpong MODEL (architecture + Agent) AND its trainer,
in one file.

  * Tournament loads this file for its `Agent` class:
        python arena.py realpong.py:realpong.pt  other.py:other.pt
  * You run this file to TRAIN (only happens when executed directly):
        python realpong.py --fresh          # train from scratch
        python realpong.py                  # resume realpong.pt
        python realpong.py --episodes 500   # stop after N

Importing this file (what the tournament does) just gives you `Agent` — the
training loop under `main()` does not run on import.

Model: Karpathy "Pong from pixels" — a policy net over the difference of two
80x80 frames. The env (arena.PongSym) is symmetric, so the policy plays either
side. Trained with REINFORCE + value baseline + entropy. Reward shaping adds a
small return bonus AND potential-based movement shaping (move toward the ball's
predicted intercept, re-centre when it recedes) to promote good positioning.

Opponent curriculum: a quick `random` warm-up, then an EASY ball-follower that
gets harder one small notch at a time. The bf's difficulty is a `skill` scalar
mapped to its reaction LAG: at skill 0.50 it chases a stale ball position (lag 24,
sluggish/easy); each +0.02 sharpens it (less lag); at skill 1.00 it tracks the live
ball (lag 0). skill rises each time the agent clears the win-rate gate; once it has
beaten the tracker down to skill SELFPLAY_AT_SKILL (a competent defender), it
graduates to SELF-PLAY vs a pool of frozen past selves.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arena import (PongSym, TrackerAgent, RandomAgent, UP, DOWN, D,
                   SIZE, PADDLE_H, PADDLE_X_L, PADDLE_X_R, BALL_SPEED_MAX)

HERE = Path(__file__).resolve().parent
SAVE = HERE / "realpong.pt"
BEST = HERE / "realpong_best.pt"     # best-ever model by the fixed-tracker yardstick

# Environment for rollouts AND the keep-best eval. Defaults to standard PongSym; main()
# swaps in arena_chaos.ChaosPong with --chaos, writing realpong_chaos.pt / realpong_chaos_best.pt
# so a chaos run never touches realpong.pt / realpong_best.pt.
ENV_CLASS = PongSym
# With --both, MIX_ENVS = [standard, chaos] and each training episode is drawn from them with
# P(chaos) = MIX_CHAOS_FRAC, so ONE generalist learns both arenas but PRIORITISES chaos.
MIX_ENVS = None
MIX_CHAOS_FRAC = 0.6        # share of training episodes (and eval weight) on chaos when --both


# ── THE MODEL (used by the tournament; trained by main() below) ───────────────
class Net(nn.Module):
    """Policy net: input = difference of two 80x80 frames (6400)."""
    def __init__(self, hidden=256):
        super().__init__()
        # input = [current frame (position) , current - previous (motion)] = 2*D
        self.fc1 = nn.Linear(2 * D, hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.policy_head(h)).squeeze(-1), self.value_head(h).squeeze(-1)


def features(cur, prev):
    """Network input: current frame (absolute position) + the motion (diff).
    Pure diff hides where a stationary paddle is; the current frame restores it."""
    diff = cur - prev if prev is not None else np.zeros(D, np.float32)
    return np.concatenate([cur, diff]).astype(np.float32)


class Agent:
    """Competition contract: reset() + act(80x80 frame, own paddle on RIGHT) -> 2|3."""
    def __init__(self, weights_path=None, stochastic=True, seed=0):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            try:
                self.net.load_state_dict(state)
            except RuntimeError:
                print(f"[warning] {weights_path} doesn't fit this model (architecture changed) "
                      f"-> using random weights. Retrain with: python realpong.py --fresh")
        self.net.eval()
        self.prev = None
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, frame):
        cur = frame.astype(np.float32).ravel()
        x = features(cur, self.prev)
        self.prev = cur
        prob, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        p = float(prob.item())
        up = self.rng.random() < p if self.stochastic else p > 0.5
        return UP if up else DOWN


def bf_lag(skill):
    """Map the difficulty scalar to the ball-follower's reaction LAG (the proven
    difficulty knob: it tracks where the ball WAS `lag` steps ago). skill=SKILL_MIN
    -> LAG_EASY (sluggish); skill=SKILL_MAX -> 0 (tracks the live ball). Higher skill
    = sharper = harder. (The curriculum STARTS at SKILL_START, not SKILL_MIN.)"""
    frac = (SKILL_MAX - skill) / (SKILL_MAX - SKILL_MIN)     # 1.0 at SKILL_MIN .. 0.0 at SKILL_MAX
    return max(0, int(round(LAG_EASY * frac)))


def skill_for_lag(lag):
    """Inverse of bf_lag: the skill that yields (about) this starting lag — used to start the
    external/ball-follower ladder at a chosen lag instead of the easy end."""
    frac = max(0.0, min(1.0, lag / LAG_EASY))
    return round(SKILL_MAX - frac * (SKILL_MAX - SKILL_MIN), 4)


# match-length curriculum tied to SKILL (not episodes), so the ramp to full 21-point
# games is GUARANTEED to happen before the self-play graduation:
#   random warm-up            -> 5-point  matches (fast, dense signal)
#   early tracker (skill<0.60)-> 10-point matches
#   tracker skill>=0.60       -> 21-point matches  (reached BEFORE self-play at skill 0.70)
#   self-play                 -> 21-point matches
POINTS_WARMUP, POINTS_MID, POINTS_FULL = 5, 10, 21
POINTS_21_AT_SKILL = 0.60     # full 21-pt games from this skill on (start is 0.70, so 21-pt throughout)

def match_points(mode, skill):
    if mode == "random":
        return POINTS_WARMUP
    if mode == "selfplay" or skill >= POINTS_21_AT_SKILL:
        return POINTS_FULL
    return POINTS_MID

batch_size    = 32         # episodes per optimizer step (raised 16->32: lower-variance REINFORCE
                           # gradients -> steadier climb, clears the gate more reliably)
learning_rate = 1e-3
gamma         = 0.99
lam           = 0.95       # GAE(lambda): bootstraps advantages with the value fn -> much lower variance
value_coef    = 0.5
entropy_coef  = 0.003      # lower -> lets the policy sharpen/commit (was 0.005; squeezes a bit more win rate)
grad_clip     = 1.0
hit_bonus     = 0.02       # base reward for a return (teaches defense without out-valuing a +1 point)
hit_speed     = 0.04       # EXTRA return bonus scaled by ball speed at contact: rewards returning the
                           # FAST balls (the ones it currently misses) more than slow ones
center_coef   = 0.03       # EXTRA return bonus for contact near the paddle CENTRE (well-positioned,
                           # safe catch) vs a risky edge catch; 0 at the edges, max at dead-centre
move_coef     = 0.08       # potential-based DEFENSE/MOVEMENT shaping: rewards moving toward the
                           # ball's predicted intercept (anticipation) and re-centering when it
                           # recedes. Potential-based => provably does NOT change the optimal
                           # policy. Prioritized over offense for now; still below the scoring signal.
offense_coef  = 0.01       # OFFENSE shaping (on our return only): bonus for placing the ball far
                           # from the opponent's paddle. Kept SMALL for now so DEFENSE is the
                           # priority — fires per-return, so a large coef would out-weigh the
                           # dense per-step defense signal. Raise it later to push offense.
window        = 50         # win rate & accuracy over the last 50 games (was 100 -> advances ~2x faster)
eval_every    = 100        # every N episodes, eval vs the FIXED lag-8 tracker and keep the best-ever model
eval_games    = 8          # fixed-seed games (both sides) -> deterministic yardstick; 8 halves the
                           # eval pause vs 16 while keeping the same per-game-averaged score scale

# ── ball-follower difficulty ladder ──────────────────────────────────────────────
# `skill` rises 0.50 -> 1.00 in +0.02 steps; bf_lag() maps it to the bf's reaction lag:
# skill 0.50 = lag LAG_EASY (sluggish, easy), skill 1.00 = lag 0 (tracks the live ball).
random_gate   = 0.80       # win 80% vs random before the ball-follower ladder begins
level_gate    = 0.85       # advance bf difficulty above 85% win (was 0.90 -- above the MLP's
                           # ceiling at these lags, so it stalled; 0.85 keeps the ladder moving)
SKILL_MIN     = 0.50       # bf-lag mapping floor: skill 0.50 -> lag LAG_EASY, skill 1.00 -> lag 0
SKILL_START   = 0.70       # the ball-follower ladder STARTS here (~lag 14) -- per request
SKILL_STEP    = 0.02       # each level sharpens the bf by one notch
SKILL_MAX     = 1.00       # hardest tracker: lag 0 (tracks the live ball)
LAG_EASY      = 24         # bf reaction lag at skill SKILL_MIN (very sluggish)
SELFPLAY_AT_SKILL = 0.86   # graduate bf -> SELF-PLAY on REACHING this skill (~lag 7). NOT accuracy-gated
                           # (accuracy caps ~50% here); skill is the reachable difficulty signal.
POOL_SIZE     = 5          # keep the last N frozen selves as self-play opponents


def discount(rewards):
    out = np.zeros_like(rewards, dtype=np.float64)
    run = 0.0
    for i in reversed(range(rewards.size)):
        if abs(rewards[i]) >= 0.5: run = 0.0   # reset at POINT boundaries (+-1), not hit bonuses
        run = run * gamma + rewards[i]
        out[i] = run
    return out


def gae(rewards, values):
    """Generalized Advantage Estimation. Bootstraps each step's advantage with the value
    function (lower variance than the raw Monte-Carlo return). Resets at POINT boundaries
    (|reward|>=0.5) -- each point is its own episode, exactly like discount()."""
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if abs(rewards[t]) >= 0.5 else 1.0      # point scored -> terminal
        next_v = values[t + 1] if (t + 1 < len(rewards)) else 0.0
        delta = rewards[t] + gamma * nonterminal * next_v - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def _ideal_y(env):
    """Where the RIGHT paddle's CENTRE ideally is. When the ball is approaching,
    that's its predicted vertical intercept at the paddle column (wall bounces
    folded in) — rewarding this teaches anticipation, not just chasing. When the
    ball is receding, the paddle's CURRENT centre — i.e. no pull anywhere, so the
    paddle just holds position instead of being magnetised back to the middle."""
    hi = SIZE - 1.0
    if env.bvx > 0 and env.bx < PADDLE_X_R:        # heading toward us
        steps = (PADDLE_X_R - env.bx) / env.bvx
        y = env.by + env.bvy * steps
        period = 2 * hi
        y = y % period
        return y if y <= hi else period - y        # fold top/bottom reflections
    return env.pad_r + PADDLE_H / 2.0              # receding -> hold position (flat potential, no centre pull)


def _potential(env):
    """Φ(s) in [-1, 0]: closer the paddle centre is to its ideal spot, higher Φ."""
    paddle_c = env.pad_r + PADDLE_H / 2.0
    return -abs(paddle_c - _ideal_y(env)) / SIZE


def _offense(env):
    """On OUR return, reward sending the ball FAR from the opponent: predict the ball's
    intercept at the LEFT (opponent) column from its post-hit velocity and measure the
    gap to the opponent's current paddle centre. Bigger gap = harder to reach = better
    placement (rewards using the paddle 'kick' to angle shots away). Returns [0, 1]."""
    if env.bvx >= 0:                              # not heading toward the opponent
        return 0.0
    hi = SIZE - 1.0
    steps = (env.bx - PADDLE_X_L) / (-env.bvx)
    y = env.by + env.bvy * steps
    period = 2 * hi
    y = y % period
    target = y if y <= hi else period - y         # fold wall reflections
    opp_c = env.pad_l + PADDLE_H / 2.0
    return min(1.0, abs(target - opp_c) / SIZE)


def play_episode(net, opponent, seed, points, env_cls=None):
    """Roll out one game with NO grad (fast, no autograd graph). env_cls picks the environment
    for this episode (defaults to ENV_CLASS). Stores per-step inputs and
    sampled actions; the gradient comes from ONE batched forward+backward in main() (much
    faster than building/backpropping a per-step graph -- the old bottleneck)."""
    env = (env_cls or ENV_CLASS)(seed=seed, points=points)
    obs = env.reset(seed=seed)
    opponent.reset()
    prev = None
    xs, ups, rewards = [], [], []      # network inputs, sampled action (True=UP), shaped reward
    hits = misses = 0                  # ball arrivals at OUR paddle: returned vs missed
    done = False
    while not done:
        cur = obs["right"].ravel().astype(np.float32)
        x = features(cur, prev)
        prev = cur
        with torch.no_grad():
            prob, _ = net(torch.from_numpy(x).unsqueeze(0))
        up = bool(torch.rand(()).item() < float(prob.item()))
        action = UP if up else DOWN
        xs.append(x); ups.append(up)
        phi_before = _potential(env)
        obs, rew, done, info = env.step(action, opponent.act(obs["left"]))
        # potential-based movement shaping: F = gamma*Phi(s') - Phi(s). Skipped on a
        # scoring step, where the env re-serves and the position jump would be spurious.
        move = 0.0 if rew["right"] != 0.0 else move_coef * (gamma * _potential(env) - phi_before)
        if info["hit_r"]:                              # we returned the ball
            ret_bonus = hit_bonus + hit_speed * abs(env.bvx) / BALL_SPEED_MAX   # harder (faster) return -> more
            ret_bonus += offense_coef * _offense(env)  # + placement away from the opponent
            offset = (env.by - env.pad_r) / PADDLE_H - 0.5         # 0 at paddle centre, +-0.5 at edges
            ret_bonus += center_coef * max(0.0, 1.0 - 2.0 * abs(offset))   # + more for centred contact
        else:
            ret_bonus = 0.0
        rewards.append(rew["right"] + move + ret_bonus)
        hits += info["hit_r"]; misses += info["miss_r"]
    return xs, ups, rewards, env.score_r, env.score_l, hits, misses


def atomic_save(obj, path):
    """Crash-safe save: write to a temp file, verify it loads, then atomically replace."""
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)   # verify before replacing
    os.replace(tmp, path)


@torch.no_grad()
def evaluate_vs_bf(net, n=eval_games, lag=8, points=21, env_cls=None):
    """Greedy net vs the FIXED lag-8 tracker (the arena's `bf`), both sides. A stable
    'how good is this model' yardstick for keep-best. Returns (win_rate, conceded/game)."""
    rng = np.random.default_rng(777)
    cls = env_cls or ENV_CLASS
    wins = conceded = 0
    for g in range(n):
        env = cls(seed=int(rng.integers(1 << 30)), points=points)
        ob = env.reset(seed=int(rng.integers(1 << 30)))
        opp = TrackerAgent(lag=lag, seed=int(rng.integers(1 << 30))); opp.reset()
        prev = None
        net_right = (g % 2 == 0)                              # play both sides
        done = False
        while not done:
            frame = ob["right"] if net_right else ob["left"]
            cur = frame.ravel().astype(np.float32)
            x = features(cur, prev); prev = cur
            prob, _ = net(torch.from_numpy(x).unsqueeze(0))
            a = UP if float(prob.item()) > 0.5 else DOWN      # greedy
            if net_right: ob, _, done, _ = env.step(a, opp.act(ob["left"]))
            else:         ob, _, done, _ = env.step(opp.act(ob["right"]), a)
        my, their = (env.score_r, env.score_l) if net_right else (env.score_l, env.score_r)
        wins += int(my > their); conceded += their
    return wins / n, conceded / n


class LaggedAgent:
    """Wrap an external Agent so it reacts to a frame from `lag` steps ago — sluggish,
    i.e. handicapped/easy (same idea as TrackerAgent's reaction lag). lag 0 = full strength.
    The wrapped agent's own motion channel still works: it just sees a delayed frame stream."""
    def __init__(self, agent, lag):
        self.agent = agent
        self.buf = deque(maxlen=lag + 1)         # last lag+1 frames; buf[0] is `lag` steps old once full
    def reset(self):
        self.agent.reset()
        self.buf.clear()
    def act(self, frame):
        self.buf.append(frame)
        return self.agent.act(self.buf[0])       # feed the oldest buffered (stale) frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="stop after N (0 = until Ctrl-C)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved weights")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--reset-curriculum", action="store_true",
                    help="keep weights but reset mode->random and episode->0 (unstick a stalled run)")
    ap.add_argument("--opponent-file", default=None,
                    help="train vs a FIXED external opponent, e.g. pong.py:pong_best.pt "
                         "(loads <file>.Agent(<weights>); skips tracker/selfplay curriculum)")
    ap.add_argument("--selfplay", action="store_true",
                    help="force SELF-PLAY now: spar a pool of frozen past selves (skips the rest "
                         "of the tracker ladder); keeps current weights")
    ap.add_argument("--ball-follower", action="store_true",
                    help="train vs the ball-follower (tracker) at INCREASING difficulty (lag 24->0), "
                         "21-pt games, NO self-play graduation; keeps current weights and realpong_best")
    ap.add_argument("--start-lag", type=int, default=None,
                    help="for --opponent-file: start the opponent's lag ladder at THIS lag "
                         "(e.g. 5) instead of the easy end; it still climbs toward lag 0")
    ap.add_argument("--chaos", action="store_true",
                    help="train on the HARDER arena_chaos.ChaosPong env (random ball speed). Saves to "
                         "realpong_chaos.pt / realpong_chaos_best.pt and warm-starts from realpong.pt, "
                         "so realpong.pt / realpong_best.pt stay untouched")
    ap.add_argument("--both", action="store_true",
                    help="train ONE generalist on BOTH envs (chaos-weighted, see --chaos-frac). Warm-starts "
                         "from the AVERAGE of realpong_best.pt and realpong_chaos_best.pt; saves to "
                         "realpong_both.pt / realpong_both_best.pt. The source bests stay untouched")
    ap.add_argument("--chaos-frac", type=float, default=0.6,
                    help="with --both: fraction of episodes (and eval weight) on the chaos env (default 0.6)")
    args = ap.parse_args()

    # ── env + save-path selection (each mode writes its OWN files -> no collision) ──
    global ENV_CLASS, MIX_ENVS, MIX_CHAOS_FRAC, SAVE, BEST
    if args.both:
        from arena_chaos import ChaosPong
        MIX_ENVS = [PongSym, ChaosPong]         # [0]=standard, [1]=chaos
        MIX_CHAOS_FRAC = args.chaos_frac
        if SAVE == HERE / "realpong.pt":        # only redirect the DEFAULT paths (tests can override)
            SAVE = HERE / "realpong_both.pt"
        if BEST == HERE / "realpong_best.pt":
            BEST = HERE / "realpong_both_best.pt"
        print(f"*** GENERALIST: {int((1-MIX_CHAOS_FRAC)*100)}/{int(MIX_CHAOS_FRAC*100)} standard/chaos "
              f"-> {SAVE.name} / {BEST.name} ***")
    elif args.chaos:
        from arena_chaos import ChaosPong
        ENV_CLASS = ChaosPong
        if SAVE == HERE / "realpong.pt":        # only redirect the DEFAULT paths (tests can override)
            SAVE = HERE / "realpong_chaos.pt"
        if BEST == HERE / "realpong_best.pt":
            BEST = HERE / "realpong_chaos_best.pt"
        print(f"*** CHAOS env (random ball speed) -> {SAVE.name} / {BEST.name} ***")

    # ── optional: external fixed opponent (e.g. pong_best) ─────────────────────
    external_opp = None
    if args.opponent_file:
        spec_str = args.opponent_file
        fpath, _, wpath = spec_str.partition(":")
        if not os.path.exists(fpath): fpath = str(HERE / fpath)
        if wpath and not os.path.exists(wpath): wpath = str(HERE / wpath)
        spec = importlib.util.spec_from_file_location("_opp_mod", fpath)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        external_opp = mod.Agent(wpath) if wpath else mod.Agent()
        print(f"opponent: external {fpath}:{wpath or '(no weights)'}")

    if args.reset_curriculum and SAVE.exists():
        ck = torch.load(SAVE, map_location="cpu", weights_only=False)
        if isinstance(ck, dict):
            ck["mode"] = "random"
            ck["episode"] = 0
            ck["skill"] = SKILL_START
            torch.save(ck, SAVE)
            print("reset curriculum (weights preserved) -> random warm-up, then bf ladder from 0.50")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    episode = 0
    mode = "random"
    skill = SKILL_START
    best_score = float("-inf")  # best yardstick score so far (for keep-best)
    restore_window = None      # rolling win/hit window carried over from the checkpoint
    if SAVE.exists() and not args.fresh:
        ck = torch.load(SAVE, map_location="cpu", weights_only=False)
        try:
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            if isinstance(ck, dict) and "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
            episode = int(ck.get("episode", 0)) if isinstance(ck, dict) else 0
            if isinstance(ck, dict): best_score = ck.get("best_score", float("-inf"))
            if isinstance(ck, dict) and "skill" in ck:
                # already on the bf ladder -> restore it exactly
                mode = ck.get("mode", "tracker")
                skill = float(ck["skill"])
                if mode not in ("random", "tracker", "selfplay", "external"):   # unknown -> safe default
                    mode, skill = "tracker", SKILL_START
                restore_window = ck                     # carry the 100-game window across restarts
                print(f"resumed realpong.pt at episode {episode} (opponent: {mode}, bf skill {skill:.2f})")
            else:
                # legacy checkpoint (pre-ladder): KEEP the trained weights but start the
                # ball-follower ladder fresh at the easy end (0.50), and continue training.
                mode, skill = "tracker", SKILL_START
                print(f"resumed realpong.pt at episode {episode}; weights kept, "
                      f"starting easy ball-follower ladder at skill {skill:.2f}")
        except (RuntimeError, ValueError):
            print("existing realpong.pt is from the OLD model architecture -> starting fresh")
    elif args.both and (HERE / "realpong_best.pt").exists() and (HERE / "realpong_chaos_best.pt").exists():
        # FIRST generalist run: warm-start from the AVERAGE of the two specialist bests (a
        # "model soup" of the standard-best and chaos-best), then adapt on the 50/50 mix.
        def _state(p):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            return ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        sa = _state(HERE / "realpong_best.pt"); sb = _state(HERE / "realpong_chaos_best.pt")
        try:
            avg = {k: (sa[k] + sb[k]) / 2.0 for k in sa}
            net.load_state_dict(avg)
            print("GENERALIST warm-start: averaged realpong_best.pt + realpong_chaos_best.pt "
                  "(read-only); curriculum starts fresh")
        except (RuntimeError, ValueError, KeyError):
            print("could not average the two bests (mismatch) -> fresh")
    elif args.chaos and (HERE / "realpong.pt").exists():
        # FIRST chaos run: warm-start weights from the trained standard model (read-only);
        # curriculum/best start fresh because the env (and so the yardstick) is different.
        ck = torch.load(HERE / "realpong.pt", map_location="cpu", weights_only=False)
        try:
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            print("CHAOS warm-start: copied weights from realpong.pt (read-only); curriculum starts fresh")
        except (RuntimeError, ValueError):
            print("could not warm-start from realpong.pt (mismatch) -> fresh")
    else:
        print("fresh realpong (symmetric env)")

    # external-opponent lag ladder: spar the external agent (e.g. pong_best), handicapped by
    # reaction lag that we progressively REMOVE (lag 24 -> 0) as we clear the win gate. Keeps
    # realpong's trained weights; lag 0 == full-strength external opponent.
    if external_opp is not None:
        if args.start_lag is not None:       # explicit starting lag (e.g. --start-lag 5) -> overrides
            mode, skill = "external", skill_for_lag(args.start_lag)
            restore_window = None
            print(f">>> external lag ladder starts at lag {bf_lag(skill)} "
                  f"(skill {skill:.3f}) -> climbs toward lag 0")
        elif mode == "external":
            print(f">>> external lag ladder resumed at skill {skill:.2f} (lag {bf_lag(skill)})")
        else:                                # switching INTO external from another curriculum
            mode, skill = "external", SKILL_MIN
            restore_window = None            # old window was vs a different opponent -> start fresh
            print(f">>> training vs external opponent — lag ladder starts EASY "
                  f"at skill {skill:.2f} (lag {bf_lag(skill)})")

    recent = deque(maxlen=window)
    recent_wins = deque(maxlen=window)
    recent_hits = deque(maxlen=window)
    recent_misses = deque(maxlen=window)
    if restore_window is not None:     # don't reset progress toward the next level on restart
        recent.extend(restore_window.get("recent", [])[-window:])
        recent_wins.extend(restore_window.get("recent_wins", [])[-window:])
        recent_hits.extend(restore_window.get("recent_hits", [])[-window:])
        recent_misses.extend(restore_window.get("recent_misses", [])[-window:])
        if recent_wins:
            print(f"    carried over {len(recent_wins)}/{window}-game window "
                  f"(win {np.mean(recent_wins)*100:.1f}%) — keeps progress toward the next level")
    opt.zero_grad()
    start = episode
    if args.ball_follower:                         # force the ball-follower ladder (no self-play)
        mode, skill = "tracker", 0.86              # start here (lag 7) and climb to lag 0
        recent.clear(); recent_wins.clear(); recent_hits.clear(); recent_misses.clear()
        print(f">>> ball-follower ladder: tracker from skill {skill:.2f} (lag {bf_lag(skill)}) "
              f"-> climbs to lag 0, NO self-play, 21-pt games")
    elif args.selfplay:                            # forced self-play (overrides curriculum / external)
        mode, skill = "selfplay", max(skill, SELFPLAY_AT_SKILL)
        print(">>> forced SELF-PLAY: sparring a pool of frozen past selves")
    # resumed already AT/past the graduation skill (e.g. stuck at the top of the ladder) -> self-play now
    elif mode == "tracker" and skill >= SELFPLAY_AT_SKILL - 1e-9:
        mode = "selfplay"
        print(f">>> resumed at skill {skill:.2f} (>= {SELFPLAY_AT_SKILL}) -> SELF-PLAY now")
    print(f"training on {ENV_CLASS.__name__} (symmetric). opponent: {mode}. Ctrl-C to stop.")

    # self-play: a reusable opponent whose weights we swap to a frozen past snapshot of self
    opp_agent = Agent(stochastic=True, seed=12345)
    pool = []                                   # frozen snapshots (state_dicts)
    if mode == "selfplay":                      # resumed straight into self-play -> seed the pool
        pool.append({k: v.detach().clone() for k, v in net.state_dict().items()})

    def snapshot():
        return {k: v.detach().clone() for k, v in net.state_dict().items()}

    def save():
        atomic_save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                     "episode": episode, "mode": mode, "skill": skill, "best_score": best_score,
                     "recent": list(recent), "recent_wins": list(recent_wins),
                     "recent_hits": list(recent_hits), "recent_misses": list(recent_misses)}, SAVE)

    try:
        while args.episodes == 0 or episode - start < args.episodes:
            if mode == "external":              # external sparring partner, handicapped by lag
                opponent = LaggedAgent(external_opp, bf_lag(skill))   # play_episode calls reset()
                points = POINTS_FULL
            else:
                points = match_points(mode, skill)
                if mode == "random":
                    opponent = RandomAgent(int(rng.integers(1 << 30)))
                elif mode == "selfplay":         # opponent = a frozen past snapshot of self
                    opp_agent.net.load_state_dict(pool[int(rng.integers(len(pool)))])
                    opponent = opp_agent
                else:                            # tracker: easy(laggy) -> hard(lag 0) ball-follower
                    opponent = TrackerAgent(lag=bf_lag(skill), seed=int(rng.integers(1 << 30)))
            # generalist: draw this episode's env chaos-weighted from the mix (standard | chaos)
            ep_env = (MIX_ENVS[1] if rng.random() < MIX_CHAOS_FRAC else MIX_ENVS[0]) if MIX_ENVS else None
            t0 = time.perf_counter()
            xs, ups, rewards, sr, sl, hits, misses = play_episode(
                net, opponent, int(rng.integers(1 << 30)), points, env_cls=ep_env)
            t_play = time.perf_counter() - t0

            t0 = time.perf_counter()
            # ONE batched forward+backward over the whole episode (was per-step -> ~10x cheaper).
            X = torch.from_numpy(np.stack(xs))                           # (T, 2*D)
            up_t = torch.tensor(ups)
            probs, values_t = net(X)                                     # batched forward, with grad
            adv_np = gae(np.array(rewards), values_t.detach().numpy())   # GAE advantages (low-variance)
            adv = torch.tensor(adv_np, dtype=torch.float32)
            returns = adv + values_t.detach()                            # value target = GAE return
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            logp = torch.log(torch.where(up_t, probs, 1 - probs) + 1e-8)
            policy_loss = -(logp * adv).sum()
            value_loss = value_coef * (values_t - returns).pow(2).mean()
            p = probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
            (policy_loss + value_loss - entropy_coef * entropy).backward()

            episode += 1
            if episode % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step(); opt.zero_grad()
            t_upd = time.perf_counter() - t0

            reward_sum = float(sr - sl)        # point margin (shaping bonuses excluded from display)
            recent.append(reward_sum)
            recent_wins.append(1 if sr > sl else 0)
            recent_hits.append(hits); recent_misses.append(misses)
            avg = float(np.mean(recent))
            winrate = float(np.mean(recent_wins))
            arrivals = sum(recent_hits) + sum(recent_misses)
            accuracy = (sum(recent_hits) / arrivals) if arrivals else 0.0   # return rate
            if mode == "external": label = f"external(skill={skill:.2f},lag={bf_lag(skill)})"
            else: label = f"bf(skill={skill:.2f},lag={bf_lag(skill)})" if mode == "tracker" else mode
            env_tag = f" | env {ep_env.__name__}" if MIX_ENVS else ""
            print(f"ep {episode:5d} | {points:2d}pt | {sr}-{sl} | reward {reward_sum:+3.0f} "
                  f"| win {len(recent_wins):3d}/{window} {winrate*100:5.1f}% | acc {accuracy*100:4.0f}% | opp {label}{env_tag} "
                  f"| {len(rewards):5d}st play {t_play:5.2f}s upd {t_upd*1000:4.0f}ms")

            gate = random_gate if mode == "random" else level_gate
            advanced = False
            if len(recent_wins) == window and winrate >= gate:
                if mode == "external":                       # sharpen the external opponent one notch
                    if skill < SKILL_MAX - 1e-9:
                        skill = round(min(skill + SKILL_STEP, SKILL_MAX), 2)
                        advanced = True
                        print(f">>> difficulty up: external lag -> {bf_lag(skill)} "
                              f"(skill {skill:.2f}, >=85% win rate)")
                    else:
                        print(">>> at full-strength external opponent (lag 0) — holding here")
                elif mode == "random":
                    mode, skill = "tracker", SKILL_START; advanced = True
                    print(f">>> warm-up cleared -> easy ball-follower ladder begins at skill {skill:.2f}")
                elif mode == "tracker" and args.ball_follower:   # ball-follower ladder: climb to lag 0, no self-play
                    if skill < SKILL_MAX - 1e-9:
                        skill = round(min(skill + SKILL_STEP, SKILL_MAX), 2); advanced = True
                        print(f">>> difficulty up: bf skill -> {skill:.2f} (lag {bf_lag(skill)}, >=85% win)")
                    else:
                        print(">>> at hardest ball-follower (lag 0) — holding here")
                elif mode == "tracker":                      # cleared the gate -> sharpen one notch
                    skill = round(skill + SKILL_STEP, 2); advanced = True
                    if skill >= SELFPLAY_AT_SKILL - 1e-9:    # REACHED the graduation skill -> spar self
                        skill = min(skill, SKILL_MAX)
                        mode = "selfplay"; pool.append(snapshot())
                        print(f">>> reached bf skill {skill:.2f} (lag {bf_lag(skill)}) -> SELF-PLAY (pool 1)")
                    else:
                        print(f">>> difficulty up: bf skill -> {skill:.2f} (>=85% win rate)")
                elif mode == "selfplay":                     # beating the pool -> add a stronger self
                    pool.append(snapshot()); pool[:] = pool[-POOL_SIZE:]; advanced = True
                    print(f">>> self-play: added a stronger snapshot (pool {len(pool)})")
                if advanced:
                    recent.clear(); recent_wins.clear()
                    recent_hits.clear(); recent_misses.clear()

            if episode % args.save_every == 0:
                save()

            if episode % eval_every == 0:                    # keep the BEST-ever model (fixed yardstick)
                t0 = time.perf_counter()
                if MIX_ENVS:                                 # generalist: good at BOTH, chaos-weighted
                    ws, cs = evaluate_vs_bf(net, env_cls=MIX_ENVS[0])    # standard
                    wc, cc = evaluate_vs_bf(net, env_cls=MIX_ENVS[1])    # chaos
                    w = MIX_CHAOS_FRAC
                    score = (1 - w) * (ws - 0.01 * cs) + w * (wc - 0.01 * cc)   # chaos-weighted combined
                    detail = f"std win {ws*100:.0f}% | chaos win {wc*100:.0f}%"
                else:
                    wr, conc = evaluate_vs_bf(net)
                    score = wr - 0.01 * conc                 # win rate, tie-broken by fewer points conceded
                    detail = f"win {wr*100:.0f}% | conceded {conc:.1f}"
                tag = ""
                if score > best_score:
                    best_score = score
                    atomic_save({"model": net.state_dict()}, BEST)
                    tag = f" -> NEW BEST (saved {BEST.name})"
                print(f"   [eval {time.perf_counter()-t0:.1f}s] vs bf-lag8: {detail}{tag}")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save()
        print(f"saved {SAVE}")


if __name__ == "__main__":
    main()
