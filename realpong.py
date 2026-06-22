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
ball (lag 0). skill rises by +0.02 each time the agent wins >=90% of the last 100
games, and then STAYS at the top. The opponent is always the ball-follower — there
is no self-play stage.
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arena import (PongSym, TrackerAgent, RandomAgent, UP, DOWN, D,
                   SIZE, PADDLE_H, PADDLE_X_L, PADDLE_X_R, BALL_SPEED_MAX)

HERE = Path(__file__).resolve().parent
SAVE = HERE / "realpong.pt"


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
    difficulty knob: it tracks where the ball WAS `lag` steps ago). Easiest level
    (skill=SKILL_START) -> LAG_EASY (sluggish, clearly beatable); hardest
    (skill=SKILL_MAX) -> 0 (tracks the live ball). Higher skill = sharper = harder."""
    frac = (SKILL_MAX - skill) / (SKILL_MAX - SKILL_START)   # 1.0 easiest .. 0.0 hardest
    return max(0, int(round(LAG_EASY * frac)))


# match-length curriculum: short games first (fast, dense signal), then longer,
# ending on full 21-point official matches.
#   episodes  <1000          -> 5-point matches
#   1000..4999               -> 10-point matches
#   >=5000                   -> 21-point official matches
PHASE1_END, PHASE2_END = 1000, 5000
POINTS_P1, POINTS_P2, POINTS_OFFICIAL = 5, 10, 11   # train on 11-pt games (was 21): ~2x faster games,
                                                    # same win-rate signal; the agent still plays 21 in the arena

def match_points(ep):
    if ep < PHASE1_END: return POINTS_P1
    if ep < PHASE2_END: return POINTS_P2
    return POINTS_OFFICIAL

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
move_coef     = 0.08       # potential-based DEFENSE/MOVEMENT shaping: rewards moving toward the
                           # ball's predicted intercept (anticipation) and re-centering when it
                           # recedes. Potential-based => provably does NOT change the optimal
                           # policy. Prioritized over offense for now; still below the scoring signal.
offense_coef  = 0.01       # OFFENSE shaping (on our return only): bonus for placing the ball far
                           # from the opponent's paddle. Kept SMALL for now so DEFENSE is the
                           # priority — fires per-return, so a large coef would out-weigh the
                           # dense per-step defense signal. Raise it later to push offense.
window        = 50         # win rate & accuracy over the last 50 games (was 100 -> advances ~2x faster)

# ── ball-follower difficulty ladder ──────────────────────────────────────────────
# `skill` rises 0.50 -> 1.00 in +0.02 steps; bf_lag() maps it to the bf's reaction lag:
# skill 0.50 = lag LAG_EASY (sluggish, easy), skill 1.00 = lag 0 (tracks the live ball).
random_gate   = 0.80       # win 80% vs random before the ball-follower ladder begins
level_gate    = 0.85       # advance bf difficulty above 85% win (was 0.90 -- above the MLP's
                           # ceiling at these lags, so it stalled; 0.85 keeps the ladder moving)
SKILL_START   = 0.50       # easiest level
SKILL_STEP    = 0.02       # each level sharpens the bf by one notch
SKILL_MAX     = 1.00       # hardest level: lag 0 -> then stay here (no self-play)
LAG_EASY      = 24         # bf reaction lag at the easiest level (very sluggish, >90% beatable)


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
    ball is receding, the centre (ready position) — rewarding this teaches the
    paddle to reset for the next rally instead of drifting."""
    hi = SIZE - 1.0
    if env.bvx > 0 and env.bx < PADDLE_X_R:        # heading toward us
        steps = (PADDLE_X_R - env.bx) / env.bvx
        y = env.by + env.bvy * steps
        period = 2 * hi
        y = y % period
        return y if y <= hi else period - y        # fold top/bottom reflections
    return hi / 2.0                                # receding -> ready position (centre)


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


def play_episode(net, opponent, seed, points):
    env = PongSym(seed=seed, points=points)
    obs = env.reset(seed=seed)
    opponent.reset()
    prev = None
    logps, values, probs, rewards = [], [], [], []
    hits = misses = 0          # ball arrivals at OUR paddle: returned vs missed
    done = False
    while not done:
        cur = obs["right"].ravel().astype(np.float32)
        x = features(cur, prev)
        prev = cur
        prob, value = net(torch.from_numpy(x).unsqueeze(0))
        prob = prob.squeeze(0); value = value.squeeze(0)
        up = torch.rand(()) < prob
        action = UP if up.item() else DOWN
        logps.append(torch.log((prob if up else 1 - prob) + 1e-8))
        values.append(value); probs.append(prob)
        phi_before = _potential(env)
        obs, rew, done, info = env.step(action, opponent.act(obs["left"]))
        # potential-based movement shaping: F = gamma*Phi(s') - Phi(s). Skipped on a
        # scoring step, where the env re-serves and the position jump would be spurious.
        move = 0.0 if rew["right"] != 0.0 else move_coef * (gamma * _potential(env) - phi_before)
        if info["hit_r"]:                              # we returned the ball
            ret_bonus = hit_bonus + hit_speed * abs(env.bvx) / BALL_SPEED_MAX   # harder (faster) return -> more
            ret_bonus += offense_coef * _offense(env)  # + placement away from the opponent
        else:
            ret_bonus = 0.0
        rewards.append(rew["right"] + move + ret_bonus)
        hits += info["hit_r"]; misses += info["miss_r"]
    return logps, values, probs, rewards, env.score_r, env.score_l, hits, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="stop after N (0 = until Ctrl-C)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved weights")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--reset-curriculum", action="store_true",
                    help="keep weights but reset mode->random and episode->0 (unstick a stalled run)")
    args = ap.parse_args()

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
    restore_window = None      # rolling win/hit window carried over from the checkpoint
    if SAVE.exists() and not args.fresh:
        ck = torch.load(SAVE, map_location="cpu", weights_only=False)
        try:
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            if isinstance(ck, dict) and "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
            episode = int(ck.get("episode", 0)) if isinstance(ck, dict) else 0
            if isinstance(ck, dict) and "skill" in ck:
                # already on the bf ladder -> restore it exactly
                mode = ck.get("mode", "tracker")
                skill = float(ck["skill"])
                if mode not in ("random", "tracker"):   # legacy self-play state -> hardest bf
                    mode, skill = "tracker", SKILL_MAX
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
    else:
        print("fresh realpong (symmetric env)")

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
    print(f"training on PongSym (symmetric). opponent: {mode}. Ctrl-C to stop.")

    def save():
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "episode": episode, "mode": mode, "skill": skill,
                    "recent": list(recent), "recent_wins": list(recent_wins),
                    "recent_hits": list(recent_hits), "recent_misses": list(recent_misses)}, SAVE)

    try:
        while args.episodes == 0 or episode - start < args.episodes:
            points = match_points(episode)
            if mode == "random":
                opponent = RandomAgent(int(rng.integers(1 << 30)))
            else:                                # tracker: easy(laggy) -> hard(lag 0) ball-follower
                opponent = TrackerAgent(lag=bf_lag(skill), seed=int(rng.integers(1 << 30)))
            logps, values, probs, rewards, sr, sl, hits, misses = play_episode(
                net, opponent, int(rng.integers(1 << 30)), points)

            values_t = torch.stack(values)
            adv_np = gae(np.array(rewards), values_t.detach().numpy())   # GAE advantages (low-variance)
            adv = torch.tensor(adv_np, dtype=torch.float32)
            returns = adv + values_t.detach()                            # value target = GAE return
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(torch.stack(logps) * adv).sum()
            value_loss = value_coef * (values_t - returns).pow(2).mean()
            p = torch.stack(probs).clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
            (policy_loss + value_loss - entropy_coef * entropy).backward()

            episode += 1
            if episode % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step(); opt.zero_grad()

            reward_sum = float(sr - sl)        # point margin (shaping bonuses excluded from display)
            recent.append(reward_sum)
            recent_wins.append(1 if sr > sl else 0)
            recent_hits.append(hits); recent_misses.append(misses)
            avg = float(np.mean(recent))
            winrate = float(np.mean(recent_wins))
            arrivals = sum(recent_hits) + sum(recent_misses)
            accuracy = (sum(recent_hits) / arrivals) if arrivals else 0.0   # return rate
            label = f"bf(skill={skill:.2f},lag={bf_lag(skill)})" if mode == "tracker" else mode
            print(f"ep {episode:5d} | {points:2d}pt | {sr}-{sl} | reward {reward_sum:+3.0f} "
                  f"| win {len(recent_wins):3d}/{window} {winrate*100:5.1f}% | acc {accuracy*100:4.0f}% | opp {label}")

            gate = random_gate if mode == "random" else level_gate
            if len(recent_wins) == window and winrate >= gate:
                if mode == "random":
                    mode, skill = "tracker", SKILL_START
                    print(f">>> warm-up cleared -> easy ball-follower ladder begins at skill {skill:.2f}")
                elif skill < SKILL_MAX - 1e-9:
                    skill = min(SKILL_MAX, round(skill + SKILL_STEP, 2))
                    print(f">>> difficulty up: bf skill -> {skill:.2f} (>=90% win rate)")
                else:
                    print(">>> at hardest ball-follower (skill 1.00) -> staying here (no self-play)")
                recent.clear(); recent_wins.clear()
                recent_hits.clear(); recent_misses.clear()

            if episode % args.save_every == 0:
                save()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save()
        print(f"saved {SAVE}")


if __name__ == "__main__":
    main()
