"""realpong.py  --  train the realpong model on the symmetric Pong env (arena.PongSym).

REINFORCE (policy gradient) with a value baseline and an entropy bonus, on the
difference of two 80x80 frames (Karpathy "Pong from pixels"). Because the env is
symmetric, the agent trains as the RIGHT player and the learned policy plays
either side at eval.

Curriculum: start vs a RANDOM opponent (easy, gives signal), then graduate to
the scripted ball-tracker once the model consistently wins.

    python realpong.py                 # resume training realpong.pt
    python realpong.py --fresh         # start from scratch (use this: old weights
                                       #   were trained on a different env)
    python realpong.py --episodes 500  # stop after N episodes
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from arena import PongSym, Net, TrackerAgent, RandomAgent, UP, DOWN, D

HERE = Path(__file__).resolve().parent
SAVE = HERE / "realpong.pt"

# match-length curriculum: short games first (fast, dense signal), then longer,
# ending on full 21-point official matches.
#   episodes  <1000          -> 5-point matches
#   1000..4999               -> 10-point matches
#   >=5000                   -> 21-point official matches
PHASE1_END, PHASE2_END = 1000, 5000
POINTS_P1, POINTS_P2, POINTS_OFFICIAL = 5, 10, 21

def match_points(ep):
    if ep < PHASE1_END: return POINTS_P1
    if ep < PHASE2_END: return POINTS_P2
    return POINTS_OFFICIAL

batch_size    = 16         # episodes accumulated per optimizer step
learning_rate = 1e-3
gamma         = 0.99
value_coef    = 0.5
entropy_coef  = 0.01
grad_clip     = 1.0
graduate_winrate = 0.98    # must win 98% vs the current opponent before graduating
window        = 50


def discount(rewards):
    out = np.zeros_like(rewards, dtype=np.float64)
    run = 0.0
    for i in reversed(range(rewards.size)):
        if rewards[i] != 0: run = 0.0
        run = run * gamma + rewards[i]
        out[i] = run
    return out


def play_episode(net, opponent, seed, points):
    env = PongSym(seed=seed, points=points)
    obs = env.reset(seed=seed)
    prev = None
    logps, values, probs, rewards = [], [], [], []
    hits = misses = 0          # ball arrivals at OUR paddle: returned vs missed
    done = False
    while not done:
        cur = obs["right"].ravel()
        diff = cur - prev if prev is not None else np.zeros(D, np.float32)
        prev = cur
        prob, value = net(torch.from_numpy(diff).unsqueeze(0))
        prob = prob.squeeze(0); value = value.squeeze(0)
        up = torch.rand(()) < prob
        action = UP if up.item() else DOWN
        logps.append(torch.log((prob if up else 1 - prob) + 1e-8))
        values.append(value); probs.append(prob)
        obs, rew, done, info = env.step(action, opponent.act(obs["left"]))
        rewards.append(rew["right"])
        hits += info["hit_r"]; misses += info["miss_r"]
    return logps, values, probs, rewards, env.score_r, env.score_l, hits, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="stop after N (0 = until Ctrl-C)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved weights")
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    episode = 0
    mode = "random"
    if SAVE.exists() and not args.fresh:
        ck = torch.load(SAVE, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        if isinstance(ck, dict) and "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        episode = int(ck.get("episode", 0)) if isinstance(ck, dict) else 0
        mode = ck.get("mode", "random") if isinstance(ck, dict) else "random"
        print(f"resumed realpong.pt at episode {episode} (opponent: {mode})")
    else:
        print("fresh realpong (symmetric env)")

    recent = deque(maxlen=window)
    recent_wins = deque(maxlen=window)
    recent_hits = deque(maxlen=window)
    recent_misses = deque(maxlen=window)
    opt.zero_grad()
    start = episode
    print(f"training on PongSym (symmetric). opponent: {mode}. Ctrl-C to stop.")

    def save():
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "episode": episode, "mode": mode}, SAVE)

    try:
        while args.episodes == 0 or episode - start < args.episodes:
            points = match_points(episode)
            opponent = RandomAgent(int(rng.integers(1 << 30))) if mode == "random" else TrackerAgent()
            logps, values, probs, rewards, sr, sl, hits, misses = play_episode(
                net, opponent, int(rng.integers(1 << 30)), points)

            returns = torch.tensor(discount(np.array(rewards)), dtype=torch.float32)
            values_t = torch.stack(values)
            adv = returns - values_t.detach()
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

            reward_sum = float(sum(rewards))
            recent.append(reward_sum)
            recent_wins.append(1 if sr > sl else 0)
            recent_hits.append(hits); recent_misses.append(misses)
            avg = float(np.mean(recent))
            winrate = float(np.mean(recent_wins))
            arrivals = sum(recent_hits) + sum(recent_misses)
            accuracy = (sum(recent_hits) / arrivals) if arrivals else 0.0   # return rate
            print(f"ep {episode:5d} | {points:2d}pt | {sr}-{sl} | reward {reward_sum:+3.0f} "
                  f"| win {winrate*100:4.0f}% | acc {accuracy*100:4.0f}% | opp {mode}")

            if mode == "random" and len(recent_wins) == window and winrate >= graduate_winrate:
                mode = "tracker"; recent.clear(); recent_wins.clear()
                recent_hits.clear(); recent_misses.clear()
                print(">>> graduated: now training vs the ball-tracker")

            if episode % args.save_every == 0:
                save()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save()
        print(f"saved {SAVE}")


if __name__ == "__main__":
    main()
