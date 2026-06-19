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

TRAIN_POINTS  = 5          # short games while training -> fast episodes
batch_size    = 16         # episodes accumulated per optimizer step
learning_rate = 1e-3
gamma         = 0.99
value_coef    = 0.5
entropy_coef  = 0.01
grad_clip     = 1.0
graduate_at   = 1.0        # avg reward (over the window) to move random -> tracker
window        = 50


def discount(rewards):
    out = np.zeros_like(rewards, dtype=np.float64)
    run = 0.0
    for i in reversed(range(rewards.size)):
        if rewards[i] != 0: run = 0.0
        run = run * gamma + rewards[i]
        out[i] = run
    return out


def play_episode(net, opponent, seed):
    env = PongSym(seed=seed, points=TRAIN_POINTS)
    obs = env.reset(seed=seed)
    prev = None
    logps, values, probs, rewards = [], [], [], []
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
        obs, rew, done = env.step(action, opponent.act(obs["left"]))
        rewards.append(rew["right"])
    return logps, values, probs, rewards, env.score_r, env.score_l


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
    opt.zero_grad()
    start = episode
    print(f"training on PongSym (symmetric). opponent: {mode}. Ctrl-C to stop.")

    def save():
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "episode": episode, "mode": mode}, SAVE)

    try:
        while args.episodes == 0 or episode - start < args.episodes:
            opponent = RandomAgent(int(rng.integers(1 << 30))) if mode == "random" else TrackerAgent()
            logps, values, probs, rewards, sr, sl = play_episode(net, opponent, int(rng.integers(1 << 30)))

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
            avg = float(np.mean(recent))
            print(f"ep {episode:5d} | {sr}-{sl} | reward {reward_sum:+3.0f} | avg {avg:+5.2f} | opp {mode}")

            if mode == "random" and len(recent) == window and avg >= graduate_at:
                mode = "tracker"; recent.clear()
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
