"""Train realpong with a stronger both-side curriculum on pong_v3."""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
from pettingzoo.atari import pong_v3

from agent_ale import (
    D,
    DEVICE,
    DOWN,
    NOOP,
    UP,
    PolicyNet,
    ball_follower_action,
    preprocess,
)


HERE = Path(__file__).resolve().parent
SAVE = HERE / "realpong.pt"
SEED = 1

POINTS_TO_WIN = 21
MAX_CYCLES = 8000
RANDOM_ACTIONS = (UP, DOWN, NOOP)

batch_size = 10
learning_rate = 1e-3
gamma = 0.99
value_coef = 0.5
entropy_coef = 0.005
grad_clip = 1.0

min_random_episodes = 250
min_mixed_episodes = 150
gate_window = 50
random_gate_win_rate = 0.80
random_gate_margin = 8.0
mixed_gate_win_rate = 0.55
mixed_gate_margin = 0.0


def discount_rewards(rewards):
    discounted = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for i in reversed(range(rewards.size)):
        if rewards[i] != 0:
            running = 0.0
        running = running * gamma + rewards[i]
        discounted[i] = running
    return discounted


def agent_view(obs, side):
    if side == "right":
        return obs["first_0"]
    return obs["second_0"][:, ::-1, :]


def opponent_action(mode, frame, side, rng, last_ball_y):
    if mode == "random":
        return int(rng.choice(RANDOM_ACTIONS)), last_ball_y
    if mode == "mixed" and rng.random() < 0.5:
        return int(rng.choice(RANDOM_ACTIONS)), last_ball_y
    action, last_ball_y = ball_follower_action(frame, side=side, last_ball_y=last_ball_y)
    if action == NOOP:
        action = int(rng.choice((UP, DOWN)))
    return action, last_ball_y


def play_episode(env, net, side, curriculum, rng):
    obs, _ = env.reset()
    prev = None
    logps, values, probs, rewards = [], [], [], []
    last_ball_y = None
    agent_score = opponent_score = 0

    while env.agents and max(agent_score, opponent_score) < POINTS_TO_WIN:
        frame = agent_view(obs, side)
        cur = preprocess(frame)
        diff = cur - prev if prev is not None else np.zeros(D, np.float32)
        prev = cur

        prob, value = net(torch.from_numpy(diff).unsqueeze(0).to(DEVICE))
        prob = prob.squeeze(0)
        value = value.squeeze(0)
        choose_up = torch.rand((), device=DEVICE) < prob
        agent_action = UP if choose_up.item() else DOWN

        logps.append(torch.log((prob if choose_up else 1 - prob) + 1e-8))
        values.append(value)
        probs.append(prob)

        if side == "right":
            opp_action, last_ball_y = opponent_action(
                curriculum, obs["first_0"], "left", rng, last_ball_y
            )
            actions = {"first_0": agent_action, "second_0": opp_action}
        else:
            opp_action, last_ball_y = opponent_action(
                curriculum, obs["first_0"], "right", rng, last_ball_y
            )
            actions = {"first_0": opp_action, "second_0": agent_action}

        obs, reward_map, _, _, _ = env.step(actions)
        first_reward = reward_map.get("first_0", 0.0)
        reward = first_reward if side == "right" else -first_reward
        rewards.append(reward)
        if reward > 0:
            agent_score += 1
        elif reward < 0:
            opponent_score += 1

    returns = torch.tensor(
        discount_rewards(np.array(rewards)), dtype=torch.float32, device=DEVICE
    )
    return {
        "logps": torch.stack(logps),
        "values": torch.stack(values),
        "probs": torch.stack(probs),
        "returns": returns,
        "reward_sum": float(sum(rewards)),
        "agent_score": agent_score,
        "opponent_score": opponent_score,
    }


def build_loss(ep):
    values = ep["values"]
    returns = ep["returns"]
    advantage = returns - values.detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    policy_loss = -(ep["logps"] * advantage).sum()
    value_loss = value_coef * (values - returns).pow(2).mean()
    p = ep["probs"].clamp(1e-6, 1 - 1e-6)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
    return policy_loss + value_loss - entropy_coef * entropy


def load_checkpoint(net, opt):
    if not SAVE.exists():
        return 0, "random", []
    ck = torch.load(SAVE, map_location=DEVICE, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        net.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        return int(ck.get("episode", 0)), str(ck.get("curriculum", "random")), list(ck.get("recent", []))
    net.load_state_dict(ck)
    return 0, "random", []


def save_checkpoint(net, opt, episode, curriculum, recent):
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "episode": episode,
            "curriculum": curriculum,
            "recent": list(recent),
        },
        SAVE,
    )


def should_advance(curriculum, episode, recent):
    if len(recent) < gate_window:
        return False
    win_rate = float(np.mean([1 if item["reward_sum"] > 0 else 0 for item in recent]))
    margin = float(np.mean([item["reward_sum"] for item in recent]))
    if curriculum == "random":
        return (
            episode >= min_random_episodes
            and win_rate >= random_gate_win_rate
            and margin >= random_gate_margin
        )
    if curriculum == "mixed":
        return (
            episode >= min_random_episodes + min_mixed_episodes
            and win_rate >= mixed_gate_win_rate
            and margin >= mixed_gate_margin
        )
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Train realpong.")
    parser.add_argument("--episodes", type=int, default=0, help="Stop after N episodes. 0 means run until interrupted.")
    parser.add_argument("--fresh", action="store_true", help="Ignore the saved checkpoint and train from scratch.")
    parser.add_argument("--save-every", type=int, default=25, help="Save every N episodes.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = PolicyNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    episode = 0
    curriculum = "random"
    previous_recent = []
    if os.path.exists(SAVE) and not args.fresh:
        episode, curriculum, previous_recent = load_checkpoint(net, opt)
        print(f"resumed {SAVE.name} at episode {episode} ({curriculum})")
    else:
        print(f"fresh realpong -> {SAVE.name}")

    recent = deque(previous_recent[-gate_window:], maxlen=gate_window)
    env = pong_v3.parallel_env(max_cycles=MAX_CYCLES)
    opt.zero_grad()
    start_episode = episode
    print("training realpong: both sides, curriculum random -> mixed -> follower. Ctrl-C to stop.")

    try:
        while args.episodes == 0 or episode - start_episode < args.episodes:
            side = "right" if episode % 2 == 0 else "left"
            ep = play_episode(env, net, side, curriculum, rng)
            build_loss(ep).backward()
            episode += 1

            if episode % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()
                opt.zero_grad()

            recent.append(
                {
                    "reward_sum": ep["reward_sum"],
                    "agent_score": ep["agent_score"],
                    "opponent_score": ep["opponent_score"],
                    "side": side,
                }
            )
            if should_advance(curriculum, episode, recent):
                curriculum = "mixed" if curriculum == "random" else "follower"
                recent.clear()
                print(f">>> curriculum advanced to {curriculum}")

            avg = float(np.mean([item["reward_sum"] for item in recent])) if recent else 0.0
            print(
                f"episode {episode:5d} | side {side:5s} | "
                f"score {ep['agent_score']:2d}-{ep['opponent_score']:2d} | "
                f"reward {ep['reward_sum']:+5.0f} | avg {avg:+6.2f} | {curriculum}"
            )

            if episode % args.save_every == 0:
                save_checkpoint(net, opt, episode, curriculum, recent)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()
        save_checkpoint(net, opt, episode, curriculum, recent)
        print(f"saved {SAVE}")


if __name__ == "__main__":
    main()

