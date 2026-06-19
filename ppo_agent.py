""" ppo_agent.py  --  the EXTRA model: a CNN actor-critic over 4 stacked frames.
    Defines the network AND the arena Agent contract, so it battles via:
        python arena_ale.py ppo_agent.py:ppo.pt realpong.pt

    Modern stack (vs the MLP baselines): a small CNN reads 4 stacked preprocessed
    frames (so it sees motion/velocity directly, no hand-coded frame-differencing),
    with a shared trunk feeding a policy head (UP/DOWN) and a value head.
    Trained by ppo_pong.py (PPO + GAE + entropy + self-play).
"""

import os
import collections
import numpy as np
import torch
import torch.nn as nn

from agent_ale import preprocess, UP, DOWN          # verified preprocess (-> 6400 = 80x80)

STACK = 4                                            # frames stacked -> the net sees motion


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(STACK, 16, 8, stride=4), nn.ReLU(),   # 80x80 -> 19x19
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),      # 19x19 -> 8x8
            nn.Flatten(),
        )
        self.trunk  = nn.Sequential(nn.Linear(32 * 8 * 8, 256), nn.ReLU())
        self.pi     = nn.Linear(256, 2)              # logits for [UP, DOWN]
        self.vf     = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(self.conv(x))
        return self.pi(h), self.vf(h).squeeze(-1)


def stack_to_tensor(frames, device="cpu"):
    """ deque of (80,80) float frames -> tensor (1, STACK, 80, 80) """
    fs = list(frames)
    while len(fs) < STACK:
        fs.insert(0, np.zeros((80, 80), np.float32))
    return torch.from_numpy(np.stack(fs[-STACK:])).unsqueeze(0).to(device)


class Agent:
    """ arena contract: reset() + act(frame 210x160x3) -> 2 (UP) | 3 (DOWN). Greedy. """
    def __init__(self, weights_path=None):
        self.net = ActorCritic()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        self.net.eval()
        self.frames = collections.deque(maxlen=STACK)

    def reset(self):
        self.frames.clear()

    @torch.no_grad()
    def act(self, frame):
        self.frames.append(preprocess(frame).reshape(80, 80))
        logits, _ = self.net(stack_to_tensor(self.frames))
        a = int(torch.argmax(logits, dim=-1).item())     # greedy at eval
        return UP if a == 0 else DOWN
