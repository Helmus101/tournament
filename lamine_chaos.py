"""lamine_chaos.py -- Lamine's tournament entry (CHAOS arena). Pairs with lamine_chaos.pt.

A CNN agent. Input = 2 channels of 80x80: [current frame (position), current - previous (motion)].
2 conv layers (16/32) -> fc 256 -> policy head (P(UP)) + value head. Trained with PPO self-play
against a pool of past selves inside the chaos environment (randomised ball speed, jolts, curve),
pure win/loss reward.

Self-contained: only needs numpy + torch. The arena imports this module and calls
Agent(weights_path); act() returns 2 (UP) or 3 (DOWN), own paddle on the RIGHT.
"""
import os
import numpy as np
import torch
import torch.nn as nn

UP, DOWN = 2, 3
D = 80 * 80


class Net(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 8, stride=4, padding=2), nn.ReLU(),   # (2,80,80) -> (16,20,20)
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),  # -> (32,10,10) = 3200
        )
        self.fc = nn.Linear(32 * 10 * 10, hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, 2, 80, 80)
        h = self.conv(x).flatten(1)
        h = torch.relu(self.fc(h))
        return torch.sigmoid(self.policy_head(h)).squeeze(-1), self.value_head(h).squeeze(-1)


def features(cur, prev):
    """cur, prev: flat 6400 frames. Returns a (2, 80, 80) image: [position, motion]."""
    diff = cur - prev if prev is not None else np.zeros(D, np.float32)
    return np.stack([cur.reshape(80, 80), diff.reshape(80, 80)]).astype(np.float32)


class Agent:
    def __init__(self, weights_path=None):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            try:
                self.net.load_state_dict(state)
            except Exception as e:
                print(f"[lamine_chaos] weights don't fit this CNN ({e}) -- using random init")
        self.net.eval()
        self.prev = None

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, frame):
        cur = frame.astype(np.float32).ravel()      # arena gives 80x80 binary, own paddle RIGHT
        x = features(cur, self.prev)
        self.prev = cur
        prob, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        return UP if float(prob.item()) > 0.5 else DOWN
