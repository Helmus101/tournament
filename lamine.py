"""lamine.py -- Lamine's tournament entry (REGULAR arena). Pairs with lamine.pt.

A CNN agent. Input = 2 channels of 80x80: [current frame (position), current - previous (motion)].
3 conv layers (32/64/64) -> fc 256 -> policy head (P(UP)) + value head, ~1.7M params, full 80x80
resolution. Trained by first cloning a strong opponent to reach its level, then PPO self-play
against a pool of past selves with a pure win/loss reward to push past it.

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
            nn.Conv2d(2, 32, 8, stride=4, padding=2), nn.ReLU(),   # (2,80,80) -> (32,20,20)
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # -> (64,10,10)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # -> (64,10,10) = 6400
        )
        self.fc = nn.Linear(64 * 10 * 10, hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, 2, 80, 80)
        h = self.conv(x).flatten(1)
        h = torch.relu(self.fc(h))
        return torch.sigmoid(self.policy_head(h)).squeeze(-1), self.value_head(h).squeeze(-1)


def features(cur, prev):
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
                print(f"[lamine] weights don't fit this CNN ({e}) -- using random init")
        self.net.eval()
        self.prev = None

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, frame):
        cur = frame.astype(np.float32).ravel()
        x = features(cur, self.prev)
        self.prev = cur
        prob, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        return UP if float(prob.item()) > 0.5 else DOWN
