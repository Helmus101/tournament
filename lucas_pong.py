"""lucas_pong.py  --  Soumission de Lucas : politique CNN entrainee en PPO.
Fonctionne sur les DEUX arenes : arena.py (standard) ET arena_chaos.py.

Contract (charge par arena.py comme `lucas_pong.py:lucas.pt`):
    class Agent: __init__(weights_path), reset(), act(frame)->2/3/0

  python arena.py        realpong.py:realpong.pt  lucas_pong.py:lucas.pt
  python arena_chaos.py  realpong.py:realpong.pt  lucas_pong.py:lucas.pt

Le reseau ET le pretraitement exact vivent ici -> entrainement et jeu identiques.

Input to the net = 2 channels x 80 x 80:
    channel 0 = current frame (1.0 = paddle/ball pixel)
    channel 1 = current - previous frame  (motion; localizes the 1-px ball)
Policy outputs 2 logits; index -> env action: 0=UP(2), 1=DOWN(3). Always moving (no STAY).
"""
import numpy as np
import torch
import torch.nn as nn

UP, DOWN = 2, 3
ACTIONS = [UP, DOWN]                # policy index -> env action (always moving; no STAY)
N_ACT = 2
SIZE = 80
# DEVICE is used by TRAINING (train_ppo.py) for big batched updates -> keep CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PLAY device: act() runs ONE 2x80x80 sample per frame. At that size CPU beats GPU
# (no per-frame host<->device copy / kernel-launch overhead), so play on CPU.
PLAY_DEVICE = torch.device("cpu")


class PolicyCNN(nn.Module):
    """2x80x80 -> conv stack -> (policy logits[3], value[1])."""
    def __init__(self, n_actions=N_ACT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=8, stride=4),    # 80 -> 19
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),   # 19 -> 8
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 8 * 8, 256), nn.ReLU(inplace=True))
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        h = self.fc(h)
        return self.pi(h), self.v(h).squeeze(-1)


class FrameProcessor:
    """Turns a raw 80x80 frame into the 2-channel [current, current-prev] input.
    Holds the previous frame; MUST be reset at the start of each game."""
    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, frame):
        cur = np.asarray(frame, dtype=np.float32)
        diff = cur - self.prev if self.prev is not None else np.zeros_like(cur)
        self.prev = cur
        return np.stack([cur, diff], axis=0)          # (2, 80, 80)


class Agent:
    """Play-time agent. Optimized for low per-frame latency (act() is called once
    per env step), WITHOUT changing any decision -- the argmax is byte-for-byte
    identical to the naive forward. Optimizations:
      * run on CPU with a single thread  (single-sample inference: thread-dispatch
        and GPU transfer overhead dominate the tiny conv -> both hurt; ~1.7x faster)
      * torch.inference_mode + pre-allocated numpy/torch buffers (zero alloc/frame)
      * skip the value head (unused in play) and reduce argmax(2) to a scalar compare
      * optional frozen TorchScript graph (a further ~7%), with eager fallback.
    """
    def __init__(self, weights_path=None):
        torch.set_num_threads(1)                       # biggest single win on this tiny net
        self.net = PolicyCNN().to(PLAY_DEVICE)
        if weights_path:
            ck = torch.load(weights_path, map_location=PLAY_DEVICE, weights_only=False)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.net.load_state_dict(sd)
        self.net.eval()

        # zero-allocation buffers: [current, current-prev] reused every frame
        self._buf = np.zeros((2, SIZE, SIZE), dtype=np.float32)
        self._x = torch.zeros((1, 2, SIZE, SIZE), dtype=torch.float32, device=PLAY_DEVICE)
        self.prev = None

        # policy-only module (no value head) + try to freeze it with TorchScript
        self._infer = self._make_infer(self.net)

    @staticmethod
    def _make_infer(net):
        class _PiOnly(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.conv, self.fc, self.pi = net.conv, net.fc, net.pi
            def forward(self, x):
                h = self.conv(x)
                h = self.fc(h.reshape(1, -1))
                return self.pi(h)[0]                    # (2,) logits
        m = _PiOnly(net).eval()
        try:
            with torch.inference_mode():
                m = torch.jit.freeze(torch.jit.trace(m, torch.zeros(1, 2, SIZE, SIZE)))
                for _ in range(8):                      # warm up the JIT graph
                    m(torch.zeros(1, 2, SIZE, SIZE))
        except Exception:
            pass                                        # eager fallback (still fast)
        return m

    def reset(self):
        self.prev = None

    @torch.inference_mode()
    def act(self, frame) -> int:
        cur = np.asarray(frame, dtype=np.float32)
        self._buf[0] = cur
        if self.prev is not None:
            np.subtract(cur, self.prev, out=self._buf[1])
        else:
            self._buf[1].fill(0.0)
        self.prev = cur
        self._x.copy_(torch.from_numpy(self._buf))
        logits = self._infer(self._x)
        return ACTIONS[1 if logits[1].item() > logits[0].item() else 0]
