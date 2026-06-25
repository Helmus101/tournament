"""pong.py -- the pong MODEL (CNN + Agent) AND its PPO trainer, in one file.

  * The tournament imports this file for its `Agent` class:
        python arena.py pong.py:pong_best.pt  bf random
  * You run this file to TRAIN (only happens when executed directly):
        python pong.py --fresh                       # PPO from scratch
        python pong.py --fresh --init-from realpong.py:realpong.pt   # warm-start from realpong
        python pong.py                               # resume pong.pt
        python pong.py --rollouts 20                 # run 20 rollouts then exit
Importing this file (what the tournament does) only gives you `Agent`; the training
loop under main() does not run on import.

Pixels only, no hand-coded tracking. The agent sees the arena's real 80x80 frame, shrinks
it to 40x40 internally (2x2 max-pool -- the env is unchanged, the agent just "squints"), and
feeds a 2-channel image [position, motion] to a small CNN. The motion channel = current minus
previous downsampled frame, so the net reads ball SPEED straight from the pixels at any speed.
Two actions only (UP/DOWN) -- the paddle is always moving. Acts every frame.

Trained with PPO (clipped surrogate + GAE + value-clip + entropy + KL early-break), vectorized
over copies of the UNMODIFIED arena env. Curriculum: random warm-up -> lagged ball-follower
ladder (lag 24->0 in steps of 1, advance at >=90% win over a persistent 100-game window) ->
self-play vs a pool of frozen past selves once return accuracy >= ~50%. Reward = env +/-1
(dominant) + small annealed shaping for harder returns. Rollout runs on CPU, the PPO update on
MPS (Apple GPU); keeps the BEST checkpoint by a fixed yardstick (vs bf lag-8) with early-stop.
"""
from __future__ import annotations

import argparse
import importlib
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arena import PongSym, TrackerAgent, RandomAgent, BALL_SPEED_MAX, SIZE

UP, DOWN = 2, 3
ACTIONS = [UP, DOWN]          # policy index 0 -> UP, 1 -> DOWN
DS = 40                       # downsampled side (2x2 max-pool of the 80x80 frame)

HERE = Path(__file__).resolve().parent
SAVE = HERE / "pong.pt"
BEST = HERE / "pong_best.pt"

# Environment used for rollouts AND the keep-best eval. Defaults to the standard PongSym;
# main() swaps in arena_chaos.ChaosPong with --chaos, writing pong_chaos.pt / pong_chaos_best.pt
# so a chaos run never touches pong.pt / pong_best.pt (the standard-arena best).
ENV_CLASS = PongSym
# With --both, MIX_ENVS = [standard, chaos] and each rollout game is drawn from them with
# P(chaos) = MIX_CHAOS_FRAC, so ONE generalist learns both arenas but PRIORITISES chaos.
MIX_ENVS = None
MIX_CHAOS_FRAC = 0.6        # share of rollout games on the chaos env when --both (set by --chaos-frac)


def new_env(rng, points):
    """Make a fresh env for a rollout game: a chaos-weighted draw from MIX_ENVS (generalist) or ENV_CLASS."""
    if MIX_ENVS:
        cls = MIX_ENVS[1] if rng.random() < MIX_CHAOS_FRAC else MIX_ENVS[0]   # [0]=standard, [1]=chaos
    else:
        cls = ENV_CLASS
    return cls(seed=int(rng.integers(1 << 30)), points=points)


# ══════════════════════════════════════════════════════════════════════════════
#  THE MODEL  (used by the tournament; trained by main() below)
# ══════════════════════════════════════════════════════════════════════════════
def downsample(frame):
    """80x80 -> 40x40 by 2x2 max-pool. Frames are binary, so max-pool keeps the
    1-pixel ball and the paddle columns. Agent-side preprocessing only -- the
    environment still produces the full 80x80 frame."""
    f = np.asarray(frame, dtype=np.float32).reshape(SIZE, SIZE)
    return f.reshape(DS, 2, DS, 2).max(axis=(1, 3))


def features(cur_ds, prev_ds):
    """Build the (2, 40, 40) network input: [position, motion]. Motion is the diff of
    the two most-recent downsampled frames; zero on the very first frame of a game."""
    motion = (cur_ds - prev_ds) if prev_ds is not None else np.zeros_like(cur_ds)
    return np.stack([cur_ds, motion]).astype(np.float32)


class Net(nn.Module):
    """Small CNN -> shared trunk -> policy logits (2) + value. CNN (not MLP) so it can
    localize the 1-pixel ball anywhere via shared spatial filters."""
    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1), nn.ReLU(),   # (2,40,40) -> (32,20,20)
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # -> (64,10,10)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # -> (64,10,10) = 6400
        )
        self.fc = nn.Linear(64 * 10 * 10, hidden)
        self.policy_head = nn.Linear(hidden, 2)    # logits over [UP, DOWN]
        self.value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2.0)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fc.weight, np.sqrt(2.0)); nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.policy_head.weight, 0.01); nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0);  nn.init.zeros_(self.value_head.bias)

    def forward(self, x):                          # x: (B, 2, 40, 40)
        h = self.conv(x).flatten(1)
        h = torch.relu(self.fc(h))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class Agent:
    """Tournament contract: reset() at game start, act(80x80 frame, own paddle RIGHT) -> 2|3."""
    def __init__(self, weights_path=None):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            try:
                self.net.load_state_dict(state)
            except Exception as e:                 # LOUD: never silently play with random weights
                raise RuntimeError(f"[pong] weights {weights_path} don't fit this CNN: {e}")
        self.net.eval()
        self.prev_ds = None

    def reset(self):
        self.prev_ds = None

    @torch.no_grad()
    def act(self, frame):
        cur_ds = downsample(frame)
        x = features(cur_ds, self.prev_ds)
        self.prev_ds = cur_ds
        logits, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        return ACTIONS[int(torch.argmax(logits, dim=-1).item())]   # greedy at eval


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING  (only runs when this file is executed directly)
# ══════════════════════════════════════════════════════════════════════════════
N_ENVS        = 8
ROLLOUT_STEPS = 4096            # transitions per rollout (~512 per env)
K_EPOCHS      = 4
MINIBATCH     = 512
LR            = 2.5e-4
CLIP_EPS      = 0.2
GAMMA         = 0.99
LAM           = 0.95
VALUE_COEF    = 0.5
ENT_COEF      = 0.01
GRAD_CLIP     = 0.5
TARGET_KL     = 0.02
ANNEAL_ROLLOUTS = 300           # shaping weights anneal 1->0 over this many rollouts

HIT_BASE    = 0.02              # returning the ball at all (accuracy)
HIT_SPEED   = 0.03              # bonus scaled by ball speed at contact (harder/faster ball)
HIT_STRETCH = 0.01              # bonus for an off-centre return (a "further"/harder save)

WINDOW          = 100
RANDOM_GATE     = 0.80
LEVEL_GATE      = 0.85          # advance the lag ladder above 85% win (keeps it moving as lag drops)
LAG_START       = 24           # start vs the easy tracker (lag 24)
SELFPLAY_ACC    = 0.50
SELFPLAY_AT_LAG = 12
POOL_SIZE       = 5
POOL_GATE       = 0.60

EVAL_EVERY    = 5
EVAL_GAMES    = 21
EARLY_STOP_K  = 10

DEVICE = torch.device("cpu")    # update device, set in main()
EXTERNAL_FACTORY = None          # if set (by --opponent-file), builds the fixed external sparring opponent


def match_points(phase, lag, pool_size):
    if phase == "random":
        return 5            # quick cold-start warm-up only
    return 21               # train on full 21-point games (matches the tournament)


def agent_from_state(state):
    """A greedy self-play opponent: the pong CNN with a frozen snapshot, its own frame stack."""
    a = Agent()
    a.net.load_state_dict(state)
    a.net.eval()
    return a


def make_opponent(phase, lag, pool, rng):
    if phase == "external":
        return EXTERNAL_FACTORY()                # a fixed external agent (e.g. newfolder)
    if phase == "random":
        return RandomAgent(int(rng.integers(1 << 30)))
    if phase == "tracker":
        return TrackerAgent(lag=lag, seed=int(rng.integers(1 << 30)))
    return agent_from_state(pool[int(rng.integers(len(pool)))])


def collect_rollout(net, phase, lag, points, pool, anneal, rng):
    """Play N_ENVS games in lockstep on CPU; policy = RIGHT player, opponent = LEFT.
    Returns CPU tensors (main moves them to the update device) + finished-game stats."""
    steps = max(1, ROLLOUT_STEPS // N_ENVS)
    envs = [new_env(rng, points) for _ in range(N_ENVS)]
    obs = [e.reset(seed=int(rng.integers(1 << 30))) for e in envs]
    opps = [make_opponent(phase, lag, pool, rng) for _ in range(N_ENVS)]
    for o in opps:
        o.reset()
    prev_ds = [None] * N_ENVS
    buffers = [[] for _ in range(N_ENVS)]
    games = []

    for _ in range(steps):
        cur_ds = [downsample(obs[i]["right"]) for i in range(N_ENVS)]
        X = np.stack([features(cur_ds[i], prev_ds[i]) for i in range(N_ENVS)])
        with torch.no_grad():
            logits, values = net(torch.from_numpy(X))           # CPU forward (no per-step GPU sync)
            logp_all = torch.log_softmax(logits, dim=-1)
            p_up = logp_all[:, 0].exp()                          # P(action 0 = UP)
            a_idx = (torch.rand(logits.shape[0]) >= p_up).long() # manual 2-way sample (no multinomial)
            logp = logp_all.gather(1, a_idx.unsqueeze(1)).squeeze(1)
        a_idx_np = a_idx.numpy(); logp_np = logp.numpy(); val_np = values.numpy()

        for i in range(N_ENVS):
            a_right = ACTIONS[int(a_idx_np[i])]
            a_left = opps[i].act(obs[i]["left"])
            ob, rew, done, info = envs[i].step(a_right, a_left)
            r = float(rew["right"])
            scored = (r != 0.0)
            shaped = 0.0
            if not scored and info["hit_r"]:                     # we returned the ball
                speed = abs(envs[i].bvx) / BALL_SPEED_MAX
                stretch = abs(envs[i].by - SIZE / 2.0) / (SIZE / 2.0)
                shaped = anneal * (HIT_BASE + HIT_SPEED * speed + HIT_STRETCH * stretch)
            buffers[i].append(dict(x=X[i], a=int(a_idx_np[i]), logp=float(logp_np[i]),
                                   value=float(val_np[i]), reward=r + shaped,
                                   boundary=(scored or done),
                                   hit=int(info["hit_r"]), miss=int(info["miss_r"])))
            prev_ds[i] = cur_ds[i]
            obs[i] = ob
            if done:
                games.append(1 if envs[i].score_r > envs[i].score_l else 0)
                envs[i] = new_env(rng, points)
                obs[i] = envs[i].reset(seed=int(rng.integers(1 << 30)))
                opps[i] = make_opponent(phase, lag, pool, rng); opps[i].reset()
                prev_ds[i] = None

    cur_ds = [downsample(obs[i]["right"]) for i in range(N_ENVS)]
    Xb = np.stack([features(cur_ds[i], prev_ds[i]) for i in range(N_ENVS)])
    with torch.no_grad():
        _, boot = net(torch.from_numpy(Xb))
    boot = boot.numpy()

    for i in range(N_ENVS):                          # GAE per env (reset at point/game boundaries)
        buf = buffers[i]
        adv = 0.0
        for t in reversed(range(len(buf))):
            nonterm = 0.0 if buf[t]["boundary"] else 1.0
            next_v = boot[i] if t == len(buf) - 1 else buf[t + 1]["value"]
            delta = buf[t]["reward"] + GAMMA * nonterm * next_v - buf[t]["value"]
            adv = delta + GAMMA * LAM * nonterm * adv
            buf[t]["adv"] = adv
            buf[t]["ret"] = adv + buf[t]["value"]

    flat = [b for buf in buffers for b in buf]
    hits = sum(b["hit"] for b in flat); misses = sum(b["miss"] for b in flat)
    data = dict(                                     # CPU tensors; main() moves them to the update device
        x=torch.from_numpy(np.stack([b["x"] for b in flat])),
        a=torch.tensor([b["a"] for b in flat], dtype=torch.long),
        logp=torch.tensor([b["logp"] for b in flat], dtype=torch.float32),
        ret=torch.tensor([b["ret"] for b in flat], dtype=torch.float32),
        adv=torch.tensor([b["adv"] for b in flat], dtype=torch.float32),
        val=torch.tensor([b["value"] for b in flat], dtype=torch.float32),
    )
    return data, games, hits, misses


def ppo_update(net, opt, data):
    adv = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
    n = data["x"].shape[0]
    ent_log = 0.0
    for _ in range(K_EPOCHS):
        idx = torch.randperm(n, device=DEVICE)
        stop = False
        for s in range(0, n, MINIBATCH):
            mb = idx[s:s + MINIBATCH]
            logits, value = net(data["x"][mb])
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, data["a"][mb].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - data["logp"][mb])
            a_mb = adv[mb]
            l_clip = -torch.min(ratio * a_mb,
                                torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a_mb).mean()
            v_clip = data["val"][mb] + (value - data["val"][mb]).clamp(-CLIP_EPS, CLIP_EPS)
            l_v = VALUE_COEF * torch.max((value - data["ret"][mb]).pow(2),
                                         (v_clip - data["ret"][mb]).pow(2)).mean()
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            loss = l_clip + l_v - ENT_COEF * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            ent_log = float(ent.item())
            with torch.no_grad():
                approx_kl = (data["logp"][mb] - logp).mean().item()
            if approx_kl > 1.5 * TARGET_KL:          # PPO safeguard against destructive steps
                stop = True; break
        if stop:
            break
    return ent_log


@torch.no_grad()
def eval_vs(net, opp_factory, n_games, points=21, env_cls=None):
    cls = env_cls or ENV_CLASS
    me = Agent()
    me.net.load_state_dict({k: v.detach().cpu() for k, v in net.state_dict().items()})
    me.net.eval()
    rng = np.random.default_rng(12345)               # fixed seeds -> low-variance yardstick
    wins, conceded = 0, 0
    for g in range(n_games):
        env = cls(seed=int(rng.integers(1 << 30)), points=points)
        ob = env.reset(seed=int(rng.integers(1 << 30)))
        opp = opp_factory(rng); opp.reset(); me.reset()
        net_on_right = (g % 2 == 0)                  # play both sides
        done = False
        while not done:
            if net_on_right:
                ob, _, done, _ = env.step(me.act(ob["right"]), opp.act(ob["left"]))
            else:
                ob, _, done, _ = env.step(opp.act(ob["right"]), me.act(ob["left"]))
        my, their = (env.score_r, env.score_l) if net_on_right else (env.score_l, env.score_r)
        wins += 1 if my > their else 0
        conceded += their
    return wins / n_games, conceded / n_games


def _yardstick_env(net, env_cls):
    wr8, conc8 = eval_vs(net, lambda r: TrackerAgent(lag=8, seed=int(r.integers(1 << 30))), EVAL_GAMES, env_cls=env_cls)
    wr0, _ = eval_vs(net, lambda r: TrackerAgent(lag=0, seed=int(r.integers(1 << 30))), max(7, EVAL_GAMES // 3), env_cls=env_cls)
    return wr8 - 0.01 * conc8 + 0.25 * wr0, wr8, conc8, wr0


def yardstick(net):
    # generalist: score on BOTH envs, combined with the SAME chaos weighting as training
    # (keep-best then favours the chaos arena, matching the priority).
    if MIX_ENVS:
        a = _yardstick_env(net, MIX_ENVS[0])     # standard
        b = _yardstick_env(net, MIX_ENVS[1])     # chaos
        w = MIX_CHAOS_FRAC
        return tuple((1 - w) * x + w * y for x, y in zip(a, b))
    return _yardstick_env(net, ENV_CLASS)


def atomic_save(obj, path):
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)   # verify before replacing
    os.replace(tmp, path)


def distill(net, spec, rng, n_frames=6000, epochs=3):
    """Warm-start: clone another agent's policy into our CNN (behavior cloning). Reads the teacher only."""
    code, wpath = spec.split(":")
    if not os.path.exists(wpath): wpath = str(HERE / wpath)   # resolve relative to this file's dir
    mod = importlib.import_module(code[:-3] if code.endswith(".py") else code)
    try:
        teacher = mod.Agent(wpath, stochastic=False)
    except TypeError:
        teacher = mod.Agent(wpath)
    print(f"[distill] cloning {spec} into the CNN over ~{n_frames} frames ...")
    xs, ys = [], []
    while len(xs) < n_frames:
        env = ENV_CLASS(seed=int(rng.integers(1 << 30)), points=21)
        ob = env.reset(seed=int(rng.integers(1 << 30)))
        teacher.reset(); opp = TrackerAgent(lag=8, seed=int(rng.integers(1 << 30))); opp.reset()
        prev_ds = None; done = False
        while not done and len(xs) < n_frames:
            frame = ob["right"]
            a = teacher.act(frame)
            cur_ds = downsample(frame)
            xs.append(features(cur_ds, prev_ds)); ys.append(0 if a == UP else 1)
            prev_ds = cur_ds
            ob, _, done, _ = env.step(a, opp.act(ob["left"]))
    X = torch.from_numpy(np.stack(xs)).to(DEVICE)
    Y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(Y), device=DEVICE)
        tot = 0.0
        for s in range(0, len(Y), 256):
            mb = perm[s:s + 256]
            logits, _ = net(X[mb])
            loss = ce(logits, Y[mb])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        print(f"[distill] epoch {ep + 1}/{epochs} ce={tot / max(1, len(Y) // 256):.3f}")
    print("[distill] done -> policy warm-started (value head learns fresh in PPO).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=0, help="stop after N rollouts (0 = until Ctrl-C)")
    ap.add_argument("--fresh", action="store_true", help="ignore saved checkpoint")
    ap.add_argument("--init-from", type=str, default=None, help="warm-start by distilling code.py:weights.pt")
    ap.add_argument("--selfplay", action="store_true",
                    help="force self-play now: spar a pool of frozen past selves (overrides phase)")
    ap.add_argument("--opponent-file", type=str, default=None,
                    help="train vs a fixed external agent code.py:weights.pt (e.g. newfolder.py:newfolder_trained_best.pt)")
    ap.add_argument("--save-every", type=int, default=2, help="save pong.pt every N rollouts")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--chaos", action="store_true",
                    help="train on the HARDER arena_chaos.ChaosPong env (random, non-physical ball "
                         "speed). Saves to pong_chaos.pt / pong_chaos_best.pt and warm-starts from "
                         "pong.pt, so pong.pt / pong_best.pt (the standard-arena best) stay untouched")
    ap.add_argument("--both", action="store_true",
                    help="train ONE generalist on BOTH envs (chaos-weighted, see --chaos-frac). Saves to "
                         "pong_both.pt / pong_both_best.pt (best by chaos-weighted std+chaos score); "
                         "pong.pt / pong_best.pt / willem-cnn.pt stay untouched")
    ap.add_argument("--chaos-frac", type=float, default=0.6,
                    help="with --both: fraction of training (and eval weight) on the chaos env (default 0.6)")
    args = ap.parse_args()

    # ── env + save-path selection (each mode writes its OWN files -> no collision) ──
    global ENV_CLASS, MIX_ENVS, MIX_CHAOS_FRAC, SAVE, BEST
    if args.both:
        from arena_chaos import ChaosPong
        MIX_ENVS = [PongSym, ChaosPong]         # [0]=standard, [1]=chaos
        MIX_CHAOS_FRAC = args.chaos_frac
        if SAVE == HERE / "pong.pt":            # only redirect the DEFAULT paths (tests can override)
            SAVE = HERE / "pong_both.pt"
        if BEST == HERE / "pong_best.pt":
            BEST = HERE / "pong_both_best.pt"
        print(f"*** GENERALIST: {int((1-MIX_CHAOS_FRAC)*100)}/{int(MIX_CHAOS_FRAC*100)} standard/chaos "
              f"-> {SAVE.name} / {BEST.name} ***")
    elif args.chaos:
        from arena_chaos import ChaosPong
        ENV_CLASS = ChaosPong
        if SAVE == HERE / "pong.pt":            # only redirect the DEFAULT paths (tests can override)
            SAVE = HERE / "pong_chaos.pt"
        if BEST == HERE / "pong_best.pt":
            BEST = HERE / "pong_chaos_best.pt"
        print(f"*** CHAOS env (random ball speed) -> {SAVE.name} / {BEST.name} ***")

    global DEVICE
    use_mps = (args.device in ("auto", "mps")) and torch.backends.mps.is_available()
    if use_mps:
        DEVICE = torch.device("mps")
        print("device: MPS (Apple GPU) for the PPO update; rollout on CPU. First rollout pays a one-time ~60s compile.")
    else:
        DEVICE = torch.device("cpu")
        print("device: CPU (no MPS found) — slower; the PPO update is the heavy part.")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = Net().to(DEVICE)                 # training net (+ optimizer) on the update device
    net_cpu = Net()                        # rollout net on CPU, re-synced from `net` each rollout
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)

    rollout = 0
    phase, lag = "tracker", LAG_START   # start vs the EASY tracker (lag 24): random is a chaotic,
                                        # unlearnable teacher; a tracker rallies and gives real signal
    pool, best_score, no_improve = [], float("-inf"), 0
    win_w, hit_w, miss_w = deque(maxlen=WINDOW), deque(maxlen=WINDOW), deque(maxlen=WINDOW)

    if SAVE.exists() and not args.fresh:
        ck = torch.load(SAVE, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        rollout = ck.get("rollout", 0); phase = ck.get("phase", "tracker"); lag = ck.get("lag", LAG_START)
        pool = [{k: v.cpu() for k, v in s.items()} for s in ck.get("pool", [])]
        best_score = ck.get("best_score", float("-inf")); no_improve = ck.get("no_improve", 0)
        win_w.extend(ck.get("win_w", [])); hit_w.extend(ck.get("hit_w", [])); miss_w.extend(ck.get("miss_w", []))
        print(f"resumed pong.pt @ rollout {rollout} (phase {phase}, lag {lag}, pool {len(pool)})")
    else:
        if args.init_from:
            distill(net, args.init_from, rng)
            opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)   # fresh optimizer after distill
            print("fresh pong + warm-start")
        elif args.both and (HERE / "willem-cnn.pt").exists():
            # FIRST generalist run: warm-start from willem-cnn.pt -- the current champion
            # (best at BOTH arenas) -- and keep adapting via the chaos-weighted mix.
            ck = torch.load(HERE / "willem-cnn.pt", map_location=DEVICE, weights_only=False)
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            print("GENERALIST warm-start: copied weights from willem-cnn.pt (read-only); curriculum starts fresh")
        elif args.both and (HERE / "pong.pt").exists():
            ck = torch.load(HERE / "pong.pt", map_location=DEVICE, weights_only=False)
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            print("GENERALIST warm-start: copied weights from pong.pt (read-only); curriculum starts fresh")
        elif args.chaos and (HERE / "pong.pt").exists():
            # FIRST chaos run: warm-start weights from the already-trained standard model
            # (read-only, same architecture) instead of starting cold; curriculum starts fresh.
            ck = torch.load(HERE / "pong.pt", map_location=DEVICE, weights_only=False)
            net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
            print("CHAOS warm-start: copied weights from pong.pt (read-only); curriculum starts fresh")
        else:
            print("fresh pong")

    # ── fixed external sparring opponent (e.g. newfolder) — overrides the curriculum ──
    if args.opponent_file:
        global EXTERNAL_FACTORY
        code, wpath = args.opponent_file.split(":")
        if not os.path.exists(wpath): wpath = str(HERE / wpath)   # resolve relative to this file's dir
        extmod = importlib.import_module(code[:-3] if code.endswith(".py") else code)
        extck = torch.load(wpath, map_location="cpu", weights_only=False)
        extstate = extck["model"] if isinstance(extck, dict) and "model" in extck else extck
        def make_external():
            a = extmod.Agent()                       # build its net (random), then load the cached weights
            a.net.load_state_dict(extstate); a.net.eval()
            return a
        make_external()                              # fail loudly NOW if weights don't fit its Net
        EXTERNAL_FACTORY = make_external
        phase = "external"                           # train every game vs this fixed opponent
        print(f"opponent: fixed external agent {args.opponent_file} (curriculum disabled)")

    if phase == "external" and EXTERNAL_FACTORY is None:   # resumed an external-phase checkpoint w/o --opponent-file
        phase = "tracker"                                 # crash fix: fall back to the tracker ladder
        print("note: checkpoint was sparring an external opponent, but no --opponent-file given "
              "-> resuming vs the tracker (re-pass --opponent-file to keep sparring that agent)")

    if args.selfplay:                                     # force self-play now (overrides any phase)
        phase = "selfplay"
        print("forced self-play: sparring a pool of frozen past selves")
    if phase == "selfplay" and not pool:                  # seed the pool with the current model
        pool.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    def save():
        atomic_save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                     "optimizer": opt.state_dict(),
                     "rollout": rollout, "phase": phase, "lag": lag,
                     "pool": [{k: v.cpu() for k, v in s.items()} for s in pool],
                     "best_score": best_score, "no_improve": no_improve,
                     "win_w": list(win_w), "hit_w": list(hit_w), "miss_w": list(miss_w)}, str(SAVE))

    start = rollout
    print(f"training pong (PPO). phase {phase}. Ctrl-C to stop.")
    try:
        while args.rollouts == 0 or rollout - start < args.rollouts:
            anneal = max(0.0, 1.0 - rollout / ANNEAL_ROLLOUTS)
            points = match_points(phase, lag, len(pool))
            net_cpu.load_state_dict({k: v.detach().cpu() for k, v in net.state_dict().items()})
            data, wins, hits, misses = collect_rollout(net_cpu, phase, lag, points, pool, anneal, rng)
            data = {k: v.to(DEVICE) for k, v in data.items()}    # rollout on CPU, update on the device
            ent = ppo_update(net, opt, data)
            rollout += 1

            win_w.extend(wins); hit_w.append(hits); miss_w.append(misses)
            winrate = float(np.mean(win_w)) if win_w else 0.0
            arr = sum(hit_w) + sum(miss_w)
            acc = (sum(hit_w) / arr) if arr else 0.0

            if len(win_w) >= WINDOW:
                if phase == "random" and winrate >= RANDOM_GATE:
                    phase, lag = "tracker", LAG_START; win_w.clear(); hit_w.clear(); miss_w.clear()
                    print(f">>> warm-up cleared -> ball-follower ladder begins at lag {lag}")
                elif phase == "tracker":
                    if acc >= SELFPLAY_ACC and lag <= SELFPLAY_AT_LAG:
                        phase = "selfplay"; pool.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                        win_w.clear(); hit_w.clear(); miss_w.clear()
                        print(f">>> accuracy {acc*100:.0f}% @ lag {lag} -> SELF-PLAY (pool 1)")
                    elif winrate >= LEVEL_GATE and lag > 0:
                        lag -= 1; win_w.clear(); hit_w.clear(); miss_w.clear()
                        print(f">>> difficulty up: tracker lag -> {lag}")
                elif phase == "selfplay" and winrate >= POOL_GATE:
                    pool.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}); pool[:] = pool[-POOL_SIZE:]
                    win_w.clear(); hit_w.clear(); miss_w.clear()
                    print(f">>> self-play: added a stronger snapshot (pool {len(pool)})")

            opp_info = f"pool{len(pool)}" if phase == "selfplay" else f"lag{lag:2d}"
            print(f"r{rollout:4d} | {phase:8s} {opp_info:6s} | {points:2d}pt | win {winrate*100:4.0f}% "
                  f"| acc {acc*100:4.0f}% | ent {ent:.3f} | anneal {anneal:.2f}")

            if rollout % args.save_every == 0:
                save()

            if rollout % EVAL_EVERY == 0:
                score, wr8, conc8, wr0 = yardstick(net)
                tag = ""
                if score > best_score + 1e-4:
                    best_score = score; no_improve = 0
                    atomic_save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}}, str(BEST))
                    tag = f" -> NEW BEST (saved {BEST.name})"
                else:
                    no_improve += 1
                print(f"   [eval] vs bf8 win {wr8*100:.0f}% conceded {conc8:.1f} | vs bf0 win {wr0*100:.0f}% "
                      f"| score {score:+.3f} (best {best_score:+.3f}){tag}")
                hardest = (phase == "tracker" and lag == 0)   # self-play is open-ended -> no auto-stop
                if hardest and no_improve >= EARLY_STOP_K:
                    print(f">>> early stop: no improvement for {EARLY_STOP_K} evals at the hardest level.")
                    break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        save()
        print(f"saved {SAVE}  (best -> {BEST})")


if __name__ == "__main__":
    main()
