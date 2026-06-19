""" realpong.py  --  Karpathy policy-gradient (the original algorithm) with an
    IMPROVED training stack, on pong_v3. Trains agent_ale.PolicyNet -> realpong.pt
    which the arena loads directly:  python arena_ale.py realpong.pt bf

    Follows Karpathy's "Pong from pixels":
      * input = difference of two preprocessed 80x80 frames
      * sample UP/DOWN from the policy head's P(UP)
      * discount rewards, RESET at each point boundary (Pong-specific)

    Improved training stack (what makes it actually learn, and faster):
      1. VALUE BASELINE  -> advantage = discounted_return - V(s)  (agent_ale's
         value head); far less gradient variance than raw returns.
      2. OPPONENT CURRICULUM -> bootstrap vs a RANDOM opponent (beatable, gives a
         signal) then auto-graduate to the strong ball_follower. Avoids the
         "never scores vs a too-strong opponent -> stuck at -21" trap.
      3. Adam optimizer, batched updates, resume from checkpoint.

    Run:  python realpong.py        (Ctrl-C to stop; resumes realpong.pt)
"""

import os
import numpy as np
import torch
from pettingzoo.atari import pong_v3

from agent_ale import PolicyNet, preprocess, ball_follower_action, UP, DOWN, NOOP, D, DEVICE

# ── hyperparameters (Karpathy core + stack tuning) ──────────────────────────────
batch_size    = 10
learning_rate = 1e-3
gamma         = 0.99
value_coef    = 0.5
SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realpong.pt")
SEED = 1


def discount_rewards(r):
    out = np.zeros_like(r, dtype=np.float64)
    add = 0.0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            add = 0.0                       # reset at each point (Pong-specific)
        add = add * gamma + r[t]
        out[t] = add
    return out


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    net = PolicyNet().to(DEVICE)
    if os.path.exists(SAVE):
        ck = torch.load(SAVE, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        print(f"resumed {os.path.basename(SAVE)}")
    else:
        print(f"fresh PolicyNet -> {os.path.basename(SAVE)}")
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    env = pong_v3.parallel_env(max_cycles=8000)
    opp_mode = "random"                     # curriculum starts beatable
    running = None
    ep = 0
    opt.zero_grad()
    print("training realpong on pong_v3 (RIGHT) -- opponent: RANDOM (bootstrap) -- Ctrl-C to stop\n")
    try:
        while True:
            obs, _ = env.reset(seed=SEED + ep)
            prev = None
            logps, values, rewards = [], [], []
            bf_last = None
            sa = sb = 0                              # cap the game at 21 (pong_v3 won't on its own)
            while env.agents and max(sa, sb) < 21:
                frame = obs["first_0"]
                cur = preprocess(frame)
                diff = cur - prev if prev is not None else np.zeros(D, np.float32)
                prev = cur
                prob, value = net(torch.from_numpy(diff).unsqueeze(0).to(DEVICE))
                prob = prob.squeeze(0)
                up = torch.rand((), device=DEVICE) < prob
                action = UP if up.item() else DOWN
                logps.append(torch.log((prob if up else 1 - prob) + 1e-8))
                values.append(value.squeeze(0))
                if opp_mode == "follower":
                    a_left, bf_last = ball_follower_action(frame, "left", bf_last)
                else:
                    a_left = int(rng.choice([UP, DOWN, NOOP]))
                obs, rew, term, trunc, _ = env.step({"first_0": action, "second_0": a_left})
                r = rew.get("first_0", 0.0)
                rewards.append(r)
                if r > 0: sa += 1
                elif r < 0: sb += 1

            # ── policy gradient with value baseline ──
            returns = torch.tensor(discount_rewards(np.array(rewards)), dtype=torch.float32, device=DEVICE)
            values_t = torch.stack(values)
            adv = returns - values_t.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(torch.stack(logps) * adv).sum()
            value_loss = value_coef * (values_t - returns).pow(2).mean()
            (policy_loss + value_loss).backward()

            ep += 1
            if ep % batch_size == 0:
                opt.step(); opt.zero_grad()

            reward_sum = float(sum(rewards))
            running = reward_sum if running is None else running * 0.99 + reward_sum * 0.01
            print(f"episode {ep:5d} | reward {reward_sum:+5.0f} | running {running:+6.2f} | opp {opp_mode}")
            if opp_mode == "random" and running is not None and running > 10:
                opp_mode = "follower"
                print(">>> curriculum: graduating to ball_follower (strong)")
            if ep % 50 == 0:
                torch.save(net.state_dict(), SAVE)
    except KeyboardInterrupt:
        print("\nstopped")
    torch.save(net.state_dict(), SAVE)
    print(f"saved {SAVE}")


if __name__ == "__main__":
    main()
