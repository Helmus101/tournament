# Pong Tournament

This folder runs a fair tournament between `realpong` and a selected opponent.
The needed `new_v3` model files are copied into this folder:

- `realpong.py`
- `realpong.pt`
- `karpathy_pong.py`
- `karpathy_pong.pt`
- `agent_ale.py`

Available opponents:

- `ball_follower`, a scripted paddle that tracks the ball
- `karpathy_pong` using `karpathy_pong.pt`

The match alternates sides every game because Atari Pong gives the two paddles
different starting conditions. The winner is decided by total points across all
games.

## Run

From the project root:

```bash
pongenv/bin/python tournament/run_tournament.py
```

By default this runs `realpong` against `ball_follower`.

Use a different number of games:

```bash
pongenv/bin/python tournament/run_tournament.py --games 10
```

Run against `karpathy_pong` instead:

```bash
pongenv/bin/python tournament/run_tournament.py --opponent karpathy_pong
```

Save a custom result file:

```bash
pongenv/bin/python tournament/run_tournament.py --output tournament/results/my_match.json
```

## Requirements

The runner expects the same dependencies as `new_v3`:

```bash
pip install -r tournament/requirements.txt
AutoROM --accept-license
```
