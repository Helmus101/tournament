# Pong Tournament

A clean, symmetric Pong environment with a self-play trainer and a fair
round-robin tournament. Two files of code, one model.

## Files

| File | What it is |
|---|---|
| `arena.py` | **Everything for running:** the symmetric `PongSym` environment, the policy network, the agent contract, and the tournament. Run it to hold a tournament. |
| `realpong.py` | Trainer — trains `realpong.pt` on `PongSym`. |
| `realpong.pt` | The trained weights. |
| `requirements.txt` | Dependencies. |

## Setup

```bash
pip install -r tournament/requirements.txt
source pongenv/bin/activate          # so `python` has torch/numpy
```

## Run a tournament

```bash
python tournament/arena.py                          # realpong.pt vs the scripted tracker
python tournament/arena.py realpong.pt bf random    # add reference opponents
python tournament/arena.py realpong.pt alice.pt     # any number of models
python tournament/arena.py --games 21               # games per pairing
```

## Add your own model

1. **Standard network** — drop a `.pt` file (a `state_dict` for `arena.Net`) in
   this folder and pass its name:
   ```bash
   python tournament/arena.py realpong.pt yourmodel.pt
   ```
2. **Custom architecture** — put an `Agent` class in your own `.py` file and pass
   `yourfile.py:yourweights.pt`:
   ```python
   class Agent:
       def __init__(self, weights_path=None): ...
       def reset(self): ...                 # start of each game
       def act(self, frame) -> int:         # frame: 80x80 float, own paddle on the RIGHT
           ...                               # return 2 (UP) or 3 (DOWN)
   ```
   ```bash
   python tournament/arena.py realpong.pt yourfile.py:yourweights.pt
   ```

## Train

```bash
python tournament/realpong.py --fresh        # start clean (recommended)
python tournament/realpong.py                # resume realpong.pt
python tournament/realpong.py --episodes 500 # stop after N
```

Trains as the right player (symmetric env → plays both sides); curriculum goes
`random` opponent → scripted `tracker` once it consistently wins.

## Environment (`arena.PongSym`)

Symmetric by construction, so swapping sides is fair. Full spec:

| Parameter | Value |
|---|---|
| Field | 80 × 80 grid (a frame flattens to 6400) |
| Paddle height | 16 px |
| Paddle speed | 3 px / step |
| Paddle columns | left x=3, right x=76 |
| Ball horizontal speed | 2 px / step |
| Ball vertical speed | up to ±2 px / step (set by serve + paddle hit angle) |
| Serve | ball at centre, random vertical speed, random left/right direction, **seeded** |
| Walls | top/bottom reflect the ball |
| Bounce | angle depends on where the ball hits the paddle (centre = flat, edge = steep) |
| Scoring | ball past a paddle → opponent scores; game is first to 21 |
| Truncation | 8000 steps if neither reaches 21 |
| Observation | 80 × 80 binary frame, **canonical** (each player sees its own paddle on the right) |
| Actions | `2` = UP, `3` = DOWN, `0` = NOOP |
| Reward | +1 score, −1 concede (per player, per point) |
