# Pong Tournament

A clean, symmetric Pong environment with a self-play trainer and a fair
round-robin tournament.

## Files

| File | What it is |
|---|---|
| `arena.py` | The symmetric `PongSym` environment + the tournament runner. **Architecture-agnostic** — it never assumes a network; every model brings its own code. Built-in opponents: `bf` (tracker), `random`. |
| `realpong.py` | The realpong **model** (`Net` + `Agent` class) **and** its trainer, in one file. The tournament imports it for the `Agent`; running it directly trains. |
| `realpong.pt` | realpong's trained weights. |
| `submission_template.py` | Copy-paste starting point for a competitor's entry (a working `Agent`). |
| `requirements.txt` | Dependencies. |

## Setup

```bash
pip install -r tournament/requirements.txt
source pongenv/bin/activate          # so `python` has torch/numpy
```

## Run a tournament

```bash
python tournament/arena.py                                          # realpong vs the tracker (default)
python tournament/arena.py realpong.py:realpong.pt alice.py:alice.pt bob.py:bob.pt   # 3 entrants
python tournament/arena.py realpong.py:realpong.pt bf random        # add reference opponents
python tournament/arena.py --best-of 5                              # longer sets (default 3)
python tournament/arena.py --headless                               # text only, no window
```

### Format
- **Round-robin:** every entrant plays every other.
- Each pairing is a **best-of-3 set**, each game to **21 points**.
- **Winning the set = 1 tournament point.**
- Final ranking is by **points**. If two (or more) are tied, they play a single
  **game to 21** to break it.

A **visual window opens by default** (green = right paddle, orange = left, yellow
= ball). Up/Down change speed, Esc quits. Use `--headless` to run text-only (or
if there's no display, it falls back to text automatically).

Each model is given as `code.py:weights.pt`. `bf` (scripted tracker) and
`random` are the only built-in opponents and need no files.

## Add your own model

The arena assumes **no architecture** — every model brings its own code, so any
design (MLP, CNN, anything) can compete. A model = **two files**: a `.py` with
your `Agent` class (your original model code) and a `.pt` of weights.

```python
# yourmodel.py
class Agent:
    def __init__(self, weights_path=None):   # build YOUR network, load the weights
        ...
    def reset(self): ...                      # called at the start of each game
    def act(self, frame) -> int:              # frame: 80x80 float, own paddle on the RIGHT
        ...                                    # return 2 (UP) or 3 (DOWN)
```

```bash
python tournament/arena.py realpong.py:realpong.pt yourmodel.py:yourmodel.pt
```

`realpong.py` is a complete worked example of this contract (its `Agent` class is
near the top). A bare `.pt` is **not** enough on its own — weights have no
architecture, so you must ship the `.py` too.

## Train

```bash
python tournament/realpong.py --fresh        # start clean (recommended)
python tournament/realpong.py                # resume realpong.pt
python tournament/realpong.py --episodes 500 # stop after N
```

Trains as the right player (symmetric env → plays both sides). Two curricula run
together:
- **opponent:** `random` → scripted `tracker` only after a **98% win rate** vs random
- **match length:** episodes `<1000` are 5-point matches, `1000–4999` are
  10-point, `≥5000` are full **21-point official** matches

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
