# lamine — tournament entry

A custom CNN agent for the Pong arena.

| file | what it is |
|------|------------|
| `newfolder.py` | the model + the `Agent` class (self-contained: `numpy` + `torch`) |
| `newfolder_trained_best.pt` | the trained weights |

## Run

```bash
python arena.py submissions/lamine/newfolder.py:submissions/lamine/newfolder_trained_best.pt realpong.py:realpong.pt bf
```

Honors the contract — `Agent.__init__(weights_path)`, `Agent.reset()`, `Agent.act(frame) -> 2/3`. Nothing outside this folder is touched.

## Results

Current arena, round-robin, best of 3 to 21:

| opponent | result |
|----------|--------|
| `realpong` | **21–2, 21–4** (set won) |
| `bf` (tracker) | **21–0, 21–0** (0 conceded) |
| round-robin | **🏆 CHAMPION** |
