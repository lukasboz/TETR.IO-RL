# TETR.IO RL

A working local reinforcement learning project for modern block stacking, with an optional TETR.IO custom-room connector. Python/PyTorch trains a **Double DQN** that ranks legal placements. A standalone HTML replay lets you watch its decisions without running a server.

## Play against the trained bot

Open [the local versus arena](runs/arena.html) and choose **Start match**. It runs entirely offline with the learned model embedded. You can play against the DQN, watch it face the heuristic, or watch two DQN players exchange garbage. Bot speed is adjustable. See [arena controls and rules](arena/README.md).

```powershell
python -m tetrio_rl arena --checkpoint runs/initial/latest.pt --output runs/arena.html
Start-Process runs/arena.html
```

This is a standalone local arena. Integration with Tetr Online's upstream game remains pending source access; the portable browser and Rust model exports are ready under `portable/`.

## Start here

Python 3.10+ is required. Run these commands from the project directory:

```powershell
python -m pip install -e .

# Watch the fixed heuristic baseline immediately; no training needed.
python -m tetrio_rl replay --policy heuristic --max-pieces 500 --output runs/baseline.html
Start-Process runs/baseline.html

# Train a neural policy. This stops and saves after 100,000 placements.
python -m tetrio_rl train --steps 100000 --output runs/dqn

# Compare the trained policy against random and heuristic policies on identical seeds.
python -m tetrio_rl evaluate --checkpoint runs/dqn/latest.pt --compare --episodes 10 --max-pieces 500 --output runs/dqn/evaluation.json

# Watch the neural policy.
python -m tetrio_rl replay --policy dqn --checkpoint runs/dqn/latest.pt --output runs/dqn/replay.html
Start-Process runs/dqn/replay.html
```

This workspace also includes a completed 10,000-placement training run under `runs/initial/`, with 2,437 gradient updates. Its weights are a starting point for further training, not a competitive multiplayer model. Open `runs/initial/replay.html` to watch it. The evaluation report is `runs/initial/evaluation.json`. Generated runs and checkpoints are excluded from Git by default.

On five held-out seeds with a 500-piece limit, that checkpoint averaged **193.6 cleared lines**, versus **0.0** for random play and **198.8** for the fixed heuristic. The DQN and heuristic both reached the piece limit in all five episodes. These are local stacking results; they do not establish multiplayer strength.

--

Disclaimer: this is independent research software, not affiliated with or endorsed by TETR.IO.

This project will eventually be ported to TETR.IO once bot API applications open up again (hence the name).