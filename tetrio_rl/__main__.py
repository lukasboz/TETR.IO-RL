from __future__ import annotations

import argparse


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and watch a TETR.IO-style RL agent")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a placement Double DQN")
    training.add_argument("--steps", type=positive, default=100000)
    training.add_argument("--output", default="runs/dqn")
    training.add_argument("--resume")
    training.add_argument("--batch-size", type=positive, default=64)
    training.add_argument("--learning-rate", type=float, default=0.0003)
    training.add_argument("--train-every", type=positive, default=4)
    training.add_argument("--save-every", type=positive, default=5000)
    training.add_argument("--log-every", type=positive, default=10)
    training.add_argument("--epsilon-start", type=float, default=0.2)
    training.add_argument("--epsilon-end", type=float, default=0.02)
    training.add_argument("--epsilon-decay", type=positive, default=50000)
    training.add_argument("--teacher-steps", type=int, default=2000,
                          help="Use heuristic exploration before this global step (0 disables)")
    training.add_argument("--shaping", type=float, default=0.1)
    evaluation = commands.add_parser("evaluate", help="Evaluate on fixed episode seeds")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--policy", choices=["dqn", "heuristic", "random"], default="heuristic")
    evaluation.add_argument("--compare", action="store_true")
    evaluation.add_argument("--episodes", type=positive, default=10)
    evaluation.add_argument("--output")
    playback = commands.add_parser("replay", help="Export a standalone interactive HTML replay")
    playback.add_argument("--checkpoint")
    playback.add_argument("--policy", choices=["dqn", "heuristic", "random"], default="heuristic")
    playback.add_argument("--output", default="runs/replay.html")
    bridge = commands.add_parser("bridge", help="JSON-lines inference worker for the live connector")
    bridge.add_argument("--checkpoint")
    exporter = commands.add_parser("export", help="Export a checkpoint for browser/native inference")
    exporter.add_argument("--checkpoint", required=True)
    exporter.add_argument("--output", default="portable/model.json")
    exporter.add_argument("--format", choices=["json", "rust"], default="json")
    arena = commands.add_parser("arena", help="Build a playable offline human-versus-DQN arena")
    arena.add_argument("--checkpoint", default="runs/initial/latest.pt")
    arena.add_argument("--output", default="runs/arena.html")
    for sub in (training, evaluation, playback):
        sub.add_argument("--seed", type=int, default=0 if sub is training else 10000)
        sub.add_argument("--max-pieces", type=positive, default=2000)
        sub.add_argument("--garbage-every", type=int, default=0)
        sub.add_argument("--garbage-lines", type=positive, default=1)
    for sub in (training, evaluation, playback, bridge):
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--threads", type=positive, default=1)
    args = parser.parse_args()
    if args.command == "export":
        from .export import run
        run(args)
        return
    if args.command == "arena":
        from .arena import run
        run(args)
        return
    if getattr(args, "policy", "") == "dqn" and not args.checkpoint:
        parser.error("--policy dqn requires --checkpoint")
    if args.command == "train":
        if not (0 <= args.epsilon_end <= args.epsilon_start <= 1):
            parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
        if args.learning_rate <= 0 or args.teacher_steps < 0 or args.shaping < 0:
            parser.error("learning rate must be positive; teacher steps and shaping must be nonnegative")
    from . import bridge as worker
    from . import workflows
    {"train": workflows.train, "evaluate": workflows.evaluate,
     "replay": workflows.replay, "bridge": worker.serve}[args.command](args)


if __name__ == "__main__":
    main()
