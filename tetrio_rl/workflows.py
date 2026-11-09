"""Training, seeded evaluation and portable replay generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env import TetrisEnv
from .features import encode, encode_candidates, heuristic_scores


def potential(board: np.ndarray) -> float:
    f = encode(board)
    return float(-0.51 * f[10] * 240 - 0.76 * f[11] * 240 - 0.18 * f[12] * 216)


def train(args) -> None:
    torch.set_num_threads(args.threads)
    agent = DQNAgent.load(args.resume, args.device) if args.resume else DQNAgent(
        AgentConfig(batch_size=args.batch_size, learning_rate=args.learning_rate),
        seed=args.seed, device=args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "last_train_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    env = TetrisEnv(args.seed, args.max_pieces, args.garbage_every, args.garbage_lines)
    rng = random.Random(args.seed)
    start_steps = agent.steps
    started = time.monotonic()
    episode, episode_reward, losses = 1, 0.0, []
    fields = ["step", "episode", "pieces", "lines", "return", "epsilon", "loss", "seconds"]
    log_path = output / "train.csv"
    append = args.resume and log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not append:
            writer.writeheader()
        try:
            for local_step in range(args.steps):
                candidates = env.candidates()
                if not candidates:
                    env.reset()
                    candidates = env.candidates()
                features = encode_candidates(candidates)
                progress = min(1.0, agent.steps / max(1, args.epsilon_decay))
                epsilon = args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)
                # Optional heuristic behavior gathers useful early replay. Targets always
                # come from Double DQN, with no imitation loss or heuristic at inference.
                if agent.steps < args.teacher_steps and rng.random() >= epsilon:
                    index = int(heuristic_scores(features).argmax())
                else:
                    index = agent.choose(features, epsilon)
                old_potential = potential(env.board)
                result = env.step(index)
                # Time limits bootstrap from the real successor state. Topouts do not.
                if result.truncated:
                    env.done = False
                next_features = encode_candidates(env.candidates())
                terminal = result.terminated or not len(next_features)
                next_potential = 0.0 if terminal else potential(env.board)
                reward = result.reward + args.shaping * (agent.config.gamma * next_potential - old_potential)
                agent.remember(features[index], reward, next_features, terminal)
                episode_reward += reward
                if agent.steps % args.train_every == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                if terminal or result.truncated:
                    row = dict(step=agent.steps, episode=episode, pieces=env.pieces,
                               lines=env.lines, return_=episode_reward)
                    row["return"] = row.pop("return_")
                    row.update(epsilon=round(epsilon, 4),
                               loss=float(np.mean(losses)) if losses else "",
                               seconds=round(time.monotonic() - started, 2))
                    writer.writerow(row)
                    stream.flush()
                    if episode % args.log_every == 0:
                        print(f"step={agent.steps:>7} episode={episode:>5} lines={env.lines:>4} "
                              f"pieces={env.pieces:>4} epsilon={epsilon:.3f} "
                              f"loss={row['loss']} elapsed={row['seconds']}s", flush=True)
                    episode += 1
                    episode_reward, losses = 0.0, []
                    env.reset()
                if (local_step + 1) % args.save_every == 0:
                    agent.save(output / "latest.pt", {"seed": args.seed, "episodes": episode})
        except KeyboardInterrupt:
            print("Training interrupted; saving current weights.", flush=True)
        finally:
            agent.save(output / "latest.pt", {"seed": args.seed, "episodes": episode})
    print(f"Saved {output / 'latest.pt'} ({agent.steps - start_steps} new steps, "
          f"{agent.updates} updates, {time.monotonic() - started:.1f}s)", flush=True)


def run_episode(policy: str, seed: int, max_pieces: int, agent: DQNAgent | None = None,
                record: bool = False, garbage_every: int = 0, garbage_lines: int = 1) -> dict:
    env = TetrisEnv(seed, max_pieces, garbage_every, garbage_lines)
    rng = random.Random(seed)
    frames = [{"board": env.board.tolist(), "lines": 0, "pieces": 0,
               "current": env.current, "held": env.held, "queue": env.queue[:5]}] if record else []
    total_reward = 0.0
    topout = False
    while not env.done:
        candidates = env.candidates()
        if not candidates:
            topout = True
            break
        if policy == "random":
            index = rng.randrange(len(candidates))
        else:
            features = encode_candidates(candidates)
            if policy == "heuristic":
                index = int(heuristic_scores(features).argmax())
            elif policy == "dqn" and agent is not None:
                index = agent.choose(features)
            else:
                raise ValueError("DQN policy requires a checkpoint")
        action = candidates[index].action
        result = env.step(index)
        total_reward += result.reward
        topout = result.terminated
        if record:
            frames.append({"board": env.board.tolist(), "lines": env.lines, "pieces": env.pieces,
                           "current": env.current, "held": env.held, "queue": env.queue[:5],
                           "keys": list(action.keys), "clear": result.lines})
    return {"policy": policy, "seed": seed, "pieces": env.pieces, "lines": env.lines,
            "reward": round(total_reward, 4), "topout": topout, "frames": frames}


def evaluate(args) -> None:
    torch.set_num_threads(args.threads)
    agent = DQNAgent.load(args.checkpoint, args.device) if args.checkpoint else None
    policies = ["random", "heuristic"] + (["dqn"] if agent else []) if args.compare else [args.policy]
    all_results, summaries = [], []
    for policy in policies:
        results = [run_episode(policy, args.seed + i, args.max_pieces, agent,
                               garbage_every=args.garbage_every, garbage_lines=args.garbage_lines)
                   for i in range(args.episodes)]
        all_results.extend([{k: v for k, v in r.items() if k != "frames"} for r in results])
        summary = {"policy": policy, "episodes": len(results),
                   "mean_lines": round(float(np.mean([r["lines"] for r in results])), 2),
                   "mean_pieces": round(float(np.mean([r["pieces"] for r in results])), 2),
                   "topouts": sum(r["topout"] for r in results)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoint": args.checkpoint,
                                    "checkpoint_steps": agent.steps if agent else None,
                                    "max_pieces": args.max_pieces,
                                    "garbage_every": args.garbage_every,
                                    "garbage_lines": args.garbage_lines,
                                    "summary": summaries, "episodes": all_results}, indent=2), encoding="utf-8")


def replay(args) -> None:
    from .replay import write_html
    torch.set_num_threads(args.threads)
    agent = DQNAgent.load(args.checkpoint, args.device) if args.checkpoint else None
    result = run_episode(args.policy, args.seed, args.max_pieces, agent, record=True,
                         garbage_every=args.garbage_every, garbage_lines=args.garbage_lines)
    result["checkpoint"] = args.checkpoint
    result["checkpoint_steps"] = agent.steps if agent else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_html(output, result)
    print(f"Saved {output.resolve()} ({result['lines']} lines in {result['pieces']} pieces)")
