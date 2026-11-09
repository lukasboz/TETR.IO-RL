"""Double DQN over legal placement features, with replay and a target network."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .features import FEATURE_DIM, FEATURE_NAMES


@dataclass
class AgentConfig:
    hidden: int = 128
    gamma: float = 0.99
    learning_rate: float = 0.0003
    batch_size: int = 64
    replay_size: int = 20000
    learning_starts: int = 256
    target_every: int = 250


class QNetwork(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class DQNAgent:
    def __init__(self, config: AgentConfig | None = None, seed: int = 0,
                 device: str = "cpu"):
        self.config = config or AgentConfig()
        if self.config.batch_size < 1 or self.config.replay_size < self.config.batch_size:
            raise ValueError("Replay capacity must be at least the positive batch size")
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.online = QNetwork(self.config.hidden).to(self.device)
        self.target = QNetwork(self.config.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay: deque = deque(maxlen=self.config.replay_size)
        self.steps = 0
        self.updates = 0

    @torch.inference_mode()
    def scores(self, features: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.empty(0, dtype=np.float32)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return self.online(x).cpu().numpy()

    def choose(self, features: np.ndarray, epsilon: float = 0) -> int:
        if not len(features):
            raise ValueError("Cannot choose from an empty candidate list")
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(features))
        return int(self.scores(features).argmax())

    def remember(self, features: np.ndarray, reward: float, next_features: np.ndarray,
                 terminated: bool) -> None:
        self.replay.append((features.astype(np.float32, copy=True), float(reward),
                            next_features.astype(np.float16, copy=True), bool(terminated)))
        self.steps += 1

    def learn(self) -> float | None:
        cfg = self.config
        if len(self.replay) < max(cfg.learning_starts, cfg.batch_size):
            return None
        batch = self.rng.sample(list(self.replay), cfg.batch_size)
        max_actions = max(1, max(len(item[2]) for item in batch))
        successors = np.zeros((len(batch), max_actions, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((len(batch), max_actions), dtype=bool)
        for i, (_, _, next_features, terminated) in enumerate(batch):
            if not terminated and len(next_features):
                successors[i, :len(next_features)] = next_features
                mask[i, :len(next_features)] = True
        x = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        next_x = torch.as_tensor(successors, device=self.device)
        valid = torch.as_tensor(mask, device=self.device)
        rewards = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # Double DQN: select using online weights, evaluate using target weights.
            selection = self.online(next_x).masked_fill(~valid, -torch.inf).argmax(dim=1)
            values = self.target(next_x).gather(1, selection[:, None]).squeeze(1)
            values = torch.where(valid.any(dim=1), values, 0)
            targets = rewards + cfg.gamma * values
        loss = nn.functional.smooth_l1_loss(self.online(x), targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % cfg.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": 1, "feature_names": list(FEATURE_NAMES),
                   "config": asdict(self.config), "online": self.online.state_dict(),
                   "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "steps": self.steps, "updates": self.updates, "metadata": metadata or {}}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> DQNAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != 1 or checkpoint.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Checkpoint feature schema/version does not match this engine")
        agent = cls(AgentConfig(**checkpoint["config"]), device=device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.steps = checkpoint["steps"]
        agent.updates = checkpoint["updates"]
        return agent
