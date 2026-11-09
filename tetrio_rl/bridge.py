"""JSON-lines scoring worker. No network access or account credentials needed."""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from .agent import DQNAgent
from .features import encode, heuristic_scores


def score_request(request: dict, agent: DQNAgent | None = None) -> dict:
    if request.get("type") != "score":
        raise ValueError("Expected a score request")
    candidates = request.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("Expected between 1 and 256 candidates")
    encoded = []
    for candidate in candidates:
        board = np.asarray(candidate["board"])
        if board.ndim != 2 or board.shape[1] != 10 or not 20 <= board.shape[0] <= 80:
            raise ValueError("Invalid board dimensions")
        lines = candidate.get("lines", 0)
        if type(lines) is not int or not 0 <= lines <= 4:
            raise ValueError("Invalid line clear count")
        encoded.append(encode(board, lines, candidate.get("held"),
                              tuple(candidate.get("next", [])), candidate.get("landing_height", 0)))
    features = np.stack(encoded)
    scores = agent.scores(features) if agent else heuristic_scores(features)
    if not np.isfinite(scores).all():
        raise ValueError("Model produced a non-finite score")
    return {"id": request.get("id"), "index": int(scores.argmax()), "scores": scores.tolist()}


def serve(args) -> None:
    torch.set_num_threads(args.threads)
    agent = DQNAgent.load(args.checkpoint, args.device) if args.checkpoint else None
    print(json.dumps({"type": "ready", "policy": "dqn" if agent else "heuristic"}), flush=True)
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 2_000_000:
                raise ValueError("Request is too large")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be an object")
            response = score_request(request, agent)
        except (ValueError, KeyError, TypeError) as exc:
            response = {"id": request.get("id") if isinstance(request, dict) else None,
                        "error": str(exc)}
        print(json.dumps(response, allow_nan=False), flush=True)
