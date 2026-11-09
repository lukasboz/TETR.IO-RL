"""Shared feature encoding for the simulator and external engine candidates."""

from __future__ import annotations

import numpy as np

from .env import Candidate, HEIGHT, PIECES, WIDTH

FEATURE_NAMES = (
    tuple(f"height_{i}" for i in range(WIDTH))
    + ("aggregate_height", "holes", "bumpiness", "max_height", "wells",
       "row_transitions", "column_transitions", "covered_holes", "lines", "landing_height")
    + tuple(f"hold_{p}" for p in "-" + PIECES)
    + tuple(f"next_{i}_{p}" for i in range(3) for p in PIECES)
)
FEATURE_DIM = len(FEATURE_NAMES)


def encode(board: np.ndarray, lines: int = 0, held: str | None = None,
           next_pieces: tuple[str, ...] = (), landing_height: float = 0) -> np.ndarray:
    occupied = np.asarray(board) != 0
    h, w = occupied.shape
    if w != WIDTH or h < 20:
        raise ValueError("Expected a top-first board with 10 columns and at least 20 rows")
    # Use a common 24-row window when a live engine supplies a larger spawn buffer.
    if h != HEIGHT:
        occupied = occupied[-HEIGHT:]
        if len(occupied) < HEIGHT:
            occupied = np.pad(occupied, ((HEIGHT - len(occupied), 0), (0, 0)))
    heights = np.where(occupied.any(axis=0), HEIGHT - occupied.argmax(axis=0), 0)
    covered = np.cumsum(occupied, axis=0)
    holes = (~occupied) & (covered > 0)
    padded = np.pad(heights, (1, 1), constant_values=HEIGHT)
    wells = np.maximum(0, np.minimum(padded[:-2], padded[2:]) - heights)
    horizontal = np.pad(occupied, ((0, 0), (1, 1)), constant_values=True)
    vertical = np.pad(occupied, ((1, 1), (0, 0)), constant_values=True)
    metrics = [heights.sum() / 240, holes.sum() / 240,
               np.abs(np.diff(heights)).sum() / 216, heights.max() / HEIGHT,
               wells.sum() / 240, np.count_nonzero(np.diff(horizontal, axis=1)) / 264,
               np.count_nonzero(np.diff(vertical, axis=0)) / 250,
               covered[holes].sum() / 2880, lines / 4, landing_height / HEIGHT]
    hold_onehot = [float(held == p) for p in (None, *PIECES)]
    next_onehot = [float(i < len(next_pieces) and next_pieces[i] == p)
                   for i in range(3) for p in PIECES]
    return np.asarray([*list(heights / HEIGHT), *metrics, *hold_onehot, *next_onehot], dtype=np.float32)


def encode_candidates(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([encode(c.board, c.lines, c.held, c.next_pieces, c.landing_height)
                     for c in candidates])


def heuristic_scores(features: np.ndarray) -> np.ndarray:
    """A fixed baseline/optional exploration teacher, independent of the Q network."""
    return (-0.510066 * features[:, 10] * 240
            - 0.760666 * features[:, 11] * 240
            - 0.184483 * features[:, 12] * 216
            + 0.760666 * features[:, 18] * 4)
