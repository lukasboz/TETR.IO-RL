"""Deterministic placement environment, with seven-bag, hold and SRS rotations.

Actions rotate and shift at spawn, then hard drop. Slides/tucks after dropping,
gravity timing, spins and multiplayer attack rules are intentionally excluded.
Rows are stored top first; the first four rows are the hidden spawn buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

WIDTH, HEIGHT, VISIBLE = 10, 24, 20
PIECES = "IJLOSTZ"
_SPAWN = {
    "I": ("....", "IIII", "....", "...."),
    "J": ("J..", "JJJ", "..."),
    "L": ("..L", "LLL", "..."),
    "O": (".OO.", ".OO.", "....", "...."),
    "S": (".SS", "SS.", "..."),
    "T": (".T.", "TTT", "..."),
    "Z": ("ZZ.", ".ZZ", "..."),
}


def _rotations(pattern: tuple[str, ...]) -> tuple:
    cells = np.array([[c != "." for c in row] for row in pattern])
    return tuple(tuple((int(x), int(y)) for y, x in np.argwhere(np.rot90(cells, -r)))
                 for r in range(4))


SHAPES = {p: _rotations(pattern) for p, pattern in _SPAWN.items()}
# SRS offsets are specified with positive Y upwards, converted at use time.
_JLSTZ = {
    (0, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (1, 2): ((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
    (2, 1): ((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
    (0, 3): ((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
}
_I = {
    (0, 1): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (1, 0): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
    (2, 1): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (2, 3): ((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
    (3, 0): ((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
    (0, 3): ((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
}


def fits(board: np.ndarray, piece: str, rotation: int, x: int, y: int) -> bool:
    return all(0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT
               and not board[y + dy, x + dx]
               for dx, dy in SHAPES[piece][rotation])


def rotate(board: np.ndarray, piece: str, rotation: int, x: int, y: int,
           direction: int = 1) -> tuple[int, int, int] | None:
    new = (rotation + direction) % 4
    if piece == "O":
        return rotation, x, y
    for dx, dy in (_I if piece == "I" else _JLSTZ)[rotation, new]:
        if fits(board, piece, new, x + dx, y - dy):
            return new, x + dx, y - dy
    return None


@dataclass(frozen=True)
class Action:
    piece: str
    rotation: int
    x: int
    y: int
    hold: bool
    keys: tuple[str, ...]


@dataclass
class Candidate:
    action: Action
    board: np.ndarray
    lines: int
    held: str | None
    next_pieces: tuple[str, ...]
    landing_height: float


@dataclass
class Step:
    reward: float
    terminated: bool
    truncated: bool
    lines: int


class TetrisEnv:
    def __init__(self, seed: int = 0, max_pieces: int = 2000,
                 garbage_every: int = 0, garbage_lines: int = 1):
        if max_pieces < 1 or garbage_every < 0 or not 1 <= garbage_lines < VISIBLE:
            raise ValueError("Invalid episode or garbage settings")
        self.rng = random.Random(seed)
        self.max_pieces = max_pieces
        self.garbage_every = garbage_every
        self.garbage_lines = garbage_lines
        self.reset(seed)

    def _refill(self) -> None:
        while len(self.queue) < 8:
            bag = list(PIECES)
            self.rng.shuffle(bag)
            self.queue.extend(bag)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.board = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.queue: list[str] = []
        self._refill()
        self.current = self.queue.pop(0)
        self.held: str | None = None
        self.pieces = self.lines = 0
        self.done = False
        self._candidates: list[Candidate] | None = None

    def candidates(self) -> list[Candidate]:
        if self.done:
            return []
        if self._candidates is not None:
            return self._candidates
        result: list[Candidate] = []
        choices = [(self.current, False, self.held, tuple(self.queue))]
        if self.held != self.current:
            choices.append((self.held or self.queue[0], True, self.current,
                            tuple(self.queue[1:] if self.held is None else self.queue)))
        # A blocked current spawn is topout, even if swapping might otherwise fit.
        if not fits(self.board, self.current, 0, 3, 2):
            self._candidates = []
            return []
        for piece, hold, held, queue in choices:
            if not fits(self.board, piece, 0, 3, 2):
                continue
            seen: set[tuple] = set()
            for turns in (0, 1, 2, -1):
                r, x, y = 0, 3, 2
                keys = ["hold"] if hold else []
                for _ in range(abs(turns)):
                    rotated = rotate(self.board, piece, r, x, y, 1 if turns > 0 else -1)
                    if rotated is None:
                        break
                    r, x, y = rotated
                    keys.append("rotateCW" if turns > 0 else "rotateCCW")
                else:
                    for shift in range(-5, 6):
                        tx, ty = x, y
                        direction = 1 if shift > 0 else -1
                        for _ in range(abs(shift)):
                            if not fits(self.board, piece, r, tx + direction, ty):
                                break
                            tx += direction
                        else:
                            while fits(self.board, piece, r, tx, ty + 1):
                                ty += 1
                            cells = tuple(sorted((tx + dx, ty + dy) for dx, dy in SHAPES[piece][r]))
                            if cells in seen:
                                continue
                            seen.add(cells)
                            board = self.board.copy()
                            for cx, cy in cells:
                                board[cy, cx] = PIECES.index(piece) + 1
                            full = np.all(board != 0, axis=1)
                            lines = int(full.sum())
                            if lines:
                                board = np.concatenate((np.zeros((lines, WIDTH), dtype=np.uint8), board[~full]))
                            moves = keys + ["moveRight" if direction > 0 else "moveLeft"] * abs(shift)
                            action = Action(piece, r, tx, ty, hold, tuple(moves + ["hardDrop"]))
                            result.append(Candidate(action, board, lines, held, queue,
                                                    HEIGHT - 1 - sum(cy for _, cy in cells) / 4))
        self._candidates = result
        return result

    def step(self, index: int) -> Step:
        candidates = self.candidates()
        if not candidates:
            raise RuntimeError("No legal moves; reset the environment")
        if not 0 <= index < len(candidates):
            raise IndexError("Action index is outside the candidate list")
        chosen = candidates[index]
        self.board = chosen.board.copy()
        self.held = chosen.held
        if chosen.action.hold and len(chosen.next_pieces) < len(self.queue):
            self.queue.pop(0)
        self._refill()
        self.current = self.queue.pop(0)
        self.pieces += 1
        self.lines += chosen.lines
        overflow = False
        if self.garbage_every and self.pieces % self.garbage_every == 0:
            n = self.garbage_lines
            overflow = bool(self.board[:n].any())
            garbage = np.full((n, WIDTH), 8, dtype=np.uint8)
            garbage[:, self.rng.randrange(WIDTH)] = 0
            self.board = np.concatenate((self.board[n:], garbage))
        self._candidates = None
        terminated = overflow or not fits(self.board, self.current, 0, 3, 2)
        truncated = not terminated and self.pieces >= self.max_pieces
        self.done = terminated or truncated
        # Base reward favors survival and clears. The trainer adds optional shaping.
        reward = 0.1 + (0.0, 1.0, 3.0, 5.0, 8.0)[chosen.lines]
        if terminated:
            reward -= 10.0
        return Step(reward, terminated, truncated, chosen.lines)

    def render(self) -> str:
        chars = "." + PIECES + "#"
        rows = ["|" + "".join(chars[int(c)] for c in row) + "|" for row in self.board[-VISIBLE:]]
        return "\n".join(rows) + f"\n+{'-' * WIDTH}+ lines={self.lines} pieces={self.pieces} hold={self.held or '-'} next={self.current}"
