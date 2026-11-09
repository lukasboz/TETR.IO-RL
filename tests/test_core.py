import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tetrio_rl.agent import AgentConfig, DQNAgent
from tetrio_rl.bridge import score_request
from tetrio_rl.env import HEIGHT, PIECES, SHAPES, TetrisEnv, fits, rotate
from tetrio_rl.features import FEATURE_DIM, encode, encode_candidates, heuristic_scores
from tetrio_rl.replay import write_html


class EnvironmentTests(unittest.TestCase):
    def test_each_seven_bag_contains_every_piece(self):
        env = TetrisEnv(seed=123)
        sequence = [env.current, *env.queue]
        while len(sequence) < 28:
            env.queue = []
            env._refill()
            sequence.extend(env.queue)
        # Initial queue contains two complete bags; newly refilled queues do too.
        for offset in range(0, 28, 7):
            self.assertEqual(set(sequence[offset:offset + 7]), set(PIECES))

    def test_empty_hold_consumes_one_extra_queue_piece(self):
        env = TetrisEnv(seed=0)
        current, queue = env.current, env.queue.copy()
        candidates = env.candidates()
        index = next(i for i, c in enumerate(candidates) if c.action.hold)
        self.assertEqual(candidates[index].action.piece, queue[0])
        env.step(index)
        self.assertEqual(env.held, current)
        self.assertEqual(env.current, queue[1])
        self.assertEqual(env.queue[:3], queue[2:5])
        self.assertEqual(env.pieces, 1)

    def test_existing_hold_does_not_consume_extra_queue_piece(self):
        env = TetrisEnv(seed=0)
        env.current, env.held = "T", "I"
        queue = env.queue.copy()
        index = next(i for i, c in enumerate(env.candidates()) if c.action.hold)
        env.step(index)
        self.assertEqual(env.held, "T")
        self.assertEqual(env.current, queue[0])
        self.assertEqual(env.queue[:3], queue[1:4])

    def test_vertical_i_clears_four_lines(self):
        env = TetrisEnv()
        env.current = env.held = "I"
        env.board[-4:] = 8
        env.board[-4:, 4] = 0
        index = next(i for i, c in enumerate(env.candidates()) if c.lines == 4)
        result = env.step(index)
        self.assertEqual(result.lines, 4)
        self.assertEqual(env.lines, 4)
        self.assertFalse(env.board.any())

    def test_spawn_collision_cannot_be_escaped_by_hold(self):
        env = TetrisEnv()
        env.board[2:6, 2:8] = 8
        self.assertEqual(env.candidates(), [])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_srs_wall_and_floor_kicks(self):
        board = np.zeros((HEIGHT, 10), dtype=np.uint8)
        self.assertTrue(fits(board, "I", 1, -2, 5))
        self.assertEqual(rotate(board, "I", 1, -2, 5), (2, 0, 5))
        self.assertTrue(fits(board, "T", 0, 3, 22))
        self.assertEqual(rotate(board, "T", 0, 3, 22), (1, 2, 21))

    def test_candidates_are_executable_and_conserve_cells(self):
        env = TetrisEnv(seed=19)
        for _ in range(35):
            candidates = env.candidates()
            before = int(np.count_nonzero(env.board))
            for c in candidates:
                r, x, y = 0, 3, 2
                for key in c.action.keys:
                    if key in ("rotateCW", "rotateCCW"):
                        rotated = rotate(env.board, c.action.piece, r, x, y,
                                         1 if key == "rotateCW" else -1)
                        self.assertIsNotNone(rotated)
                        r, x, y = rotated
                    elif key.startswith("move"):
                        x += 1 if key == "moveRight" else -1
                        self.assertTrue(fits(env.board, c.action.piece, r, x, y))
                    elif key == "hardDrop":
                        while fits(env.board, c.action.piece, r, x, y + 1):
                            y += 1
                self.assertEqual((r, x, y), (c.action.rotation, c.action.x, c.action.y))
                self.assertEqual(int(np.count_nonzero(c.board)), before + 4 - 10 * c.lines)
            env.step(int(heuristic_scores(encode_candidates(candidates)).argmax()))

    def test_garbage_and_time_limit(self):
        env = TetrisEnv(max_pieces=1, garbage_every=1, garbage_lines=2)
        result = env.step(0)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(int((env.board[-2:] == 8).sum()), 18)
        self.assertEqual(int(np.count_nonzero(env.board)), 22)
        self.assertEqual(env.candidates(), [])

    def test_seeded_trajectory_is_reproducible(self):
        first, second = TetrisEnv(17), TetrisEnv(17)
        for _ in range(30):
            f, s = first.candidates(), second.candidates()
            self.assertEqual([c.action for c in f], [c.action for c in s])
            index = int(heuristic_scores(encode_candidates(f)).argmax())
            self.assertEqual(first.step(index), second.step(index))
            np.testing.assert_array_equal(first.board, second.board)


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feature_holes_and_dimensions(self):
        board = np.zeros((24, 10), dtype=np.uint8)
        board[-3, 0] = 1
        f = encode(board, 2, "I", ("T", "O", "Z"))
        self.assertEqual(f.shape, (FEATURE_DIM,))
        self.assertAlmostEqual(float(f[11] * 240), 2)
        self.assertAlmostEqual(float(f[0] * 24), 3)
        self.assertTrue(np.isfinite(f).all())
        np.testing.assert_array_equal(encode(board[-20:]), encode(board))

    def test_terminal_and_bootstrapped_targets_and_weight_update(self):
        cfg = AgentConfig(hidden=8, gamma=0.5, batch_size=2, learning_starts=2, target_every=5)
        agent = DQNAgent(cfg)
        with torch.no_grad():
            for p in agent.online.parameters():
                p.zero_()
            for p in agent.target.parameters():
                p.zero_()
            agent.online.layers[-1].bias.fill_(4)
            agent.target.layers[-1].bias.fill_(2)
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        successors = np.zeros((2, FEATURE_DIM), dtype=np.float32)
        agent.remember(x, 1, successors, True)
        agent.remember(x, 1, successors, False)
        # Targets: terminal=1, nonterminal=1 + .5 * 2=2. Huber losses: 2.5 and 1.5.
        loss = agent.learn()
        self.assertAlmostEqual(loss, 2.0, places=5)
        self.assertLess(float(agent.online.layers[-1].bias.detach()), 4)
        self.assertEqual(agent.updates, 1)

    def test_checkpoint_preserves_scores_and_rejects_unknown_schema(self):
        agent = DQNAgent(AgentConfig(hidden=8))
        features = encode_candidates(TetrisEnv().candidates())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pt"
            agent.save(path)
            loaded = DQNAgent.load(path)
            np.testing.assert_array_equal(agent.scores(features), loaded.scores(features))
            payload = torch.load(path, weights_only=True)
            payload["feature_names"] = []
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                DQNAgent.load(path)

    def test_bridge_and_replay_artifact(self):
        board = np.zeros((24, 10), dtype=int).tolist()
        request = {"id": 4, "type": "score", "candidates": [
            {"board": board, "lines": 0}, {"board": board, "lines": 4}]}
        self.assertEqual(score_request(request)["index"], 1)
        self.assertEqual(score_request(request)["id"], 4)
        with self.assertRaises(ValueError):
            score_request({"type": "score", "candidates": []})
        result = {"policy": "random", "seed": 0, "lines": 0, "pieces": 0,
                  "topout": False, "frames": [{"board": board, "lines": 0, "pieces": 0,
                                                "current": "T", "held": None, "queue": []}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.html"
            write_html(path, result)
            html = path.read_text(encoding="utf-8")
            self.assertNotIn("__PAYLOAD__", html)
            self.assertIn(json.dumps(result, separators=(",", ":")), html)


if __name__ == "__main__":
    unittest.main()
