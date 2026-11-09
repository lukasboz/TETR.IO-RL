"""Compare portable JavaScript inference with the actual PyTorch checkpoint."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from tetrio_rl.agent import AgentConfig, DQNAgent
from tetrio_rl.env import TetrisEnv
from tetrio_rl.export import export_model
from tetrio_rl.features import encode_candidates, heuristic_scores


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for cross-runtime parity checks")
class PortableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_features_scores_and_action_match_torch(self):
        agent = DQNAgent(AgentConfig(hidden=16), seed=321)
        env = TetrisEnv(seed=123)
        candidates = []
        for _ in range(16):
            moves = env.candidates()
            candidates.extend(moves[::5])
            env.step(int(heuristic_scores(encode_candidates(moves)).argmax()))
        # Include line clears and holes through real simulator states.
        features = encode_candidates(candidates)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "network.pt"
            model_path = Path(directory) / "network.json"
            fixture_path = Path(directory) / "fixture.json"
            agent.save(checkpoint)
            payload = export_model(checkpoint, model_path)
            self.assertEqual(payload["training_steps"], 0)
            fixture_path.write_text(json.dumps([{
                "board": c.board.tolist(), "lines": c.lines, "held": c.held,
                "next": c.next_pieces, "landing_height": c.landing_height,
            } for c in candidates]), encoding="utf-8")
            script = """
import fs from 'node:fs';
const {PlacementModel, encodeCandidate} = await import(process.argv[1]);
const model = new PlacementModel(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')));
const candidates = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const result = model.rankCandidates(candidates);
console.log(JSON.stringify({...result, features: candidates.map(c => [...encodeCandidate(c)])}));
"""
            result = subprocess.run(["node", "--input-type=module", "-e", script,
                                     (ROOT / "portable/model.mjs").as_uri(),
                                     str(model_path), str(fixture_path)],
                                    capture_output=True, text=True, timeout=30, check=True)
            javascript = json.loads(result.stdout)
            np.testing.assert_array_equal(features, np.asarray(javascript["features"], dtype=np.float32))
            np.testing.assert_allclose(agent.scores(features), javascript["scores"], rtol=1e-5, atol=1e-6)
            self.assertEqual(int(agent.scores(features).argmax()), javascript["index"])

    def test_arena_candidates_match_python_training_engine(self):
        env = TetrisEnv(seed=95)
        states = []
        for step in range(20):
            if step % 4 == 0:
                states.append((env.board.copy(), env.current, env.held, env.queue.copy(), env.candidates()))
            candidates = env.candidates()
            env.step(int(heuristic_scores(encode_candidates(candidates)).argmax()))
        env = TetrisEnv()
        env.current = env.held = "I"
        env.board[-4:] = 8
        env.board[-4:, 4] = 0
        states.append((env.board.copy(), env.current, env.held, env.queue.copy(), env.candidates()))
        fixtures, expected = [], []
        for board, current, held, queue, candidates in states:
            fixtures.append({"board": board.tolist(), "current": current, "held": held, "queue": queue})
            expected.append([{"board": c.board.tolist(), "lines": c.lines, "held": c.held,
                              "next": list(c.next_pieces[:3]), "landing_height": c.landing_height,
                              "keys": list(c.action.keys),
                              "placement": {"piece": c.action.piece, "rotation": c.action.rotation,
                                            "x": c.action.x, "y": c.action.y, "hold": c.action.hold}}
                             for c in candidates])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.json"
            path.write_text(json.dumps(fixtures), encoding="utf-8")
            script = """
import fs from 'node:fs';
const {Game} = await import(process.argv[1]);
const fixtures = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const output = fixtures.map(state => {
  const game = Object.assign(new Game(), state, {falling: {r: 0, x: 3, y: 2}, done: false, holdLocked: false});
  return game.candidates().map(c => ({...c, next: c.next.slice(0,3)}));
});
console.log(JSON.stringify(output));
"""
            result = subprocess.run(["node", "--input-type=module", "-e", script,
                                     (ROOT / "arena/dist/engine.mjs").as_uri(), str(path)],
                                    capture_output=True, text=True, timeout=30, check=True)
            self.assertEqual(json.loads(result.stdout), expected)


if __name__ == "__main__":
    unittest.main()
