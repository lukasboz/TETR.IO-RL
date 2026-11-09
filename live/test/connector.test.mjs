import test from 'node:test';
import assert from 'node:assert/strict';
import { boardForPython, enumeratePlacements, keyEvents, requireBotAccount } from '../engine.mjs';
import { ScoringWorker } from '../worker.mjs';

test('bottom-first engine rows become a padded top-first model board', () => {
  const board = Array.from({ length: 40 }, () => Array(10).fill(null));
  board[0][3] = 't';
  board[23][5] = 'gb';
  const encoded = boardForPython(board);
  assert.equal(encoded.length, 24);
  assert.equal(encoded[23][3], 1);
  assert.equal(encoded[0][5], 1);
  assert.equal(board[0][3], 't');
  assert.throws(() => boardForPython([[0]]));
});

test('account guard accepts only a bot role', () => {
  assert.doesNotThrow(() => requireBotAccount({ role: 'bot' }));
  for (const role of ['user', 'admin', undefined]) assert.throws(() => requireBotAccount({ role }));
});

test('each input is released before a repeated key and events stay ordered', () => {
  const events = keyEvents(['rotateCW', 'rotateCW', 'hardDrop'], 17);
  assert.equal(events.length, 6);
  assert.deepEqual(events.map(e => e.type), ['keydown', 'keyup', 'keydown', 'keyup', 'keydown', 'keyup']);
  assert.ok(events.every(e => e.frame === 17 && e.data.subframe === 0));
});

// Minimal O-piece engine double: verifies enumeration and clone isolation,
// not compatibility with an actual TETR.IO game session.
class FakeEngine {
  constructor() {
    this.misc = { allowed: { hold: false } };
    this.board = { state: Array.from({ length: 24 }, () => Array(10).fill(null)) };
    this.held = null;
    this.queue = ['o', 'o', 'o'];
    this.x = 4;
    this.falling = { symbol: 'o', absoluteBlocks: [] };
  }
  fromSnapshot(snapshot) {
    this.board.state = structuredClone(snapshot.board);
    this.x = 4;
    this.falling.absoluteBlocks = [[4, 22], [5, 22], [4, 21], [5, 21]];
  }
  press(key) {
    if (key.startsWith('rotate')) return false;
    if (key.startsWith('move')) {
      const dx = key === 'moveRight' ? 1 : -1;
      if (this.x + dx < 0 || this.x + dx > 8) return false;
      this.x += dx;
      this.falling.absoluteBlocks = this.falling.absoluteBlocks.map(([x, y]) => [x + dx, y]);
      return true;
    }
    if (key === 'hardDrop') {
      for (const y of [0, 1]) for (const x of [this.x, this.x + 1]) this.board.state[y][x] = 'o';
      return { topout: false, lines: 0 };
    }
    return false;
  }
}

test('enumeration keeps the source snapshot intact and produces legal drop sequences', () => {
  const clone = new FakeEngine();
  const snapshot = { board: structuredClone(clone.board.state), holdLocked: false };
  const original = JSON.stringify(snapshot);
  const candidates = enumeratePlacements(clone, snapshot);
  assert.equal(candidates.length, 9);
  assert.equal(JSON.stringify(snapshot), original);
  for (const candidate of candidates) {
    assert.equal(candidate.keys.at(-1), 'hardDrop');
    assert.equal(candidate.board.flat().reduce((a, b) => a + b, 0), 4);
    assert.equal(candidate.landing_height, 0.5);
  }
});

test('real Python worker scores concurrent requests and surfaces invalid input', async () => {
  const worker = new ScoringWorker();
  try {
    const ready = await worker.ready;
    assert.equal(ready.policy, 'heuristic');
    const board = Array.from({ length: 24 }, () => Array(10).fill(0));
    const candidates = [{ board, lines: 0 }, { board, lines: 4 }];
    const responses = await Promise.all([worker.score(candidates), worker.score(candidates)]);
    assert.ok(responses.every(r => r.index === 1));
    assert.notEqual(responses[0].id, responses[1].id);
    await assert.rejects(worker.score([]), /candidates/);
  } finally {
    worker.close();
  }
});
