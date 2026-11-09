import test from 'node:test';
import assert from 'node:assert/strict';
import {encodeCandidate, FEATURE_NAMES, PlacementModel} from '../model.mjs';

const board = () => Array.from({length: 24}, () => Array(10).fill(0));

test('feature encoding normalizes 20-row and 40-row game boards', () => {
  const state = board();
  state[22][3] = 8;
  const canonical = encodeCandidate({board: state});
  assert.deepEqual(encodeCandidate({board: state.slice(-20)}), canonical);
  assert.deepEqual(encodeCandidate({board: [...board().slice(0, 16), ...state]}), canonical);
  assert.equal(canonical.length, FEATURE_NAMES.length);
});

test('malformed boards and placement data are rejected', () => {
  assert.throws(() => encodeCandidate({board: [[0]]}));
  assert.throws(() => encodeCandidate({board: board(), lines: 5}));
  assert.throws(() => encodeCandidate({board: board(), held: 'Q'}));
  assert.throws(() => encodeCandidate({board: board(), next: ['Q']}));
  const nonFinite = board();
  nonFinite[0][0] = NaN;
  assert.throws(() => encodeCandidate({board: nonFinite}));
});

test('model schema and layer dimensions are checked', () => {
  assert.throws(() => new PlacementModel({format: 'unknown'}));
  const payload = {format: 'tetrio-rl-dqn-v1', feature_names: FEATURE_NAMES, layers: [
    {type: 'linear', weight: [Array(49).fill(0)], bias: [0]},
  ]};
  const model = new PlacementModel(payload);
  assert.equal(model.rankCandidates([{board: board()}]).index, 0);
  assert.throws(() => model.rankCandidates([]));
  assert.throws(() => model.scoreFeatures([NaN]));
  payload.layers[0].weight[0].pop();
  assert.throws(() => new PlacementModel(payload));
});
