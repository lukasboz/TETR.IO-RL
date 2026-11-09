/** The 49-feature inference contract shared with tetrio_rl.features. No dependencies. */
export const PIECES = 'IJLOSTZ';
export const FEATURE_NAMES = [
  ...Array.from({length: 10}, (_, i) => `height_${i}`),
  'aggregate_height', 'holes', 'bumpiness', 'max_height', 'wells',
  'row_transitions', 'column_transitions', 'covered_holes', 'lines', 'landing_height',
  ...['-', ...PIECES].map(p => `hold_${p}`),
  ...Array.from({length: 3}, (_, i) => [...PIECES].map(p => `next_${i}_${p}`)).flat(),
];

export function encodeCandidate({board, lines = 0, held = null, next = [], landing_height = 0}) {
  if (!Array.isArray(board) || board.length < 20 || board.length > 80
      || board.some(row => !Array.isArray(row) || row.length !== 10
        || row.some(value => typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('Expected a top-first numeric board with 10 columns and 20–80 rows');
  }
  if (!Number.isInteger(lines) || lines < 0 || lines > 4 || !Number.isFinite(landing_height)) {
    throw new Error('Invalid placement line count or landing height');
  }
  if (held !== null && ![...PIECES].includes(held)) throw new Error('Invalid held piece');
  if (!Array.isArray(next) || next.some(p => ![...PIECES].includes(p))) throw new Error('Invalid next queue');
  const rows = board.slice(-24).map(row => row.map(value => value !== 0));
  while (rows.length < 24) rows.unshift(Array(10).fill(false));
  const heights = Array(10).fill(0), covered = Array(10).fill(0);
  let holes = 0, coveredHoles = 0, rowTransitions = 0, columnTransitions = 0;
  for (let y = 0; y < 24; y++) {
    let previous = true;
    for (let x = 0; x < 10; x++) {
      const occupied = rows[y][x];
      if (occupied) {
        if (heights[x] === 0) heights[x] = 24 - y;
        covered[x]++;
      } else if (covered[x] > 0) { holes++; coveredHoles += covered[x]; }
      if (previous !== occupied) rowTransitions++;
      previous = occupied;
    }
    if (previous !== true) rowTransitions++;
  }
  for (let x = 0; x < 10; x++) {
    let previous = true;
    for (let y = 0; y < 24; y++) {
      if (previous !== rows[y][x]) columnTransitions++;
      previous = rows[y][x];
    }
    if (previous !== true) columnTransitions++;
  }
  let bumpiness = 0, wells = 0;
  for (let x = 0; x < 10; x++) {
    if (x > 0) bumpiness += Math.abs(heights[x] - heights[x - 1]);
    wells += Math.max(0, Math.min(x > 0 ? heights[x - 1] : 24,
      x < 9 ? heights[x + 1] : 24) - heights[x]);
  }
  return Float32Array.from([
    ...heights.map(h => h / 24), heights.reduce((sum, h) => sum + h, 0) / 240,
    holes / 240, bumpiness / 216, Math.max(...heights) / 24, wells / 240,
    rowTransitions / 264, columnTransitions / 250, coveredHoles / 2880,
    lines / 4, landing_height / 24,
    ...[null, ...PIECES].map(p => Number(held === p)),
    ...Array.from({length: 3}, (_, i) => [...PIECES].map(p => Number(next[i] === p))).flat(),
  ]);
}

export function heuristicScore(features) {
  return -0.510066 * features[10] * 240 - 0.760666 * features[11] * 240
    - 0.184483 * features[12] * 216 + 0.760666 * features[18] * 4;
}

export class PlacementModel {
  constructor(payload) {
    if (payload?.format !== 'tetrio-rl-dqn-v1'
        || JSON.stringify(payload.feature_names) !== JSON.stringify(FEATURE_NAMES)) {
      throw new Error('Unknown model format or feature schema');
    }
    let width = FEATURE_NAMES.length;
    if (!Array.isArray(payload.layers) || payload.layers.length === 0) throw new Error('Missing model layers');
    this.layers = payload.layers.map(layer => {
      if (layer.type === 'relu') return {type: 'relu'};
      if (layer.type !== 'linear' || !Array.isArray(layer.weight) || !layer.weight.length
          || !Array.isArray(layer.bias) || layer.bias.length !== layer.weight.length
          || layer.weight.some(row => !Array.isArray(row) || row.length !== width
            || row.some(value => !Number.isFinite(value)))
          || layer.bias.some(value => !Number.isFinite(value))) throw new Error('Invalid model layer');
      width = layer.bias.length;
      return {type: 'linear', weight: layer.weight.map(row => Float32Array.from(row)),
        bias: Float32Array.from(layer.bias)};
    });
    if (width !== 1) throw new Error('Placement model must produce one score');
    this.trainingSteps = payload.training_steps;
    this.checkpointSha256 = payload.checkpoint_sha256;
  }

  scoreFeatures(features) {
    if (features.length !== FEATURE_NAMES.length || Array.from(features).some(v => !Number.isFinite(v))) {
      throw new Error('Invalid feature vector');
    }
    let values = features;
    for (const layer of this.layers) {
      if (layer.type === 'relu') {
        values = Float32Array.from(values, value => Math.max(0, value));
      } else {
        values = Float32Array.from(layer.weight, (row, i) => {
          let value = layer.bias[i];
          for (let j = 0; j < row.length; j++) value += row[j] * values[j];
          return value;
        });
      }
    }
    if (!Number.isFinite(values[0])) throw new Error('Model returned a non-finite score');
    return values[0];
  }

  rankCandidates(candidates) {
    if (!Array.isArray(candidates) || candidates.length === 0) throw new Error('No legal candidates');
    const scores = candidates.map(candidate => this.scoreFeatures(encodeCandidate(candidate)));
    let index = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[index]) index = i;
    return {index, scores};
  }
}
