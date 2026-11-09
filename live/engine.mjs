/** Triangle's engine uses bottom-first rows; Python uses top-first rows. */
export function boardForPython(state) {
  if (!Array.isArray(state) || state.length < 20 || state.some(row => row.length !== 10)) {
    throw new Error('This model requires a 10-column board with at least 20 rows');
  }
  const rows = state.slice(0, 24).map(row => row.map(cell => cell == null || cell === false || cell === 0 ? 0 : 1));
  while (rows.length < 24) rows.push(Array(10).fill(0));
  return rows.reverse();
}

export function requireBotAccount(user) {
  if (user?.role !== 'bot') throw new Error('Custom-room play requires an approved TETR.IO bot account (role=bot).');
}

function landingHeight(engine) {
  let cells = engine.falling.absoluteBlocks.map(([x, y]) => [x, y]);
  const board = engine.board.state;
  const legal = blocks => blocks.every(([x, y]) => x >= 0 && x < 10 && y >= 0
    && (y >= board.length || board[y][x] == null || board[y][x] === false || board[y][x] === 0));
  if (!legal(cells)) return null;
  while (legal(cells.map(([x, y]) => [x, y - 1]))) cells = cells.map(([x, y]) => [x, y - 1]);
  return cells.reduce((sum, [, y]) => sum + y, 0) / cells.length;
}

/** Explore only a disposable engine clone, never the network-controlled engine. */
export function enumeratePlacements(clone, snapshot) {
  const candidates = [], seen = new Set();
  for (const hold of [false, true]) {
    if (hold && (snapshot.holdLocked || !clone.misc.allowed.hold)) continue;
    for (const rotation of [[], ['rotateCW'], ['rotateCW', 'rotateCW'], ['rotateCCW']]) {
      for (let shift = -5; shift <= 5; shift++) {
        clone.fromSnapshot(snapshot);
        const keys = [...(hold ? ['hold'] : []), ...rotation,
          ...Array(Math.abs(shift)).fill(shift < 0 ? 'moveLeft' : 'moveRight')];
        if (keys.some(key => !clone.press(key))) continue;
        const height = landingHeight(clone);
        if (height === null) continue;
        const result = clone.press('hardDrop');
        if (!result || result.topout) continue;
        const candidate = {
          board: boardForPython(clone.board.state), lines: result.lines,
          held: clone.held == null ? null : String(clone.held).toUpperCase(),
          next: [clone.falling.symbol, ...clone.queue.slice(0, 2)].map(p => String(p).toUpperCase()),
          landing_height: height,
        };
        const signature = JSON.stringify(candidate);
        if (seen.has(signature)) continue;
        seen.add(signature);
        candidates.push({ ...candidate, keys: [...keys, 'hardDrop'] });
      }
    }
  }
  return candidates;
}

export function keyEvents(keys, frame) {
  // Same subframe preserves action ordering without introducing extra gravity.
  return keys.flatMap(key => ['keydown', 'keyup'].map(type => ({
    frame, type, data: { key, subframe: 0 },
  })));
}
