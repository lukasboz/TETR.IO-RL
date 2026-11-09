import {PIECES} from './model.mjs';

export const WIDTH = 10, HEIGHT = 24, VISIBLE = 20;
const PATTERNS = {
  I: ['....', 'IIII', '....', '....'], J: ['J..', 'JJJ', '...'],
  L: ['..L', 'LLL', '...'], O: ['.OO.', '.OO.', '....', '....'],
  S: ['.SS', 'SS.', '...'], T: ['.T.', 'TTT', '...'], Z: ['ZZ.', '.ZZ', '...'],
};
export const SHAPES = Object.fromEntries(Object.entries(PATTERNS).map(([piece, pattern]) => {
  let matrix = pattern.map(row => [...row].map(c => c !== '.'));
  const states = [];
  for (let r = 0; r < 4; r++) {
    states.push(matrix.flatMap((row, y) => row.flatMap((cell, x) => cell ? [[x, y]] : [])));
    matrix = matrix.map((row, y) => row.map((_, x) => matrix[matrix.length - 1 - x][y]));
  }
  return [piece, states];
}));
const JLSTZ_KICKS = {
  '0,1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
  '1,0': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
  '1,2': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
  '2,1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
  '2,3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
  '3,2': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
  '3,0': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
  '0,3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
};
const I_KICKS = {
  '0,1': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
  '1,0': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
  '1,2': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
  '2,1': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
  '2,3': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
  '3,2': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
  '3,0': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
  '0,3': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
};
export function emptyBoard() { return Array.from({length: HEIGHT}, () => Array(WIDTH).fill(0)); }

export function fits(board, piece, r, x, y) {
  return SHAPES[piece][r].every(([dx, dy]) => x + dx >= 0 && x + dx < WIDTH
    && y + dy >= 0 && y + dy < HEIGHT && board[y + dy][x + dx] === 0);
}

export class Game {
  constructor(seed = 1) {
    this.rngState = seed >>> 0;
    this.board = emptyBoard();
    this.queue = [];
    this.held = null;
    this.holdLocked = false;
    this.done = false;
    this.pieces = 0; this.lines = 0; this.sent = 0;
    this.combo = -1; this.b2b = false;
    this.fallElapsed = 0; this.lockElapsed = 0; this.lockResets = 0;
    this.lastRotation = false;
    this.spawn();
  }
  random() {
    this.rngState = (this.rngState + 0x6d2b79f5) >>> 0;
    let value = this.rngState;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  }
  refill() {
    while (this.queue.length < 8) {
      const bag = [...PIECES];
      for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(this.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
      this.queue.push(...bag);
    }
  }
  spawn(symbol = null) {
    if (this.done) return;
    if (symbol === null) { this.refill(); symbol = this.queue.shift(); this.holdLocked = false; }
    this.current = symbol;
    this.falling = {r: 0, x: 3, y: 2};
    this.fallElapsed = 0; this.lockElapsed = 0; this.lockResets = 0;
    this.lastRotation = false;
    if (!this.fits()) this.done = true;
  }
  fits(r = this.falling.r, x = this.falling.x, y = this.falling.y) {
    return fits(this.board, this.current, r, x, y);
  }
  clone() {
    const copy = Object.assign(Object.create(Game.prototype), this);
    copy.board = this.board.map(row => [...row]);
    copy.queue = [...this.queue];
    copy.falling = this.falling ? {...this.falling} : null;
    return copy;
  }
  cells(falling = this.falling) {
    if (!falling) return [];
    return SHAPES[this.current][falling.r].map(([dx, dy]) => [falling.x + dx, falling.y + dy]);
  }
  ghost() {
    if (!this.falling || this.done) return null;
    const falling = {...this.falling};
    while (this.fits(falling.r, falling.x, falling.y + 1)) falling.y++;
    return falling;
  }
  resetLock() {
    if (this.lockResets < 15) { this.lockElapsed = 0; this.lockResets++; }
  }
  shift(direction) {
    if (this.done || !this.falling || !this.fits(this.falling.r, this.falling.x + direction)) return false;
    this.falling.x += direction;
    this.lastRotation = false;
    this.resetLock();
    return true;
  }
  rotate(direction) {
    if (this.done || !this.falling) return false;
    if (this.current === 'O') return true;
    const {r, x, y} = this.falling, next = (r + direction + 4) % 4;
    for (const [dx, dy] of (this.current === 'I' ? I_KICKS : JLSTZ_KICKS)[`${r},${next}`]) {
      if (this.fits(next, x + dx, y - dy)) {
        this.falling = {r: next, x: x + dx, y: y - dy};
        this.lastRotation = true;
        this.resetLock();
        return true;
      }
    }
    return false;
  }
  rotate180() {
    if (!this.falling) return false;
    const original = {...this.falling}, rotation = this.lastRotation;
    const lockElapsed = this.lockElapsed, lockResets = this.lockResets;
    if (this.rotate(1) && this.rotate(1)) return true;
    Object.assign(this, {falling: original, lastRotation: rotation, lockElapsed, lockResets});
    return false;
  }
  hold() {
    if (this.done || !this.falling || this.holdLocked) return false;
    const piece = this.current, held = this.held;
    this.held = piece;
    this.spawn(held);
    this.holdLocked = true;
    return true;
  }
  down() {
    if (this.done || !this.falling || !this.fits(this.falling.r, this.falling.x, this.falling.y + 1)) return false;
    this.falling.y++;
    this.lockElapsed = 0;
    return true;
  }
  hardDrop() {
    if (this.done || !this.falling) return null;
    const originalY = this.falling.y;
    this.falling = this.ghost();
    if (this.falling.y !== originalY) this.lastRotation = false;
    return this.lock();
  }
  press(key) {
    switch (key) {
      case 'moveLeft': return this.shift(-1);
      case 'moveRight': return this.shift(1);
      case 'rotateCW': return this.rotate(1);
      case 'rotateCCW': return this.rotate(-1);
      case 'rotate180': return this.rotate180();
      case 'softDrop': return this.down();
      case 'hold': return this.hold();
      case 'hardDrop': return this.hardDrop();
      default: return false;
    }
  }
  tick(ms, gravity = 1) {
    if (!this.falling || this.done) return null;
    this.fallElapsed += ms;
    const interval = 1000 / gravity;
    while (this.fallElapsed >= interval) { this.fallElapsed -= interval; if (!this.down()) break; }
    if (!this.fits(this.falling.r, this.falling.x, this.falling.y + 1)) {
      this.lockElapsed += ms;
      if (this.lockElapsed >= 500) return this.lock();
    } else this.lockElapsed = 0;
    return null;
  }
  lock() {
    if (!this.falling || this.done) return null;
    const cells = this.cells(), {x, y} = this.falling;
    const filled = (cx, cy) => cx < 0 || cx >= WIDTH || cy < 0 || cy >= HEIGHT || this.board[cy][cx] !== 0;
    const corners = [[x, y], [x + 2, y], [x, y + 2], [x + 2, y + 2]];
    const spin = this.current === 'T' && this.lastRotation && corners.filter(([cx, cy]) => filled(cx, cy)).length >= 3;
    for (const [cx, cy] of cells) this.board[cy][cx] = PIECES.indexOf(this.current) + 1;
    const remaining = this.board.filter(row => row.some(cell => cell === 0));
    const lines = HEIGHT - remaining.length;
    this.board = [...Array.from({length: lines}, () => Array(WIDTH).fill(0)), ...remaining];
    this.lines += lines; this.pieces++;
    this.combo = lines ? this.combo + 1 : -1;
    const difficult = lines === 4 || (spin && lines > 0);
    let attack = spin ? [0, 2, 4, 6][lines] : [0, 0, 1, 2, 4][lines];
    if (difficult && this.b2b) attack++;
    if (lines) { attack += Math.min(4, Math.floor(this.combo / 2)); this.b2b = difficult; }
    const perfectClear = lines > 0 && this.board.every(row => row.every(cell => cell === 0));
    if (perfectClear) attack += 10;
    const result = {lines, attack, spin, perfectClear, combo: this.combo,
      landing_height: HEIGHT - 1 - cells.reduce((sum, [, cy]) => sum + cy, 0) / 4};
    this.falling = null;
    return result;
  }
  addGarbage(holes) {
    if (!holes.length) return;
    if (holes.length >= HEIGHT || this.board.slice(0, holes.length).some(row => row.some(Boolean))) this.done = true;
    this.board = this.board.slice(Math.min(HEIGHT, holes.length));
    this.board.push(...holes.slice(-HEIGHT).map(hole => Array.from({length: WIDTH}, (_, x) => x === hole ? 0 : 8)));
  }
  candidates() {
    if (this.done || !this.falling) return [];
    const candidates = [];
    for (const held of [false, true]) {
      if (held && (this.holdLocked || this.held === this.current)) continue;
      const seen = new Set();
      for (const turns of [0, 1, 2, -1]) for (let shift = -5; shift <= 5; shift++) {
        const copy = this.clone(), keys = [];
        if (held) keys.push('hold');
        keys.push(...Array(Math.abs(turns)).fill(turns < 0 ? 'rotateCCW' : 'rotateCW'));
        keys.push(...Array(Math.abs(shift)).fill(shift < 0 ? 'moveLeft' : 'moveRight'));
        if (keys.some(key => !copy.press(key)) || copy.done) continue;
        const ghost = copy.ghost();
        const signature = JSON.stringify(copy.cells(ghost).sort((a, b) => a[0] - b[0] || a[1] - b[1]));
        if (seen.has(signature)) continue;
        seen.add(signature);
        const placement = {piece: copy.current, rotation: copy.falling.r, x: ghost.x, y: ghost.y, hold: held};
        const result = copy.hardDrop();
        candidates.push({board: copy.board, lines: result.lines, held: copy.held,
          next: copy.queue.slice(), landing_height: result.landing_height,
          keys: [...keys, 'hardDrop'], placement});
      }
    }
    return candidates;
  }
}

export function cancelIncoming(packets, attack) {
  let remaining = attack;
  while (remaining > 0 && packets.length) {
    const cancelled = Math.min(remaining, packets[0].holes.length);
    packets[0].holes.splice(0, cancelled);
    remaining -= cancelled;
    if (!packets[0].holes.length) packets.shift();
  }
  return remaining;
}

export class Match {
  constructor({seed = 1, mode = 'play', pps = 1, gravity = 1, pickMove, onEvent = () => {}} = {}) {
    if (!['play', 'watch', 'selfplay'].includes(mode) || !(pps > 0) || !(gravity > 0)) throw new Error('Invalid match settings');
    if (typeof pickMove !== 'function') throw new Error('A move selector is required');
    this.seed = seed; this.mode = mode; this.pps = pps; this.gravity = gravity;
    this.pickMove = pickMove; this.onEvent = onEvent;
    this.time = 0; this.state = 'idle'; this.winner = null; this.roundTime = 0;
    this.seats = [
      {game: new Game(seed), controller: mode === 'play' ? 'human' : 'dqn', pending: [], nextBot: 0},
      {game: new Game(seed), controller: mode === 'watch' ? 'heuristic' : 'dqn', pending: [], nextBot: 0},
    ];
    this.garbageRng = new Game(seed ^ 0x1f123bb5);
  }
  start() { if (this.state === 'idle') { this.state = 'countdown'; this.countdown = 3000; } }
  pause() {
    if (['running', 'countdown'].includes(this.state)) { this.previousState = this.state; this.state = 'paused'; }
    else if (this.state === 'paused') this.state = this.previousState;
  }
  incoming(index) { return this.seats[index].pending.reduce((sum, packet) => sum + packet.holes.length, 0); }
  finish(index, result) {
    const seat = this.seats[index], opponent = this.seats[1 - index];
    const attack = cancelIncoming(seat.pending, result.attack);
    if (attack) {
      const hole = Math.floor(this.garbageRng.random() * WIDTH);
      opponent.pending.push({holes: Array(attack).fill(hole), readyAt: this.time + 700});
      seat.game.sent += attack;
    }
    if (!result.lines) {
      const holes = [];
      while (holes.length < 8 && seat.pending.length && seat.pending[0].readyAt <= this.time) {
        const packet = seat.pending[0];
        holes.push(...packet.holes.splice(0, 8 - holes.length));
        if (!packet.holes.length) seat.pending.shift();
      }
      seat.game.addGarbage(holes);
    }
    seat.game.spawn();
    this.onEvent({type: 'placement', seat: index, ...result, sent: attack});
    this.checkEnd();
  }
  checkEnd() {
    const lost = this.seats.map(seat => seat.game.done);
    if (lost.some(Boolean) && this.state !== 'ended') {
      this.winner = lost[0] && lost[1] ? -1 : lost[0] ? 1 : 0;
      this.state = 'ended';
      this.onEvent({type: 'end', winner: this.winner});
    }
  }
  input(index, key) {
    if (this.state !== 'running' || this.seats[index].controller !== 'human') return false;
    const result = this.seats[index].game.press(key);
    if (result && typeof result === 'object') this.finish(index, result);
    else this.checkEnd();
    return result;
  }
  update(ms) {
    if (!Number.isFinite(ms) || ms < 0 || ms > 250) throw new Error('Match update must be between 0 and 250ms');
    if (this.state === 'countdown') {
      this.countdown -= ms;
      if (this.countdown <= 0) {
        this.state = 'running';
        for (const seat of this.seats) seat.nextBot = 1000 / this.pps;
      }
      return;
    }
    if (this.state !== 'running') return;
    this.time += ms; this.roundTime += ms;
    for (let index = 0; index < 2 && this.state === 'running'; index++) {
      const seat = this.seats[index];
      if (seat.controller === 'human') {
        const result = seat.game.tick(ms, this.gravity);
        if (result) this.finish(index, result);
      } else if (this.time >= seat.nextBot) {
        const candidates = seat.game.candidates();
        if (!candidates.length) { seat.game.done = true; this.checkEnd(); break; }
        const chosen = this.pickMove(candidates, seat.controller);
        if (!Number.isInteger(chosen) || !candidates[chosen]) throw new Error('Invalid selected bot move');
        for (const key of candidates[chosen].keys) {
          const result = seat.game.press(key);
          if (result && typeof result === 'object') this.finish(index, result);
        }
        seat.nextBot = this.time + 1000 / this.pps;
      }
    }
  }
}
