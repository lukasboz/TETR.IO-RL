import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { ScoringWorker, projectRoot } from './worker.mjs';
import { enumeratePlacements, keyEvents, requireBotAccount } from './engine.mjs';

const { values } = parseArgs({ options: {
  checkpoint: { type: 'string' }, room: { type: 'string' }, pps: { type: 'string', default: '1' },
  heuristic: { type: 'boolean', default: false }, help: { type: 'boolean', default: false },
} });

if (values.help) {
  console.log('node --env-file=.env live/index.mjs --checkpoint runs/initial/latest.pt [--room ROOM] [--pps 1]\n'
    + 'Omit --room to create a private room. --heuristic selects the fixed baseline instead of a model.');
  process.exit(0);
}

let worker, client, stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  worker?.close();
  if (client) await client.destroy();
}

try {
  const pps = Number(values.pps);
  if (!Number.isFinite(pps) || pps <= 0 || pps > 10) throw new Error('--pps must be between 0 and 10');
  if (!!values.checkpoint === values.heuristic) throw new Error('Select --checkpoint PATH or --heuristic');
  const checkpoint = values.checkpoint ? resolve(projectRoot, values.checkpoint) : undefined;
  if (checkpoint && !existsSync(checkpoint)) throw new Error(`Checkpoint does not exist: ${checkpoint}`);
  const credentials = process.env.TETRIO_TOKEN ? { token: process.env.TETRIO_TOKEN }
    : process.env.TETRIO_USERNAME && process.env.TETRIO_PASSWORD
      ? { username: process.env.TETRIO_USERNAME, password: process.env.TETRIO_PASSWORD } : null;
  if (!credentials) throw new Error('Set TETRIO_TOKEN or TETRIO_USERNAME and TETRIO_PASSWORD in .env');
  // Validate the local model before making an account connection.
  worker = new ScoringWorker({ checkpoint });
  const ready = await worker.ready;
  const { Client } = await import('@haelp/teto');
  const { Engine } = await import('@haelp/teto/engine');
  client = await Client.create(credentials);
  requireBotAccount(client.user);
  console.log(`Connected as ${client.user.username}; policy=${ready.policy}, target=${pps} pps`);
  let round = 0;
  client.on('client.game.round.end', () => { round++; });
  client.on('client.game.over', () => { round++; });
  client.on('client.game.round.start', ([tick, engine]) => {
    const currentRound = ++round;
    const clone = new Engine(engine.initializer);
    const interval = Math.ceil(60 / pps);
    let nextFrame = 0;
    let active = true;
    if (engine.board.width !== 10 || engine.board.height !== 20) {
      console.error('Unsupported room board size; expected 10×20.');
      return;
    }
    tick(async ({ frame, engine: currentEngine }) => {
      if (!active || stopping || round !== currentRound || frame < nextFrame) return {};
      try {
        const candidates = enumeratePlacements(clone, currentEngine.snapshot());
        if (!candidates.length) { active = false; return {}; }
        const response = await worker.score(candidates.map(({ keys, ...candidate }) => candidate));
        if (round !== currentRound || stopping) return {};
        if (!Number.isInteger(response.index) || !candidates[response.index]) throw new Error('Invalid model action');
        nextFrame = frame + interval;
        return { keys: keyEvents(candidates[response.index].keys, frame) };
      } catch (error) {
        active = false;
        console.error(`Round stopped: ${error.message}`);
        return {};
      }
    });
  });
  const room = values.room ? await client.rooms.join(values.room) : await client.rooms.create('private');
  await room.switch('player');
  console.log(`Room: ${room.id}. Join it in TETR.IO and start a custom match.`);
  process.on('SIGINT', () => { void stop().then(() => process.exit(0)); });
  process.on('SIGTERM', () => { void stop().then(() => process.exit(0)); });
  client.on('client.error', error => { console.error('TETR.IO client error; stopping.'); void stop(); });
} catch (error) {
  // Do not print third-party request objects/stacks that might contain credentials.
  console.error(`Unable to start: ${error.message || 'connection failed'}`);
  await stop();
  process.exitCode = 1;
}
