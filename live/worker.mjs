import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

export const projectRoot = fileURLToPath(new URL('../', import.meta.url));

/** Persistent JSON-lines subprocess; requests are matched by id and timed out. */
export class ScoringWorker {
  constructor({ checkpoint, python = process.env.PYTHON || 'python', timeoutMs = 15000 } = {}) {
    this.pending = new Map();
    this.sequence = 0;
    this.timeoutMs = timeoutMs;
    this.closed = false;
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    const args = ['-u', '-m', 'tetrio_rl', 'bridge'];
    if (checkpoint) args.push('--checkpoint', checkpoint);
    this.child = spawn(python, args, {
      cwd: projectRoot, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      // Credentials belong to the connector, not the model worker.
      env: Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('TETRIO_'))),
    });
    this.readyTimer = setTimeout(() => this.fail(new Error('Python worker did not become ready')), timeoutMs);
    this.child.stderr.on('data', chunk => process.stderr.write(chunk));
    this.child.on('error', error => this.fail(error));
    this.child.on('exit', code => this.fail(new Error(`Python worker exited (${code})`)));
    this.child.stdin.on('error', error => this.fail(error));
    createInterface({ input: this.child.stdout }).on('line', line => {
      let message;
      try { message = JSON.parse(line); }
      catch { return this.fail(new Error('Invalid JSON from Python worker')); }
      if (message.type === 'ready') {
        clearTimeout(this.readyTimer);
        this.resolveReady(message);
        return;
      }
      const pending = this.pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error));
      else pending.resolve(message);
    });
  }

  async score(candidates) {
    await this.ready;
    if (this.closed) throw new Error('Python worker is closed');
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('Model inference timed out'));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ id, type: 'score', candidates }) + '\n');
    });
  }

  fail(error) {
    clearTimeout(this.readyTimer);
    this.closed = true;
    this.rejectReady(error);
    for (const { reject, timer } of this.pending.values()) {
      clearTimeout(timer);
      reject(error);
    }
    this.pending.clear();
    if (!this.child.killed) this.child.kill();
  }

  close() { this.fail(new Error('Worker stopped')); }
}
