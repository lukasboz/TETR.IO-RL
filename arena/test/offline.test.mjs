import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const artifact = new URL('../../runs/arena.html', import.meta.url);

class Element {
  constructor(tagName, id, attributes = '') {
    this.tagName = tagName.toUpperCase(); this.id = id; this.listeners = new Map();
    this.width = Number(attributes.match(/\bwidth="(\d+)"/)?.[1] || 300);
    this.height = Number(attributes.match(/\bheight="(\d+)"/)?.[1] || 150);
    this.value = id === 'mode' ? 'play' : id === 'gravity' || id === 'speed' ? '1' : '';
    this.disabled = false; this.hidden = false; this.childElementCount = 0;
    this.classes = new Set();
    this.classList = {add: value => this.classes.add(value), toggle: (value, yes) => yes ? this.classes.add(value) : this.classes.delete(value)};
    this.drawing = new Proxy({}, {get: () => () => {}, set: () => true});
  }
  set textContent(value) { this.text = String(value); }
  get textContent() { return this.text || ''; }
  set innerHTML(value) { this.html = value; this.childElementCount = (value.match(/<i>/g) || []).length; }
  get innerHTML() { return this.html || ''; }
  getContext() { return this.drawing; }
  addEventListener(type, listener) { this.listeners.set(type, listener); }
  fire(type, event = {}) { this.listeners.get(type)?.(event); }
}

test('single-file arena starts, accepts inputs, pauses, rematches and runs watch mode', async () => {
  const html = fs.readFileSync(artifact, 'utf8');
  assert.ok(!html.includes('<script type="module" src='));
  assert.ok(!html.includes('<link rel="stylesheet"'));
  const script = html.match(/<script type="module">([\s\S]*)<\/script>/)[1];
  assert.ok(!/^import /m.test(script));
  const elements = new Map();
  for (const match of html.matchAll(/<([a-z]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    elements.set(match[3], new Element(match[1], match[3], match[2]));
  }
  const touch = [...html.matchAll(/<button data-action="([^"]+)"/g)].map(match => {
    const element = new Element('button', 'touch-'+match[1]);element.dataset={action:match[1]};return element;
  });
  let now = 0, scheduled = null;
  const listeners = new Map(), errors = [];
  const document = {
    getElementById(id) { assert.ok(elements.has(id), `Missing DOM element: ${id}`); return elements.get(id); },
    querySelectorAll() { return touch; }, addEventListener(type, listener) { listeners.set(type, listener); },
  };
  const context = vm.createContext({document, performance:{now:()=>now},
    window:{addEventListener(type, listener){listeners.set('window:'+type,listener)}},
    console:{error:(...args)=>errors.push(args)},
    requestAnimationFrame:callback=>{scheduled=callback},
    fetch:()=>{throw new Error('An offline artifact must not fetch anything')},
  });
  await vm.runInContext(script, context);
  const inspect = code => vm.runInContext(code, context);
  const advance = (frames = 1) => { for(let i=0;i<frames;i++){now+=50;assert.ok(scheduled);const callback=scheduled;scheduled=null;callback(now)} };
  const key = (code, type = 'keydown') => listeners.get(type)({code,repeat:false,target:{tagName:'BODY'},preventDefault(){}});
  assert.equal(elements.get('start').disabled,false);
  assert.equal(inspect('policy.trainingSteps'),10000);
  elements.get('start').fire('click');advance(65);
  assert.equal(inspect('match.state'),'running');
  listeners.get('keydown')({code:'Space',ctrlKey:true,repeat:false,target:{tagName:'BODY'},preventDefault(){throw new Error('Browser shortcuts must stay available')}});
  assert.equal(inspect('match.seats[0].game.pieces'),0);
  key('Space');key('Space','keyup');
  assert.equal(inspect('match.seats[0].game.pieces'),1);
  const x=inspect('match.seats[0].game.falling.x');key('ArrowLeft');key('ArrowLeft','keyup');
  assert.equal(inspect('match.seats[0].game.falling.x'),x-1);
  key('KeyC');key('KeyC','keyup');assert.notEqual(inspect('match.seats[0].game.held'),null);
  advance(30);assert.ok(inspect('match.seats[1].game.pieces')>0);
  key('KeyP');assert.equal(inspect('match.state'),'paused');
  const time=inspect('match.time');advance(10);assert.equal(inspect('match.time'),time);
  key('KeyP');assert.equal(inspect('match.state'),'running');
  listeners.get('window:blur')();assert.equal(inspect('match.state'),'paused');
  key('KeyR');assert.equal(inspect('match.state'),'countdown');
  elements.get('mode').value='watch';elements.get('mode').fire('change');
  assert.equal(inspect('match.mode'),'watch');assert.equal(elements.get('touch-controls').hidden,true);
  elements.get('speed').value='5';elements.get('speed').fire('input');
  elements.get('start').fire('click');advance(300);
  assert.ok(inspect('match.seats[0].game.pieces')>10);assert.ok(inspect('match.seats[1].game.pieces')>10);
  assert.equal(errors.length,0);
});
