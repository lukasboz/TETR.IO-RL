import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Game, Match, cancelIncoming, fits, emptyBoard} from '../dist/engine.mjs';
import {PlacementModel, encodeCandidate, heuristicScore, PIECES} from '../dist/model.mjs';

const policy = new PlacementModel(JSON.parse(fs.readFileSync(new URL('../dist/model.json', import.meta.url))));
const choose = (candidates, controller) => controller === 'dqn' ? policy.rankCandidates(candidates).index
  : candidates.map(c => heuristicScore(encodeCandidate(c))).reduce((best, value, i, scores) => value > scores[best] ? i : best, 0);

test('two seeded games share deterministic seven-bag sequences', () => {
  const a = new Game(812), b = new Game(812);
  const sequence = [a.current, ...a.queue];
  for(let i=0;i<14;i+=7)assert.deepEqual([...sequence.slice(i,i+7)].sort(),[...PIECES].sort());
  for(let i=0;i<40;i++) {
    assert.equal(a.current,b.current);
    const candidates=a.candidates(), selected=choose(candidates,'heuristic');
    for(const game of [a,b]){for(const key of candidates[selected].keys)game.press(key);game.spawn()}
    assert.deepEqual(a.board,b.board);
  }
});

test('hold is allowed once per placement and swaps queue correctly', () => {
  const game=new Game(4), current=game.current, next=[...game.queue];
  assert.equal(game.hold(),true);
  assert.equal(game.held,current);assert.equal(game.current,next[0]);assert.equal(game.hold(),false);
  game.hardDrop();game.spawn();
  assert.equal(game.current,next[1]);assert.equal(game.holdLocked,false);
  const afterLock=[...game.queue], swapped=game.current;
  game.hold();assert.equal(game.current,current);assert.equal(game.held,swapped);
  assert.deepEqual(game.queue,afterLock);
});

test('I wall kicks and T floor kicks stay inside the board', () => {
  const game=new Game();game.current='I';game.falling={r:1,x:-2,y:5};
  assert.ok(game.rotate(1));assert.deepEqual(game.falling,{r:2,x:0,y:5});
  game.current='T';game.falling={r:0,x:3,y:22};
  assert.ok(game.rotate(1));assert.deepEqual(game.falling,{r:1,x:2,y:21});
  assert.ok(fits(game.board,'T',1,2,21));
});

test('a quad sends attacks, with a perfect-clear bonus', () => {
  const game=new Game();game.current='I';game.held='I';game.board=emptyBoard();
  for(let y=20;y<24;y++){game.board[y].fill(8);game.board[y][4]=0}
  const candidate=game.candidates().find(c=>c.lines===4);
  let result;
  for(const key of candidate.keys){const value=game.press(key);if(typeof value==='object')result=value}
  assert.equal(result.lines,4);assert.equal(result.perfectClear,true);assert.equal(result.attack,14);
  assert.equal(game.lines,4);assert.equal(game.board.flat().some(Boolean),false);
});

test('queued garbage cancels in order, rises after no-clear locks, and can top out', () => {
  const packets=[{holes:[0,0],readyAt:0},{holes:[3,3,3],readyAt:100}];
  assert.equal(cancelIncoming(packets,3),0);assert.deepEqual(packets[0].holes,[3,3]);
  assert.equal(cancelIncoming(packets,5),3);assert.equal(packets.length,0);
  const match=new Match({pickMove:choose});match.state='running';match.time=1000;
  match.seats[0].pending=[{holes:Array(10).fill(2),readyAt:0}];
  match.input(0,'hardDrop');
  assert.equal(match.incoming(0),2);assert.equal(match.seats[0].game.board.slice(-8).flat().filter(v=>v===8).length,72);
  const game=new Game();game.board[0][0]=1;game.addGarbage([1]);assert.equal(game.done,true);
});

test('line clears cancel incoming attacks and block ready garbage', () => {
  const match=new Match({pickMove:choose});match.state='running';match.time=1000;
  const game=match.seats[0].game;game.falling=null;
  match.seats[0].pending=[{holes:[3,3,3],readyAt:0}];
  match.finish(0,{lines:2,attack:1,combo:0,spin:false,perfectClear:false});
  assert.equal(match.incoming(0),2);assert.equal(game.board.flat().some(Boolean),false);
  assert.equal(match.incoming(1),0);
});

test('gravity, lock delay, countdown, pause and end states work', () => {
  const game=new Game();game.falling=game.ghost();
  assert.equal(game.tick(250),null);assert.ok(game.tick(250));
  const match=new Match({pickMove:choose});match.start();
  for(let i=0;i<12;i++)match.update(250);
  assert.equal(match.state,'running');match.pause();match.update(100);assert.equal(match.time,0);
  match.pause();match.update(100);assert.equal(match.time,100);
  match.seats[0].game.done=true;match.checkEnd();assert.equal(match.state,'ended');assert.equal(match.winner,1);
  assert.equal(match.input(0,'hardDrop'),false);
});

test('trained DQN and heuristic complete a deterministic versus simulation with real attacks', () => {
  let locks=0,attacks=0;
  const match=new Match({seed:107,mode:'watch',pps:5,pickMove:choose,
    onEvent:event=>{if(event.type==='placement'){locks++;attacks+=event.sent}}});
  match.start();
  for(let i=0;i<5000&&match.state!=='ended';i++)match.update(100);
  assert.ok(locks>30);assert.ok(attacks>0);
  assert.ok(match.seats.every(s=>s.game.board.length===24&&s.game.board.every(r=>r.length===10)));
  assert.equal(match.state,'ended');
  console.log(JSON.stringify({winner:match.winner,pieces:match.seats.map(s=>s.game.pieces),lines:match.seats.map(s=>s.game.lines),attacks}));
});
