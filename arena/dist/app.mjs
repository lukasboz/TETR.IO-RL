import {PlacementModel, encodeCandidate, heuristicScore, PIECES} from './model.mjs';
import {Game, Match, SHAPES, WIDTH, HEIGHT, VISIBLE} from './engine.mjs';

const $ = id => document.getElementById(id);
const COLORS = ['', '#53d9e6', '#7094f7', '#f1ad65', '#edce69', '#73d49c', '#bd86e9', '#ec7d8b', '#71869e'];
const KEYS = {ArrowLeft:'moveLeft',ArrowRight:'moveRight',ArrowDown:'softDrop',ArrowUp:'rotateCW',
  KeyX:'rotateCW',KeyZ:'rotateCCW',KeyA:'rotate180',KeyC:'hold',ShiftLeft:'hold',ShiftRight:'hold',Space:'hardDrop'};
const REPEATING = new Set(['moveLeft','moveRight','softDrop']);
const heldKeys = new Map();
let policy = null, match = null, lastFrame = null, round = 1, score = [0,0], lastMode = 'play';
let flashUntil = [0,0], errorState = false;

function pickMove(candidates, controller) {
  if (controller === 'dqn') return policy.rankCandidates(candidates).index;
  let best = 0, value = -Infinity;
  candidates.forEach((candidate, i) => {
    const current = heuristicScore(encodeCandidate(candidate));
    if (current > value) { value = current; best = i; }
  });
  return best;
}

function newMatch(start = false) {
  heldKeys.clear();
  const mode = $('mode').value;
  if (lastMode !== mode) { score = [0,0]; round = 1; lastMode = mode; }
  const seed = (Date.now() ^ Math.floor(Math.random() * 0x7fffffff)) >>> 0;
  flashUntil = [0,0]; errorState = false;
  match = new Match({seed, mode, pps: Number($('speed').value), gravity: Number($('gravity').value),
    pickMove, onEvent: handleEvent});
  $('seed').textContent = `SEED ${seed}`;
  $('round-label').textContent = `ROUND ${String(round).padStart(2,'0')}`;
  const labels = mode === 'play' ? ['YOU','DQN'] : mode === 'watch' ? ['DQN','HEURISTIC'] : ['DQN A','DQN B'];
  labels.forEach((label, i) => { $('name-'+i).textContent = label;
    $('type-'+i).textContent = match.seats[i].controller === 'human' ? 'KEYBOARD'
      : match.seats[i].controller === 'dqn' ? 'LEARNED POLICY' : 'FIXED BASELINE'; });
  $('touch-controls').hidden = mode !== 'play';
  updateScore();
  $('status').textContent = mode === 'play' ? 'Choose your pace, then start a match.' : 'Watch two policies exchange garbage in a live match.';
  if (start) { match.start(); $('status').textContent = 'Get ready. Both sides receive the same piece sequence.'; }
  updateButtons(); draw();
}

function updateScore() {
  $('scoreboard').textContent = `${$('name-0').textContent}  ${score[0]} — ${score[1]}  ${$('name-1').textContent}`;
}
function updateButtons() {
  $('start').disabled = !policy;
  $('start').innerHTML = `${match?.state === 'idle' ? 'Start match' : 'New round'} <span>↗</span>`;
  $('pause').disabled = !match || ['idle','ended'].includes(match.state);
  $('pause').textContent = match?.state === 'paused' ? 'Resume' : 'Pause';
}
function handleEvent(event) {
  if (event.type === 'end') {
    heldKeys.clear();
    if (event.winner >= 0) score[event.winner]++;
    updateScore(); updateButtons();
    $('status').textContent = event.winner < 0 ? 'Draw. Start a new round to go again.'
      : `${$('name-'+event.winner).textContent} wins. Press R or choose New round for a rematch.`;
  } else if (event.lines) {
    const label = event.perfectClear ? 'PERFECT CLEAR' : event.spin ? `T-SPIN ${['','SINGLE','DOUBLE','TRIPLE'][event.lines]}`
      : ['','SINGLE','DOUBLE','TRIPLE','QUAD'][event.lines];
    $('flash-'+event.seat).textContent = label;
    flashUntil[event.seat] = performance.now() + 1000;
    $('status').textContent = `${$('name-'+event.seat).textContent}: ${label.toLowerCase()}${event.sent ? ` · ${event.sent} garbage sent` : ''}${event.combo > 0 ? ` · combo ${event.combo}` : ''}`;
  }
}

function square(ctx, x, y, size, value, ghost = false) {
  ctx.fillStyle = COLORS[value];
  if (ghost) { ctx.strokeStyle = COLORS[value]; ctx.lineWidth = 2; ctx.strokeRect(x+4,y+4,size-8,size-8); return; }
  ctx.fillRect(x+1,y+1,size-2,size-2);
  ctx.fillStyle = '#ffffff28'; ctx.fillRect(x+3,y+3,size-6,3);
  ctx.fillStyle = '#00000020'; ctx.fillRect(x+3,y+size-5,size-6,2);
}
function mini(ctx, piece, offset = 0) {
  if (!piece) return;
  const blocks = SHAPES[piece][0], minX = Math.min(...blocks.map(c=>c[0])), maxX = Math.max(...blocks.map(c=>c[0]));
  const minY = Math.min(...blocks.map(c=>c[1])), width = (maxX-minX+1)*28, left = (144-width)/2;
  for (const [x,y] of blocks) square(ctx,left+(x-minX)*28,offset+(y-minY)*28+10,28,PIECES.indexOf(piece)+1);
}
function drawBoard(index) {
  const game = match.seats[index].game, canvas = $('board-'+index), ctx = canvas.getContext('2d');
  const size = canvas.width / WIDTH;
  ctx.fillStyle = '#070c13'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#152130'; ctx.lineWidth = 1;
  for (let y=0;y<VISIBLE;y++) for(let x=0;x<WIDTH;x++) {
    ctx.strokeRect(x*size,y*size,size,size);
    const value = game.board[y+HEIGHT-VISIBLE][x];
    if(value) square(ctx,x*size,y*size,size,value);
  }
  if(game.falling && !game.done) {
    ctx.globalAlpha=.4;
    for(const [x,y] of game.cells(game.ghost())) if(y>=HEIGHT-VISIBLE) square(ctx,x*size,(y-HEIGHT+VISIBLE)*size,size,PIECES.indexOf(game.current)+1,true);
    ctx.globalAlpha=1;
    for(const [x,y] of game.cells()) if(y>=HEIGHT-VISIBLE) square(ctx,x*size,(y-HEIGHT+VISIBLE)*size,size,PIECES.indexOf(game.current)+1);
  }
  const hold = $('hold-'+index).getContext('2d'), next = $('next-'+index).getContext('2d');
  hold.clearRect(0,0,144,100); next.clearRect(0,0,144,560);
  hold.globalAlpha=game.holdLocked ? 0.45 : 1; mini(hold,game.held); hold.globalAlpha=1;
  game.queue.slice(0,5).forEach((p,i)=>mini(next,p,i*106));
  $('lines-'+index).textContent=game.lines; $('pieces-'+index).textContent=game.pieces; $('sent-'+index).textContent=game.sent;
  $('pps-'+index).textContent=(game.pieces/Math.max(1,match.roundTime/1000)).toFixed(2);
  const incoming=match.incoming(index), element=$('incoming-'+index);
  element.textContent=`${incoming} incoming`; element.classList.toggle('active',incoming>0);
  const meter=$('meter-'+index), count=Math.min(20,incoming);
  if(meter.childElementCount!==count) meter.innerHTML='<i></i>'.repeat(count);
  $('flash-'+index).classList.toggle('visible',performance.now()<flashUntil[index]);
  const overlay=$('overlay-'+index);
  overlay.hidden=match.state==='running';
  overlay.className='board-overlay';
  if(!overlay.hidden) {
    let kicker='', title='', detail='';
    if(match.state==='idle') { kicker=index===0?'READY WHEN YOU ARE':'LEARNED POLICY';title=index===0?'Make your move.':'Your rival is ready.';detail='Start a match to begin'; }
    if(match.state==='countdown') { kicker='GET READY';title=String(Math.max(1,Math.ceil(match.countdown/1000)));detail=index===0&&match.mode==='play'?'Arrows to move · Space to drop':'Same queue. Different decisions.';overlay.classList.add('countdown'); }
    if(match.state==='paused') { kicker=errorState?'MATCH STOPPED':'TAKE A BREATHER';title=errorState?'Please restart.':'Paused.';detail=errorState?'Choose New round to try again':'Press P or Resume to continue'; }
    if(match.state==='ended') { const won=match.winner===index;kicker='ROUND COMPLETE';title=match.winner<0?'A draw.':won?'Victory.':'Topped out.';detail='Press R for a rematch';overlay.classList.add(won?'winner':'loser'); }
    overlay.innerHTML=`<span class="overlay-kicker">${kicker}</span><strong>${title}</strong><span>${detail}</span>`;
  }
}
function draw() {
  if(!match)return;
  drawBoard(0);drawBoard(1);
  const seconds=Math.floor(match.roundTime/1000);
  $('clock').textContent=`${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
}

function input(action) {
  if(match?.state==='running'&&match.mode==='play') match.input(0,action);
}
function togglePause() { if(!match||errorState)return; match.pause();heldKeys.clear();updateButtons();draw(); }
function startRound() {
  if(!policy)return;
  if(match?.state==='idle') { match.start();$('status').textContent='Get ready. Both sides receive the same piece sequence.';updateButtons(); }
  else { round++;newMatch(true); }
}
$('start').addEventListener('click',startRound);
$('pause').addEventListener('click',togglePause);
$('mode').addEventListener('change',()=>newMatch());
$('gravity').addEventListener('change',()=>{if(match)match.gravity=Number($('gravity').value)});
$('speed').addEventListener('input',()=>{
  const pps=Number($('speed').value);$('speed-value').textContent=pps.toFixed(2).replace(/0$/,'')+' pps';
  if(match) { match.pps=pps; for(const seat of match.seats)seat.nextBot=Math.min(seat.nextBot,match.time+1000/pps); }
});
document.addEventListener('keydown',event=>{
  if(event.ctrlKey||event.metaKey||event.altKey)return;
  if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
  if(['KeyP','Escape'].includes(event.code)) {event.preventDefault();if(!event.repeat)togglePause();return;}
  if(event.code==='KeyR'||event.code==='Enter') {event.preventDefault();if(!event.repeat&&(event.code==='KeyR'||['idle','ended'].includes(match?.state)))startRound();return;}
  const action=KEYS[event.code];
  if(!action)return;event.preventDefault();
  if(heldKeys.has(event.code)||event.repeat)return;
  heldKeys.set(event.code,{action,next:performance.now()+(action==='softDrop'?40:130)});
  input(action);
});
document.addEventListener('keyup',event=>{if(KEYS[event.code]){if(!event.ctrlKey&&!event.metaKey&&!event.altKey)event.preventDefault();heldKeys.delete(event.code)}});
window.addEventListener('blur',()=>{heldKeys.clear();if(match&&['running','countdown'].includes(match.state))togglePause()});
for(const button of document.querySelectorAll('[data-action]')) {
  const action=button.dataset.action,key='touch:'+action;
  button.addEventListener('pointerdown',event=>{event.preventDefault();button.setPointerCapture?.(event.pointerId);input(action);heldKeys.set(key,{action,next:performance.now()+130})});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(name,()=>heldKeys.delete(key));
}
function animate(now) {
  const dt=lastFrame===null?0:Math.min(50,Math.max(0,now-lastFrame));lastFrame=now;
  if(match) {
    try {
      if(match.state==='running') {
        const values=[...heldKeys.values()];
        const horizontal=values.filter(v=>['moveLeft','moveRight'].includes(v.action)).at(-1);
        for(const value of values)if(REPEATING.has(value.action)&&now>=value.next&&(!['moveLeft','moveRight'].includes(value.action)||value===horizontal)) {
          input(value.action);value.next=now+(value.action==='softDrop'?35:38);
        }
      }
      const previous=match.state;match.update(dt);if(previous!==match.state)updateButtons();
      draw();
    } catch(error) {
      heldKeys.clear();errorState=true;match.state='paused';$('pause').disabled=true;
      $('status').textContent='The match stopped unexpectedly. Choose New round to restart.';
      console.error(error);draw();
    }
  }
  requestAnimationFrame(animate);
}

async function initialize() {
  try {
    const payload=globalThis.ARENA_MODEL||await fetch('model.json').then(response=>{if(!response.ok)throw new Error('Model not found');return response.json()});
    policy=new PlacementModel(payload);
    $('model-steps').textContent=`${policy.trainingSteps.toLocaleString()} training placements`;
    newMatch();requestAnimationFrame(animate);
  } catch(error) { $('status').textContent='Could not load the policy. Regenerate this arena with the arena command.';$('start').textContent='Model unavailable';console.error(error); }
}
initialize();
