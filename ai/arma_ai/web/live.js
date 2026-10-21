import {MAPS, MAP_NAMES, ACTION_NAMES, decode, drawMap, drawArena, drawSparkline} from './render.js';

const $ = id => document.getElementById(id);
const node = (tag, className = '', text = '') => {const n=document.createElement(tag);n.className=className;n.textContent=text;return n;};
const fmt = (v, digits=3) => Number.isFinite(v) ? (v !== 0 && Math.abs(v)<.001 ? v.toExponential(2) : v.toLocaleString(undefined,{maximumFractionDigits:digits})) : '—';
const age = seconds => seconds<60 ? `${Math.floor(seconds)} s` : seconds<3600 ? `${Math.floor(seconds/60)} min` : `${Math.floor(seconds/3600)} h`;
const S = {connected:false, source:'training', view:'live', arena:0, player:null, frame:null, status:null, metrics:null, topology:null, layer:null, camera:'follow'};
let queued=false, arenaScene=null, arenaLoading=false, arenaFailed=false, networkScene=null, networkLoading=false, networkFailed=false;
let sceneKey='', sensorKey='', stageKey='', metricsKey='', topologyVersion='', configKey='';
let controlChain=Promise.resolve();
const choices={search:'Search chose this move',policy_sampled:'Sampled during exploration',greedy:'Highest-probability move',hunter:'Space-fighter opponent',waiting:'Waiting for its first decision'};
const live = () => !!(S.connected && S.source==='training' && S.status?.available && S.frame && S.frame.arena===S.arena && Date.now()/1000-S.frame.timestamp<5);
const player = () => S.frame?.players?.find(p=>p.id===S.player) || S.frame?.players?.find(p=>p.id===S.frame.me) || S.frame?.players?.[0];
const config = () => Object.keys(S.status?.config || {}).length ? S.status.config : S.metrics?.configuration?.values || {};
const modeName = mode => ({background:'Background',full:'Full performance'}[mode] || (mode ? mode : 'Custom'));
function schedule(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;render();});}}
function control(body){const payload=JSON.stringify(body);controlChain=controlChain.then(async()=>{try{const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:payload});if(!r.ok)throw Error((await r.json()).error || 'Could not select the arena');$('ui-message').textContent='';}catch(e){$('ui-message').textContent=e.message;}});return controlChain;}
function clearFrame(){S.frame=null;sceneKey=sensorKey=stageKey='';arenaScene?.clear();$('arena-fallback').getContext('2d').clearRect(0,0,$('arena-fallback').width,$('arena-fallback').height);}
function setView(view, route=true){
  S.view=['live','network','training'].includes(view)?view:'live';
  for(const b of document.querySelectorAll('[data-view]')){const active=b.dataset.view===S.view;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;$(`view-${b.dataset.view}`).hidden=!active;}
  networkScene?.setVisible(S.view==='network');
  if(route)history.replaceState(null,'',`#${S.view}`);
  if(S.view==='network')ensureNetwork();
  metricsKey='';schedule();
}
for(const b of document.querySelectorAll('[data-view]')){
  b.addEventListener('click',()=>setView(b.dataset.view));
  b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll('[data-view]')],i=tabs.indexOf(b),j=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;setView(tabs[j].dataset.view);tabs[j].focus();});
}
$('show-training').addEventListener('click',()=>setView('training'));
$('arena-select').addEventListener('input',e=>{S.arena=Number(e.target.value);S.player=null;clearFrame();schedule();control({arena:S.arena,paused:false});});
$('player-select').addEventListener('input',e=>{S.player=Number(e.target.value);sceneKey=sensorKey=stageKey='';schedule();});
$('sensor-select').addEventListener('change',()=>{sensorKey='';schedule();});
for(const mode of ['follow','arena'])$(`camera-${mode}`).addEventListener('click',()=>{S.camera=mode;arenaScene?.setMode(mode);schedule();});
$('network-stage').addEventListener('change',e=>selectStage(e.target.value));
$('network-fit').addEventListener('click',()=>networkScene?.fit());
$('network-focus').addEventListener('click',()=>networkScene?.focus(S.layer));
window.addEventListener('hashchange',()=>setView(location.hash.slice(1),false));
window.addEventListener('resize',()=>{sceneKey=sensorKey=stageKey=metricsKey='';schedule();});

async function ensureArena(){
  if(arenaScene || arenaLoading || arenaFailed)return;
  arenaLoading=true;
  try{const {LiveArena}=await import('./arena3d.js');arenaScene=new LiveArena($('live-scene'));arenaScene.setMode(S.camera);sceneKey='';}
  catch(e){arenaFailed=true;$('live-scene').hidden=true;$('arena-fallback').hidden=false;$('camera-follow').disabled=true;S.camera='arena';$('ui-message').textContent='3D unavailable in this browser; showing the same live arena from above.';}
  finally{arenaLoading=false;schedule();}
}
async function ensureNetwork(){
  if(!S.topology || networkLoading || networkFailed)return;
  try{
    if(!networkScene){networkLoading=true;const {ArchitectureScene}=await import('./architecture3d.js');networkScene=new ArchitectureScene($('network-scene'),selectStage);}
    const key=`${S.topology.version}:${S.topology.observation_version}:${S.topology.session}`;
    if(topologyVersion!==key){networkScene.setTopology(S.topology);topologyVersion=key;}
    networkScene.select(S.layer);networkScene.setVisible(S.view==='network');
  }catch(e){networkFailed=true;$('network-scene').hidden=true;$('network-list').hidden=false;$('network-help').textContent='Select a stage to see its structure and role.';}
  finally{networkLoading=false;}
}
function renderStatus(){
  const p=S.status,m=S.metrics?.last || {},c=config(),active=live(),running=S.connected && p?.running;
  $('connection-state').textContent=active?'Live':!S.connected?'Disconnected':running?'Waiting for live feed':'Training stopped';
  $('connection-state').className=`status-pill ${active?'live':running?'waiting':''}`;
  $('mode-value').textContent=modeName(p?.mode || c.mode);
  $('arenas-value').textContent=fmt(p?.n_arenas || c.games,0);
  $('update-value').textContent=fmt(p?.update ?? m.update,0);
  $('pace-value').textContent=running?fmt(p?.sps,0):'—';
  $('games-value').textContent=fmt(p?.games ?? m.games,0);
  $('phase-value').textContent=running?({warming_up:'Filling replay buffer',learning:'Learning',starting:'Starting'}[p.phase] || p.phase):'Training stopped';
  $('sample-age').textContent=active?`Actor sample ${fmt(Math.max(0,Date.now()/1000-S.frame.timestamp),1)} s ago`:'No live sample displayed';
  $('game-empty').hidden=active;
  $('empty-title').textContent=!S.connected?'Connecting to the viewer':running?'Waiting for the live actor':'Training is stopped';
  $('empty-help').textContent=!S.connected?'Keep the viewer server running to receive training updates.':running?p.message:'Choose a mode in your terminal. Your saved checkpoint will resume.';
  document.querySelector('.mode-commands').hidden=!!running;
  $('footer-state').textContent=active?'Live training · newest actor sample · camera follows actual positions':running?'Waiting for a fresh actor sample':'Training stopped · no game playback';
  const picker=$('arena-select'),n=p?.n_arenas || c.games || 0;
  if(n>0 && picker.options.length!==n){picker.replaceChildren();for(let i=0;i<n;i++)picker.add(new Option(String(i+1),String(i)));picker.value=String(S.arena);}
  picker.disabled=!running || !n;
  $('camera-follow').setAttribute('aria-pressed',String(S.camera==='follow'));
  $('camera-arena').setAttribute('aria-pressed',String(S.camera==='arena'));
  const reportAge=Number.isFinite(m.time)?Math.max(0,Date.now()/1000-m.time):null;
  $('mini-policy').textContent=fmt(m.pg);$('mini-value').textContent=fmt(m.vf);
  $('mini-lr').textContent=fmt(S.status?.hyperparameters?.lr ?? c.lr ?? m.lr,6);
  $('mini-report-age').textContent=reportAge===null?'No learning report yet':`${running?'Latest':'Saved'} report · update ${fmt(m.update,0)} · ${age(reportAge)} ago`;
  if(!active){
    if(S.frame && Date.now()/1000-S.frame.timestamp>=5)clearFrame();
    $('sensor-empty').hidden=false;$('sensor-map').hidden=true;
    $('bot-action').textContent=running?'Waiting for a fresh move':'Training stopped';
    $('bot-alive').textContent='—';$('action-detail').textContent='The view follows the actor when training is running.';
    for(const id of ['bot-speed','bot-brake','bot-kills','bot-update'])$(id).textContent='—';
    $('player-select').disabled=true;
  }
}
function unpackObservation(p){
  const o=p?.observation;if(!o)return null;
  if(!o.maps){const current=decode(o.current);if(current.length!==16384)return null;o.maps=Object.fromEntries(MAPS.map((id,i)=>[id,current.subarray(i*4096,(i+1)*4096)]));}
  return o;
}
function renderGame(){
  if(!live())return;
  ensureArena();const f=S.frame,p=player();if(!p)return;
  const picker=$('player-select'),signature=f.players.map(p=>`${p.id}:${p.name}`).join('|');
  if(picker.dataset.players!==signature){picker.replaceChildren();for(const x of f.players)picker.add(new Option(`${x.id+1} · ${x.name}`,String(x.id)));picker.dataset.players=signature;picker.value=String(p.id);}
  if(picker.dataset.following!==String(p.id)){picker.value=String(p.id);picker.dataset.following=String(p.id);}
  picker.disabled=false;
  $('match-title').textContent=f.matchup || 'Self-play';
  $('round-info').textContent=`Round ${f.round_id} · ${fmt(f.time,2)} s game time`;
  $('bot-name').textContent=`Bot ${p.id+1} · ${p.name}`;
  $('bot-action').textContent=p.action==null?'Waiting for a move':ACTION_NAMES[p.action];
  $('bot-alive').textContent=p.alive?'Alive':'Out';
  $('action-detail').textContent=`${choices[p.selection] || p.selection}${p.decision>=0?` · decision ${p.decision}`:''}${f.searching?' · thinking ahead for the next move':''}`;
  $('bot-speed').textContent=`${fmt(p.speed,1)} m/s`;$('bot-brake').textContent=p.braking?'On':'Off';$('bot-kills').textContent=fmt(p.kills,0);$('bot-update').textContent=p.policy_update==null?'Opponent logic':`u${fmt(p.policy_update,0)}`;
  const key=`${f.actor_pid}:${f.arena}:${f.round_id}:${f.decision}:${p.id}:${S.camera}`;
  if(sceneKey!==key){sceneKey=key;arenaScene?.update(f,p);if(arenaFailed)drawArena($('arena-fallback'),{...f,me:p.id});}
  const legend=$('player-legend');legend.replaceChildren();
  for(const x of f.players){const label=node('span',x.id===p.id?'selected':'',`Bot ${x.id+1}${x.id===p.id?' · following':''}${x.alive?'':' · out'}`);label.style.setProperty('--player-color',f.trails.find(t=>t.id===x.id)?.color || '#8594a2');legend.append(label);}
  $('feed-detail').textContent=`Arena ${f.arena+1} of ${f.n_arenas} · real positions and trails`;
  const o=unpackObservation(p),map=$('sensor-select').value,skey=`${key}:${o?.decision}:${map}`;
  $('sensor-empty').hidden=!!o;$('sensor-map').hidden=!o;
  if(o && sensorKey!==skey){sensorKey=skey;drawMap($('sensor-map'),o.maps[map],map);}
  const meanings={close_in:'1 m per cell. The bot faces up; walls and nearby opponents are shown.',local_in:'2 m per cell. A wider view around the same bot.',global_in:'The whole arena, rotated with the bot’s heading.',territory_in:'A race to each cell: cyan is reached first by this bot, orange by an opponent.'};
  $('sensor-explanation').textContent=o?`${meanings[map]} Actual input for decision ${o.decision}.`:'This opponent uses space-fighter logic rather than a neural observation.';
}
function selectStage(id){S.layer=id;stageKey='';networkScene?.select(id);$('network-stage').value=id;schedule();}
function renderNetwork(){
  if(!S.topology)return;ensureNetwork();
  const t=S.topology,layer=t.layers.find(x=>x.id===S.layer);if(!layer)return;
  $('network-summary').textContent=`V${t.version} · ${fmt(t.architecture.parameters,0)} parameters · ${t.architecture.scalar_count} scalar inputs · ${t.actions.length} actions · checkpoint update ${fmt(t.update,0)}`;
  $('stage-title').textContent=layer.name;$('stage-shape').textContent=layer.shape.join(' × ');
  $('stage-explanation').textContent=layer.explanation || '';$('stage-operation').textContent=layer.operation || '';
  const facts=$('stage-facts');facts.replaceChildren();
  for(const [label,value] of [['Parameters',fmt(layer.parameters || 0,0)],['Stage type',layer.kind],['Input history',`${t.architecture.history_frames} frames`]]){const row=node('div');row.append(node('dt','',label),node('dd','',value));facts.append(row);}
  const p=live()?player():null,o=unpackObservation(p),key=`${layer.id}:${S.frame?.timestamp}:${p?.id}:${!!o}`;
  if(stageKey===key)return;stageKey=key;
  const root=$('stage-live-input');root.replaceChildren();
  if(o && MAPS.includes(layer.id)){const canvas=node('canvas');canvas.setAttribute('aria-label','Live input to this stage');root.append(canvas,node('p','',`Bot ${p.id+1} · actual decision ${o.decision}`));drawMap(canvas,o.maps[layer.id],layer.id);}
  else if(o && (layer.id==='scalars_in' || layer.id==='opponents_in')){
    const list=node('dl','stage-facts');for(let i=0;i<layer.names.length;i++){const index=layer.scalar_indices?.[i] ?? i+(layer.scalar_offset || 0);if(layer.names[i]==='unused')continue;const row=node('div');row.append(node('dt','',layer.names[i]),node('dd','',fmt(o.scalars[index])));list.append(row);}root.append(list);
  }else if(p && layer.id==='policy'){
    const list=node('dl','stage-facts');for(let i=0;i<(p.policy_probs?.length || 0);i++){const row=node('div');row.append(node('dt','',t.actions[i]),node('dd','',p.mask[i]?`${fmt(p.policy_probs[i]*100,1)}%`:'Not legal'));list.append(row);}root.append(list,node('p','',`Actual actor output at decision ${p.decision}, weights u${p.policy_update}. The policy samples its executed move directly.`));
  }else if(p && layer.id==='value'){root.append(node('p','',`Recorded value: ${fmt(p.value)}. This predicts discounted outcome; it is not a win percentage.`));}
}
const LEARNING=[
  ['pg','Policy loss','Clipped actor-critic objective for executed actions and their discounted returns. It does not directly measure wins.'],
  ['vf','Value loss','Squared error in predicting discounted round outcomes. Lower means more accurate value targets on this training batch.'],
  ['lr','Learning rate','The optimizer’s step size when updating weights. This is a hyperparameter, not a measure of skill.'],
  ['ent','Action uncertainty','Entropy of the action distribution, in nats. Higher means probability is spread across more legal moves.'],
  ['clip_fraction','Clipped fraction','Fraction of executed-action probability ratios outside the policy clip interval.'],
  ['policy_kl','Policy change','Estimated divergence from the actor’s recorded sampling policy.'],
  ['grad_norm','Gradient size','Size of the gradients before clipping. Clipping limits unusually large weight updates.'],
  ['value_explained','Value fit','How much of the target variation the value prediction explains. It can be negative; closer to 1 indicates a better fit.'],
  ['aux','Auxiliary loss','Combined territory, danger and successful-attack prediction losses. These train useful shared features.'],
  ['territory','Territory loss','Error in predicting the bot’s territory share two seconds ahead.'],
  ['danger','Danger loss','Error in predicting whether the bot dies within two seconds.'],
  ['safe_kill','Attack loss','Error in predicting a credited enemy kill while surviving the next two seconds.'],
  ['own','Ownership loss','Error in predicting which arena cells belong to the bot, its opponents, or neither.']
];
const PACE=[
  ['sps','Decisions / second','Real simulator decisions generated per wall-clock second across the parallel games.'],
  ['evals_s','Positions / second','Network positions evaluated per second across the actor’s real decisions.'],
  ['replay','Replay ratio','Training samples consumed per generated main-network sample. Reusing data does not create new games.'],
  ['samples','Samples generated','Observations collected for the main network across training sessions.'],
  ['trained','Samples learned from','Total main-network samples used in optimizer steps, including repeated replay samples.'],
  ['ep_len','Round length','Average recorded round length in decisions. Each decision represents 0.05 seconds of game time.']
];
const SETTINGS={
  mode:['Mode','The selected resource preset; explicit flags can override its values.'],games:['Parallel games','Number of simulator arenas generating experience concurrently.'],threads:['Simulator workers','CPU threads used to advance simulator games.'],torch_threads:['Torch CPU threads','CPU threads available to PyTorch operations in each process.'],duty_cycle:['Work fraction','Per actor/learner work/rest target. 1 is uncapped; 0.35 adds rest. This is not a GPU utilization percentage.'],batch:['Batch size','Replay samples used in one optimizer update.'],lr:['Learning rate','Step size used by AdamW to change the weights.'],weight_decay:['Weight decay','AdamW regularization applied to the weights.'],optimizer:['Optimizer','Algorithm that applies the gradient updates.'],optimizer_eps:['Optimizer epsilon','Small stabilizing term used by AdamW.'],optimizer_betas:['Optimizer betas','Decay rates for AdamW’s gradient and squared-gradient averages.'],device:['Device','The compute device used for learning: mps is the Apple GPU.'],bf16:['Bfloat16 learning','Mixed-precision learning on supported GPU devices.'],policy_clip:['Policy clip','Maximum beneficial probability-ratio change in the clipped actor-critic objective.'],entropy_coef:['Entropy weight','Weight of the policy exploration bonus.'],max_policy_lag:['Maximum policy age','Oldest actor weight update eligible for replay, measured in learner updates.'],net:['Network format','V4 is the only supported architecture and checkpoint format.'],sims:['Search simulations','Hypothetical simulations per searched decision.'],macro:['Search step length','Number of decisions per search edge. Each decision is 0.05 seconds of game time.'],search_prob:['Search frequency','Fraction of decisions selected for lookahead search.'],gumbel:['Search exploration','Scale of Gumbel noise used to explore root actions.'],temperature:['Move temperature','Exploration temperature for moves made without search.'],value_steps:['Value target horizon','Decisions before a value estimate is used to bootstrap a training target. Terminal outcomes are used when the round ends sooner.'],value_coef:['Value loss weight','How strongly value error contributes to the total learning loss.'],aux_coef:['Auxiliary loss weight','Weight of territory, danger and successful-attack supervision.'],own_coef:['Ownership loss weight','Weight of the future territory ownership-map loss.'],max_grad_norm:['Gradient clip limit','Maximum gradient norm applied before an optimizer step.'],ring:['Main replay capacity','Number of main-network samples retained in the replay ring.'],exploiter_ring:['Exploiter replay capacity','Replay capacity reserved for the exploiter network.'],replay_ratio:['Target replay ratio','Target learning samples per generated sample.'],warmup:['Replay warmup','Fresh samples needed before learning begins in this session.'],hunter_prob:['Space-fighter matches','Fraction allocated to the simulator’s territory-search opponent.'],hunter_depth:['Space-fighter depth','Search depth of that opponent.'],pool_prob:['Past-opponent matches','Fraction allocated to frozen earlier checkpoints.'],exploiter_prob:['Exploiter matches','Fraction allocated to the network looking for weaknesses in a frozen target.'],exploiter_win:['Promotion score','Exploiter score required for promotion over its evaluation window.'],exploiter_games:['Promotion window','Recent exploiter rounds used for that score.'],exploiter_patience:['Promotion patience','Maximum exploiter updates before promotion by the fallback rule.'],pool_keep:['Opponent pool size','Maximum retained frozen opponents.'],publish_seconds:['Actor weight refresh','Interval for publishing updated learner weights to the actor.'],save_seconds:['Checkpoint interval','Interval between full resumable checkpoints.'],snapshot_seconds:['Opponent snapshot interval','Interval between adding frozen versions to the league.'],log_every:['Statistics interval','Main-network updates between learning reports.'],max_updates:['Update limit','Total update cap; 0 continues until stopped.'],seed:['Random seed','Seed for the training run’s randomized behavior.'],resume:['Resume checkpoint','Whether training continues the existing checkpoint.'],run:['Run directory','Directory containing this run’s checkpoints and reports.']
};
function renderMetrics(){
  const m=S.metrics?.last || {},c=config(),h=S.status?.hyperparameters || {},running=S.connected && S.status?.running;
  $('metrics-freshness').textContent=Number.isFinite(m.time)?`${running?'Latest logged':'Saved'} report · update ${fmt(m.update,0)} · ${new Date(m.time*1000).toLocaleString()}${S.status?.phase==='warming_up'?' · waiting for this session’s first learning update':''}`:'No learning report yet. Statistics will appear as the trainer learns.';
  $('config-source').textContent=Object.keys(S.status?.config || {}).length?(running?'Running trainer':'Last trainer configuration'):'Saved checkpoint';
  const key=`${m.time}:${S.view}:${innerWidth}`;
  if(metricsKey!==key){metricsKey=key;
    for(const [id,defs] of [['learning-metrics',LEARNING],['throughput-metrics',PACE]]){const root=$(id);root.replaceChildren();for(const [field,label,description] of defs){const card=node('article','learning-card'),head=node('header'),value=node('span','metric-value',fmt(m[field]));value.title=String(m[field] ?? 'No measurement yet');head.append(node('h3','',label),value);const chart=node('canvas');chart.setAttribute('aria-label',`${label} history`);card.append(head,chart,node('p','',description));root.append(card);drawSparkline(chart,S.metrics?.series?.[field] || []);}}
    const results=$('match-results');results.replaceChildren();const labels={win_vs_hunter:'Bot vs space fighter',win_vs_past:'Bot vs past checkpoints',win_exploiter:'Exploiter vs frozen target'};
    for(const [name,label] of Object.entries(labels)){const r=m.match_results?.[name],card=node('article','match-score');card.append(node('h3','',label),node('strong','',r?`${fmt(r.score*100,1)}%`:'—'),node('p','',r?`${r.wins} wins · ${r.draws} draws · ${r.losses} losses · ${r.games} games`:'No outcome counts in this report.'));results.append(card);}
  }
  const values={...c,...h},cfgKey=JSON.stringify(values);
  if(configKey!==cfgKey){configKey=cfgKey;const tbody=$('hyperparameters');tbody.replaceChildren();
    for(const [field,value] of Object.entries(values)){const info=SETTINGS[field] || [field.replaceAll('_',' '),'Resolved value reported by the trainer.'];const row=node('tr'),name=node('td','',info[0]);name.append(node('code','',field));const text=typeof value==='number'?fmt(value,8):Array.isArray(value)?value.map(v=>fmt(v,5)).join(', '):typeof value==='boolean'?(value?'On':'Off'):value===''?'None':String(value);row.append(name,node('td','',text),node('td','',info[1]));tbody.append(row);}
    $('configuration-help').textContent=Object.keys(values).length?`${modeName(S.status?.mode || c.mode)} · values from ${Object.keys(S.status?.config || {}).length?'the trainer’s resolved configuration':'the saved checkpoint'}. Background pacing leaves idle time; full mode has no artificial duty limit.`:'Configuration will appear when this trainer starts or a saved checkpoint provides it.';
  }
}
function render(){renderStatus();if(S.view==='live')renderGame();else if(S.view==='network')renderNetwork();else renderMetrics();}
const events=new EventSource('/events');
events.onopen=()=>{S.connected=true;schedule();};
events.onerror=()=>{S.connected=false;clearFrame();schedule();};
events.addEventListener('state',e=>{const state=JSON.parse(e.data);if(state.arena!==S.arena || state.source!==S.source){clearFrame();S.player=null;}S.source=state.source;S.arena=state.arena ?? 0;$('arena-select').value=String(S.arena);if(state.source==='training' && state.paused)control({paused:false});schedule();});
events.addEventListener('training_status',e=>{S.status=JSON.parse(e.data);if(!S.status.available)clearFrame();schedule();});
events.addEventListener('training_frame',e=>{const f=JSON.parse(e.data);if(S.source!=='training' || !S.status?.available || f.arena!==S.arena || Date.now()/1000-f.timestamp>=5)return;if(S.frame?.actor_pid===f.actor_pid && f.timestamp<=S.frame.timestamp)return;S.frame=f;schedule();});
events.addEventListener('metrics',e=>{S.metrics=JSON.parse(e.data);metricsKey=configKey='';schedule();});
events.addEventListener('topology',e=>{const t=JSON.parse(e.data);if(!t.architecture)return;const changed=!S.topology || S.topology.version!==t.version || S.topology.observation_version!==t.observation_version || S.topology.session!==t.session;S.topology=t;if(changed){const picker=$('network-stage'),list=$('network-list');picker.replaceChildren();list.replaceChildren();for(const layer of t.layers){picker.add(new Option(`${layer.name} · ${layer.shape.join(' × ')}`,layer.id));const b=node('button','',layer.name);b.addEventListener('click',()=>selectStage(layer.id));list.append(b);}picker.disabled=false;S.layer=t.layers.some(x=>x.id==='close_in')?'close_in':t.layers[0].id;picker.value=S.layer;topologyVersion='';}stageKey='';if(S.view==='network')ensureNetwork();schedule();});
setInterval(schedule,1000);
setView(location.hash.slice(1),false);
control({source:'training',paused:false});
