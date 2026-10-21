import {MAPS, MAP_NAMES, ACTIONS, ACTION_NAMES, prepareFrame, prepareAttention, drawMap, drawArena, drawMosaic, drawChannel, drawVector, drawOwnership, drawHeatmap, drawSparkline} from './render.js';

const $ = id => document.getElementById(id);
const el = (tag, className = '', text = '') => {const n = document.createElement(tag); n.className = className; n.textContent = text; return n;};
const fmt = (n, digits = 3) => Number.isFinite(n) ? n.toLocaleString(undefined, {maximumFractionDigits: digits}) : 'Unavailable';
const percent = n => `${Math.round(n * 100)}%`;
const compact = n => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${Math.round(n / 1e3)}K`;
const S = {source: 'training', selection: 'sampled', arena: 0, player: null, trainingStatus: null, benchmark: null, topology: null, frames: [], selected: null, view: 'decision', paused: false, speed: 1, connected: false,
  layer: null, channel: 0, page: 0, plane: -1, inputFrame: 'current', metrics: null, probe: null, probeBusy: null, networkMode: '3d'};
let scheduled = false, stageBuilt = null, lastMetricRender = null;
let scene = null, sceneLoading = false, sceneFailed = false, topologyRevision = 0, sceneRevision = -1;
const frame = () => S.selected || S.frames.at(-1);
const latest = () => S.frames.at(-1);
function schedule() {if (!scheduled) {scheduled = true; requestAnimationFrame(() => {scheduled = false; render();});}}
function message(text) {$('message').textContent = text;}
async function post(route, body) {
  const response = await fetch(route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.error || `Server returned ${response.status}`);
  }
}
async function control(body) {
  try {await post('/control', body); message('');} catch (e) {message(e.message);}
}
function updateRoute() {
  const hash = `#${S.view === 'network' ? `network-${S.networkMode}` : S.view}`;
  if (location.hash !== hash) history.replaceState(null, '', hash);
}
function setView(view, route = true) {
  S.view = view;
  for (const button of document.querySelectorAll('[data-view]')) {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
    $(`${button.dataset.view}-view`).hidden = !active;
  }
  syncSceneDisplay();
  if (route) updateRoute();
  schedule();
}
function syncSceneDisplay() {
  const is3d = S.networkMode === '3d', visible = S.view === 'network' && is3d;
  $('network-view').classList.toggle('scene-mode', is3d);
  for (const id of ['scene-toolbar', 'scene-help', 'architecture-3d', 'scene-caption']) $(id).hidden = !is3d;
  $('signal-path').hidden = is3d; document.querySelector('.signal-help').hidden = is3d;
  $('mode-3d').setAttribute('aria-pressed', String(is3d)); $('mode-2d').setAttribute('aria-pressed', String(!is3d));
  document.querySelector('.keyboard-hints').textContent = visible
    ? 'Drag rotate · Scroll zoom · Arrows pan · Home fit · F focus'
    : S.source === 'training' ? 'Space freeze view · Right arrow next sample' : 'Space pause match · Right arrow step · [ ] match speed';
  scene?.setVisible(visible);
  if (visible) ensureScene();
}
function setNetworkMode(mode) {S.networkMode = mode; syncSceneDisplay(); updateRoute(); schedule();}
function readRoute() {
  const route = location.hash.slice(1);
  if (route === 'network-3d' || route === 'network-2d') {S.networkMode = route.slice(-2); setView('network', false);}
  else setView(route === 'network' || route === 'training' ? route : 'decision', false);
}
async function ensureScene() {
  if (!S.topology || sceneLoading || sceneFailed) return;
  try {
    if (!scene) {
      sceneLoading = true; $('scene-status').textContent = 'Loading 3D architecture…';
      const {ArchitectureScene} = await import('./architecture3d.js');
      scene = new ArchitectureScene($('architecture-3d'), selectLayer);
      sceneLoading = false;
    }
    if (sceneRevision !== topologyRevision) {scene.setTopology(S.topology); sceneRevision = topologyRevision;}
    scene.select(S.layer); scene.setVisible(S.view === 'network' && S.networkMode === '3d');
    $('scene-status').hidden = true; $('scene-fit').disabled = $('scene-focus').disabled = false;
  } catch (error) {
    sceneLoading = false; sceneFailed = true; scene?.dispose(); scene = null;
    $('scene-status').hidden = false;
    $('scene-status').textContent = '3D could not start in this browser. Use the 2D view to inspect the architecture.';
    $('scene-status').title = error.message;
    $('scene-fit').disabled = $('scene-focus').disabled = true;
  }
}
$('mode-3d').addEventListener('click', () => setNetworkMode('3d'));
$('mode-2d').addEventListener('click', () => setNetworkMode('2d'));
$('scene-fit').addEventListener('click', () => scene?.fit());
$('scene-focus').addEventListener('click', () => scene?.focus(S.layer));
$('scene-stage').addEventListener('change', e => selectLayer(e.target.value));
window.addEventListener('hashchange', readRoute);
for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => setView(button.dataset.view));
  button.addEventListener('keydown', e => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault(); e.stopPropagation();
    const tabs = [...document.querySelectorAll('[data-view]')], i = tabs.indexOf(button);
    const j = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    setView(tabs[j].dataset.view); tabs[j].focus();
  });
}
$('pause').addEventListener('click', () => control({paused: !S.paused}));
$('step').addEventListener('click', () => {S.selected = null; control({step: 1});});
$('speed').addEventListener('input', e => {S.speed = Number(e.target.value); control({speed: S.speed});});
$('source').addEventListener('input', e => {const source = e.target.value; if (source === S.source) return; resetHistory(); S.source = source; syncSceneDisplay(); schedule(); control({source});});
$('selection').addEventListener('input', e => {S.selection = e.target.value; control({selection: S.selection}); schedule();});
$('training-arena').addEventListener('input', e => {const arena = Number(e.target.value); if (arena === S.arena) return; resetHistory(); S.arena = arena; schedule(); control({arena});});
$('training-player').addEventListener('input', e => {S.player = Number(e.target.value); schedule();});
$('watch-benchmark').addEventListener('click', () => setView('training'));
$('benchmark-run').addEventListener('click', async () => {
  $('benchmark-run').disabled = true;
  try {await post('/benchmark', {action: 'start', rounds: 20, time_budget_s: 60, size: -3, sample: false}); message('');}
  catch (e) {message(e.message); schedule();}
});
$('benchmark-cancel').addEventListener('click', async () => {
  try {await post('/benchmark', {action: 'cancel'});} catch (e) {message(e.message);}
});
$('live').addEventListener('click', () => {S.selected = null; schedule();});
$('history').addEventListener('input', e => {
  S.selected = S.frames[Number(e.target.value)] || null;
  if (!S.paused) control({paused: true});
  schedule();
});
$('probe').addEventListener('click', async () => {
  const f = frame(); if (!f) return;
  S.probeBusy = f.decision; S.probe = null; schedule();
  try {await post('/probe', {decision: f.decision, update: f.update});} catch (e) {S.probeBusy = null; message(e.message); schedule();}
});
$('follow-head').addEventListener('click', () => control({cell: null}));
document.addEventListener('keydown', e => {
  if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName) || e.target.closest('.signal-path, #architecture-3d') || e.target.isContentEditable || !S.connected) return;
  if (e.code === 'Space') {e.preventDefault(); control({paused: !S.paused});}
  else if (e.key === 'ArrowRight') {e.preventDefault(); S.selected = null; control({step: 1});}
  else if (S.source === 'inspection' && (e.key === '[' || e.key === ']')) {e.preventDefault(); control({speed: S.speed * (e.key === '[' ? .5 : 2)});}
});
window.addEventListener('resize', schedule);

const probabilityRows = ACTIONS.map((name, i) => {
  const row = el('div', 'probability'), label = el('span', '', name), track = el('div', 'probability-track');
  row.hidden = true;
  const fill = el('div', 'probability-fill'), number = el('span', 'probability-number');
  track.append(fill); row.append(label, track, number); $('probabilities').append(row);
  return {row, fill, number};
});
const searchRows = ACTIONS.map((name, i) => {
  const row = el('div', 'probability'), label = el('span', '', name), track = el('div', 'probability-track');
  row.hidden = true;
  const fill = el('div', 'probability-fill'), number = el('span', 'probability-number');
  track.append(fill); row.append(label, track, number); $('search-probabilities').append(row);
  return {row, fill, number};
});
function paintProbabilities(rows, probs, action, mask) {
  rows.forEach(({row, fill, number}, i) => {
    row.hidden = !Array.isArray(probs) || !Number.isFinite(probs[i]);
    if (row.hidden) return;
    const legal = !mask || mask[i];
    row.classList.toggle('chosen', action === i); row.classList.toggle('masked', !legal);
    fill.style.width = `${Math.max(0, Math.min(100, probs[i] * 100))}%`;
    number.textContent = legal ? percent(probs[i]) : 'Masked';
  });
}
const mapCanvases = MAPS.map((id, i) => {
  const button = el('button', 'observation-map'); button.title = `Inspect ${MAP_NAMES[i].toLowerCase()} input`;
  button.append(el('span', '', MAP_NAMES[i])); const canvas = el('canvas'); canvas.width = canvas.height = 128;
  canvas.setAttribute('aria-label', `${MAP_NAMES[i]} observation`); button.append(canvas);
  button.addEventListener('click', () => {if (S.topology?.reads.includes(id)) {selectLayer(id); setView('network');}});
  $('observation-maps').append(button); return {id, button, canvas};
});
function reading(list, name, value) {list.append(el('dt', '', name), el('dd', '', value));}
function resetHistory() {
  S.frames = []; S.selected = null; S.player = null; S.probe = null; S.probeBusy = null; probeRendered = null; stageBuilt = null;
  $('arena-empty').hidden = false; $('action').textContent = 'Waiting for a sample';
  $('action-source').textContent = ''; $('round-time').textContent = 'Waiting';
  $('decision-title').textContent = 'Last executed move'; $('policy-title').textContent = 'Network policy';
  $('probabilities-help').textContent = 'Waiting for a recorded move.'; $('training-player').disabled = true;
  $('players').replaceChildren(); $('arena-legend').replaceChildren();
  $('matchup-label').textContent = 'Waiting for the actor';
  $('source-counts').textContent = ''; $('match-result').textContent = '';
  for (const item of [...probabilityRows, ...searchRows]) item.row.hidden = true;
  $('search-policy').hidden = true;
  const canvas = $('arena'); canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}
function updateControls() {
  const training = S.source === 'training';
  $('source').disabled = $('selection').disabled = !S.connected;
  if (document.activeElement !== $('source') && $('source').value !== S.source) $('source').value = S.source;
  if (document.activeElement !== $('selection') && $('selection').value !== S.selection) $('selection').value = S.selection;
  $('arena-control').hidden = !training; $('selection-control').hidden = training;
  $('trainer-pace').hidden = !training; $('speed-control').hidden = training;
  const available = S.trainingStatus?.available;
  $('pause').disabled = $('step').disabled = !S.connected || (training && !available && !frame());
  $('speed').disabled = !S.connected || training;
  const pause = training ? (S.paused ? 'Resume view' : 'Freeze view') : (S.paused ? 'Resume match' : 'Pause match');
  if ($('pause').querySelector('span').textContent !== pause) $('pause').querySelector('span').textContent = pause; $('pause').setAttribute('aria-label', pause);
  $('pause').setAttribute('aria-pressed', String(S.paused));
  const stepLabel = training ? 'Next sample' : 'Step';
  if ($('step').querySelector('span').textContent !== stepLabel) $('step').querySelector('span').textContent = stepLabel;
  $('step').title = training ? 'Show the next available actor sample; training keeps running' : 'Advance the separate match one displayed decision';
  const icon = $('pause').querySelector('svg');
  if (icon.dataset.paused !== String(S.paused)) {
    icon.innerHTML = S.paused ? '<path d="m4 3 8 5-8 5z"/>' : '<path d="M5 3v10M11 3v10"/>';
    icon.dataset.paused = String(S.paused);
  }
  const speed = String(S.speed);
  if (![...$('speed').options].some(o => o.value === speed)) $('speed').add(new Option(`${fmt(S.speed, 2)}×`, speed));
  if (document.activeElement !== $('speed') && $('speed').value !== speed) $('speed').value = speed;
  const n = S.trainingStatus?.n_arenas || 0, picker = $('training-arena');
  if (n && picker.options.length !== n) {picker.replaceChildren(); for (let i = 0; i < n; i++) picker.add(new Option(String(i + 1), String(i)));}
  picker.disabled = !S.connected || !n; if (document.activeElement !== picker && picker.value !== String(S.arena)) picker.value = String(S.arena);
  $('connection').textContent = S.connected ? `Connected · ${location.host}` : 'Reconnecting';
  $('connection').classList.toggle('connected', S.connected);
  $('source-help').textContent = training
    ? 'Actual training arena. Freezing, stepping and browsing samples affect this display only.'
    : 'Separate real-engine match using the policy directly. No training search; these moves do not train the model.';
  $('network-source').textContent = training
    ? 'Checkpoint architecture. Live actor activations are not recorded; use Policy inspection for a separate match’s activations.'
    : 'Architecture and activations from the separate policy inspection match.';
  $('model-context').textContent = training ? 'Blacklight checkpoint' : 'Blacklight';
  $('history-title').textContent = training ? 'Sample history' : 'Decision history';
  $('history-help').textContent = training ? 'Periodic actor snapshots; intermediate decisions are not recorded in this view.' : 'Recent displayed decisions from the separate match.';
  $('decision-rate').textContent = training ? 'Viewing does not change training' : '20 decisions per second of game time';
  if (S.view !== 'network') document.querySelector('.keyboard-hints').textContent = training
    ? 'Space freeze view · Right arrow next sample' : 'Space pause match · Right arrow step · [ ] match speed';
  const f = frame(), fresh = latest();
  $('probe').disabled = training || !S.connected || !S.paused || !f || f.decision !== fresh?.decision || f.update !== S.topology?.update || S.probeBusy !== null;
  $('probe').textContent = S.probeBusy !== null ? 'Testing…' : 'Test inputs';
  if (training) {$('probe-help').textContent = 'Input tests are available in the separate Policy inspection match.'; $('probe-result').replaceChildren();}
  $('history').disabled = S.frames.length < 2; $('history').max = String(Math.max(0, S.frames.length - 1));
  $('history').value = String(Math.max(0, S.selected ? S.frames.indexOf(S.selected) : S.frames.length - 1));
  $('live').disabled = !S.selected;
  const state = S.selected ? 'History' : S.paused ? 'Frozen view' : training && !available ? 'Last sample' : 'Live';
  $('history-label').textContent = f ? `${state} · ${training ? `sample ${f.view_sample} · step` : 'decision'} ${f.decision}${training ? '' : ` · update ${f.update}`}` : 'No samples yet';
  $('training-match').hidden = !training; $('training-player').hidden = !training;
  document.querySelector('.input-test').hidden = training;
  $('prediction-section').hidden = $('observation-section').hidden = training;
  $('arena-empty').textContent = training ? S.trainingStatus?.message || 'Waiting for the training actor’s first sample.' : 'Waiting for the separate match’s first decision.';
  $('feed-status').textContent = training
    ? `${S.trainingStatus?.message || 'Waiting for the training actor.'}${f?.timestamp ? ` Sample age: ${fmt(Math.max(0, Date.now() / 1000 - f.timestamp), 1)} s.` : ''}${S.paused ? ' View frozen; trainer continues.' : ''}`
    : `Policy inspection · ${S.selection === 'greedy' ? 'highest-probability moves' : 'sampled moves'} · no search`;
  $('feed-status').classList.toggle('stale', training && !available);
  $('benchmark-run').disabled = !S.connected || !!S.benchmark?.running;
  $('benchmark-cancel').hidden = !S.benchmark?.running; $('benchmark-cancel').disabled = !S.connected || !S.benchmark?.running;
}
function renderDecision(f) {
  if (S.source === 'training') {renderTrainingArena(f); return;}
  if (!f) return;
  $('search-policy').hidden = true;
  $('decision-title').textContent = 'Inspection decision';
  $('action-source').textContent = `${f.selection === 'greedy' ? 'Highest-probability policy move' : 'Sampled from the policy'} · no search`;
  $('policy-title').textContent = 'Network policy';
  $('probabilities-help').textContent = 'Probabilities from this decision’s model inputs.';
  $('arena-empty').hidden = true; drawArena($('arena'), f);
  const legend = $('arena-legend'); legend.replaceChildren();
  for (const trail of f.trails || []) {
    const label = el('span', 'legend-player', trail.name);
    label.style.setProperty('--player-color', trail.color); legend.append(label);
  }
  $('round-time').textContent = `Round ${f.round} · ${f.time.toFixed(2)} s`;
  $('arena-title').textContent = S.selected ? 'Recorded inspection decision' : 'Policy inspection match';
  $('action').textContent = ACTION_NAMES[f.action];
  probabilityRows.forEach(({row, fill, number}, i) => {
    row.hidden = i >= f.probs.length;
    if (row.hidden) return;
    row.classList.toggle('chosen', f.action === i); row.classList.toggle('masked', !f.mask[i]);
    fill.style.width = `${Math.max(0, Math.min(100, f.probs[i] * 100))}%`;
    number.textContent = f.mask[i] ? percent(f.probs[i]) : 'Masked';
    row.title = `${ACTIONS[i]}: ${fmt(f.probs[i] * 100, 2)}%${!f.mask[i] ? ' (forbidden by turn timing)' : ''}`;
  });
  const list = $('predictions'); list.replaceChildren();
  reading(list, 'Value', fmt(f.value));
  if (f.aux) {reading(list, 'Territory in 2 s', percent(f.aux[0])); reading(list, 'Death within 2 s', percent(f.aux[1])); if (f.aux.length > 2) reading(list, 'Kill + survive in 2 s', percent(f.aux[2]));}
  else reading(list, 'Auxiliary heads', `Absent in v${S.topology.version}`);
  for (const map of mapCanvases) drawMap(map.canvas, f.maps[map.id], map.id);
  $('match-result').textContent = f.result ? `Previous round: ${f.result}` : '';
  renderProbe(f);
}
const CHOICE = {search: 'Search + exploration', policy_sampled: 'Sampled policy', greedy: 'Highest-probability policy', hunter: 'Space-fighter logic', waiting: 'Waiting for the first move'};
function renderTrainingArena(f) {
  $('arena-title').textContent = 'Training arena';
  if (!f) return;
  $('arena-empty').hidden = true;
  const players = f.players || [];
  const p = players.find(p => p.id === S.player) || players.find(p => p.id === f.me) || players[0];
  const picker = $('training-player'), signature = players.map(p => `${p.id}:${p.name}`).join('|');
  if (picker.dataset.players !== signature) {
    picker.replaceChildren(); for (const player of players) picker.add(new Option(player.name, String(player.id)));
    picker.dataset.players = signature;
  }
  picker.disabled = !players.length;
  if (p && picker.value !== String(p.id)) picker.value = String(p.id);
  drawArena($('arena'), {...f, me: p?.id ?? f.me});
  $('round-time').textContent = `Arena ${f.arena + 1} · round ${f.round_id} · ${fmt(f.time, 2)} s`;
  $('matchup-label').textContent = f.matchup || 'Training match';
  const roster = $('players'), legend = $('arena-legend'); roster.replaceChildren(); legend.replaceChildren();
  for (const player of players) {
    const trail = f.trails.find(t => t.id === player.id);
    const label = el('span', 'legend-player', player.name); label.style.setProperty('--player-color', trail?.color || '#94a8ba'); legend.append(label);
    const row = el('div', 'player-row');
    row.append(el('strong', '', player.name), el('span', '', `${player.alive ? 'Alive' : 'Out'} · ${player.learning ? 'learning' : 'opponent'}`));
    row.append(el('small', '', `${CHOICE[player.selection] || player.selection}${Number.isFinite(player.policy_update) ? ` · policy u${fmt(player.policy_update, 0)}` : ''}`));
    roster.append(row);
  }
  const counts = f.source_counts || {};
  $('source-counts').textContent = `Executed moves this round: ${counts.search || 0} searched · ${counts.policy_sampled || 0} sampled · ${counts.greedy || 0} greedy · ${counts.hunter || 0} opponent logic.`;
  if (!p) return;
  $('decision-title').textContent = 'Last executed move';
  $('action').textContent = p.action == null ? 'Waiting for first move' : ACTION_NAMES[p.action];
  const choice = p.selection === 'search' && f.gumbel === 0 ? 'Search' : CHOICE[p.selection] || p.selection;
  $('action-source').textContent = `${choice}${p.decision >= 0 ? ` · step ${p.decision}` : ''}${Number.isFinite(p.policy_update) ? ` · policy u${fmt(p.policy_update, 0)}` : ''}${f.searching ? ' · next search in progress' : ''}`;
  $('policy-title').textContent = p.selection === 'policy_sampled' ? 'Policy sampling probabilities' : p.selection === 'search' ? 'Network policy before search' : 'Network policy';
  $('probabilities-help').textContent = p.selection === 'hunter' ? 'This opponent uses its own search logic; it has no neural policy probabilities.'
    : p.selection === 'search' ? 'The network prior before lookahead. The executed move was chosen by search with its exploration settings.'
    : p.selection === 'greedy' ? 'The highest-probability legal move was chosen.' : p.selection === 'waiting' ? 'No move has executed yet.'
    : `A legal move was sampled from this distribution (temperature ${fmt(f.temperature, 2)}).`;
  paintProbabilities(probabilityRows, p.policy_probs, p.action, p.mask);
  $('search-policy').hidden = !Array.isArray(p.search_probs);
  paintProbabilities(searchRows, p.search_probs, p.action, p.mask);
  $('match-result').textContent = f.finished ? 'This training round has ended.' : '';
}
let probeRendered = null;
function renderProbe(f) {
  const key = S.probe?.decision === f.decision ? S.probe : null;
  $('probe-help').textContent = S.probeBusy === f.decision ? 'Testing substitutions on this decision and these exact weights…'
    : !S.paused ? 'Pause a decision to test what changes the policy.'
    : S.selected && f.decision !== latest()?.decision ? 'Return to Live to test the latest paused decision.'
    : f.update !== S.topology.update ? 'Weights changed. Step to a decision from the new checkpoint.'
    : 'Each test replaces one input while keeping the legal-action mask fixed.';
  if (probeRendered === key) return;
  probeRendered = key; const root = $('probe-result'); root.replaceChildren();
  if (!key) return;
  if (key.error) {root.append(el('p', 'note', key.error)); return;}
  const table = el('table', 'probe-table'), head = el('thead'), hr = el('tr');
  ['Input replaced', 'Policy shift', 'Now prefers', 'Δ value'].forEach(x => hr.append(el('th', '', x))); head.append(hr); table.append(head);
  const body = el('tbody');
  for (const result of [...key.rows].sort((a, b) => b.policy_shift - a.policy_shift)) {
    const row = el('tr'); row.append(el('td', '', result.label), el('td', 'probe-shift', `${fmt(result.policy_shift * 100, 1)}%`), el('td', '', ACTIONS[result.preferred]), el('td', '', `${result.value_delta >= 0 ? '+' : ''}${fmt(result.value_delta)}`));
    row.title = `Substituted policy: ${ACTIONS.map((a, i) => `${a} ${percent(result.probs[i])}`).join(', ')}`;
    body.append(row);
  }
  table.append(body); root.append(table, el('p', 'note', 'Policy shift is total variation: how much probability mass moved. Each map or sensor test keeps the same predecision memory and controls. Memory and control tests clear only their named input. History repeats the current frame. The live memory is never advanced by a test; unfamiliar substitutions measure sensitivity, not causation.'));
}
function nodeFor(id, label) {
  const layer = S.topology.layers.find(l => l.id === id);
  const button = el('button', `node${layer?.kind === 'auxiliary' ? ' auxiliary' : ''}`, label || layer?.short || id);
  button.dataset.stage = id; button.title = `${layer?.name || id} · ${fmt(layer?.parameters || 0, 0)} parameters`;
  button.addEventListener('click', () => selectLayer(id)); return button;
}
function buildGraph() {
  const root = $('signal-path'); root.replaceChildren();
  const arch = S.topology.architecture, columns = el('div', 'signal-branches');
  root.tabIndex = 0; root.setAttribute('role', 'region'); root.setAttribute('aria-label', 'Scrollable network graph');
  document.querySelector('.signal-help').textContent = arch.branches.length > 3
    ? 'Select a stage to inspect it. Scroll sideways for all five branches.'
    : 'Select a stage to inspect its output.';
  const graph = el('div', 'graph-body');
  graph.style.setProperty('--branch-count', String(arch.branches.length));
  graph.style.setProperty('--graph-min', arch.branches.length > 3 ? '1060px' : '0px');
  root.append(graph);
  for (const branch of arch.branches) {
    const lane = el('div', `signal-branch ${branch.id}`); lane.append(el('h3', 'branch-name', branch.name));
    const body = el('div', 'branch-body');
    if (branch.inputs.length > 1) {
      const pair = el('div', 'input-pair'); branch.inputs.forEach(id => pair.append(nodeFor(id, MAP_NAMES[MAPS.indexOf(id)]))); body.append(pair);
    } else {
      const id = branch.inputs[0], input = S.topology.layers.find(l => l.id === id);
      const label = id === 'scalars_in' ? `${input.shape[0]} exact numbers` : id === 'opponents_in' ? '3 enemy vectors' : MAP_NAMES[MAPS.indexOf(id)];
      const node = nodeFor(id, label); node.classList.add('input-node'); body.append(node);
    }
    if (!['scalars', 'opponents'].includes(branch.id)) body.append(el('p', 'branch-context', arch.history_frames === 2 ? 'current + previous' : 'current frame'));
    let residual = null;
    const ownershipSource = arch.edges.find(e => e[1] === 'own')?.[0];
    for (let stageIndex = 0; stageIndex < branch.stages.length; stageIndex++) {
      const id = branch.stages[stageIndex];
      const layer = S.topology.layers.find(l => l.id === id);
      if (layer.operation?.startsWith('Residual')) {
        if (!residual) {
          let count = 0;
          for (const next of branch.stages.slice(stageIndex)) {
            if (!S.topology.layers.find(l => l.id === next).operation?.startsWith('Residual')) break;
            count++;
          }
          if (count > 1) {residual = el('div', 'residual-stack'); residual.append(el('span', '', `${count} residual blocks`)); body.append(residual);}
        }
        (residual || body).append(nodeFor(id));
      } else {residual = null; body.append(nodeFor(id));}
      if (id === ownershipSource) {const side = el('div', 'ownership-note'); side.append(nodeFor('own')); body.append(side);}
    }
    lane.append(body); columns.append(lane);
  }
  graph.append(columns);
  if (arch.context_stages?.length) {
    const links = el('section', 'context-stages');
    links.append(el('h3', '', 'Context connections'));
    const controls = el('div', 'context-controls');
    for (const id of arch.context_stages) controls.append(nodeFor(id,
      id === 'nearby_modulation' ? 'Arena → Nearby conditioning'
        : id === 'close_modulation' ? 'Arena → Close conditioning' : 'Context → opponent query'));
    links.append(controls, el('p', 'note', 'Arena features condition each local view. Spatial features and own sensors form the opponent query.'));
    graph.append(links);
  }
  const fusion = el('div', 'fusion'); fusion.append(nodeFor('trunk1'), nodeFor('trunk2')); graph.append(fusion);
  if (arch.memory_stages?.length) {
    const memory = el('section', 'fusion'); memory.append(el('h3', 'branch-name', `Persistent memory · ${arch.memory.hidden}-unit ${arch.memory.kind}`));
    const inputs = el('div', 'context-controls'); inputs.append(nodeFor('memory_in'), nodeFor('memory_context'));
    memory.append(inputs, nodeFor('memory_state'), nodeFor('memory_residual'), nodeFor('memory_output'),
      el('p', 'note', 'State continues through this player’s life. The trunk keeps a direct path to the adapted features.'));
    graph.append(memory);
  }
  const heads = el('div', 'output-nodes'); heads.append(nodeFor('policy'), nodeFor('value'));
  if (arch.heads.auxiliary) heads.append(nodeFor('aux')); graph.append(heads);
  highlightStage();
}
function highlightStage() {
  for (const node of document.querySelectorAll('[data-stage]')) {
    const selected = node.dataset.stage === S.layer; node.classList.toggle('selected', selected); node.setAttribute('aria-pressed', String(selected));
  }
  if (S.layer) $('scene-stage').value = S.layer;
  scene?.select(S.layer);
}
function selectLayer(id) {S.layer = id; S.channel = 0; S.page = 0; S.plane = -1; S.inputFrame = 'current'; stageBuilt = null; highlightStage(); schedule();}
function statList(layer, stats) {
  const list = $('stage-stats'); if (!list) return;
  list.replaceChildren(); reading(list, 'Parameters', fmt(layer.parameters, 0));
  if (stats) {reading(list, 'Mean', fmt(stats.mean, 4)); reading(list, 'Standard deviation', fmt(stats.std, 4)); reading(list, 'Zero activations', percent(stats.zero)); reading(list, 'Range', `${fmt(stats.min, 2)} to ${fmt(stats.max, 2)}`);}
}
function buildStage(layer) {
  stageBuilt = layer.id;
  $('stage-title').textContent = layer.name.charAt(0).toUpperCase() + layer.name.slice(1);
  $('stage-shape').textContent = `${layer.shape.join(' × ')}${layer.kind === 'input' && layer.shape.length === 2 ? ' · 8 bit planes' : ''} · ${fmt(layer.parameters, 0)} parameters`;
  $('stage-explanation').textContent = layer.explanation || '';
  $('stage-operation').textContent = layer.operation || '';
  $('activation-title').textContent = layer.kind === 'input' ? 'Engine observation' : layer.kind === 'output' || layer.kind === 'auxiliary' ? 'Model output' : 'Output activations';
  const root = $('activation-content'); root.replaceChildren();
  if (layer.id === 'opponents_weights') {
    const outputs = el('div', 'head-values'); outputs.id = 'head-output'; root.append(outputs);
  } else if (layer.kind === 'conv') {
    const mosaic = el('canvas', 'feature-canvas'); mosaic.id = 'feature-mosaic'; mosaic.setAttribute('aria-label', 'Feature channels, 24 per page'); root.append(mosaic);
    mosaic.addEventListener('click', e => {
      const l = mosaic._layout; if (!l) return;
      const rect = mosaic.getBoundingClientRect(), x = e.clientX - rect.left - l.ox, y = e.clientY - rect.top;
      const col = Math.floor(x / (l.tile + l.gap)), row = Math.floor(y / l.pitch), n = l.start + row * l.cols + col;
      if (x >= 0 && col < l.cols && n < l.channels && n < l.start + 24 && x % (l.tile + l.gap) <= l.tile && y % l.pitch <= l.tile) {S.channel = n; schedule();}
    });
    const controls = el('div', 'channel-controls'), prev = el('button', '', 'Previous'), range = el('span'), next = el('button', '', 'Next');
    prev.id = 'channels-prev'; next.id = 'channels-next'; range.id = 'channels-range';
    prev.setAttribute('aria-label', 'Previous feature channels'); next.setAttribute('aria-label', 'Next feature channels');
    prev.addEventListener('click', () => {S.page = Math.max(0, S.page - 1); S.channel = S.page * 24; schedule();});
    next.addEventListener('click', () => {S.page = Math.min(Math.floor((layer.shape[0] - 1) / 24), S.page + 1); S.channel = S.page * 24; schedule();});
    controls.append(prev, range, next); root.append(controls);
    const detail = el('div', 'stage-detail'), large = el('canvas', 'channel-large'); large.id = 'channel-large'; large.setAttribute('aria-label', 'Selected feature channel');
    const info = el('div'), label = el('label', 'channel-controls'), input = el('input'); input.id = 'channel'; input.type = 'number'; input.min = '0'; input.max = String(layer.shape[0] - 1); input.value = String(S.channel);
    label.append(document.createTextNode('Channel'), input); info.append(label, el('h4', '', 'Stage statistics'));
    input.addEventListener('input', () => {S.channel = Math.max(0, Math.min(layer.shape[0] - 1, Math.trunc(Number(input.value)) || 0)); S.page = Math.floor(S.channel / 24); schedule();});
    const stats = el('dl', 'readings stats'); stats.id = 'stage-stats'; info.append(stats); detail.append(large, info); root.append(detail);
    root.append(el('p', 'note', 'Scaled brightness. Raw statistics. Cyan + · lilac −.'));
  } else if (layer.kind === 'dense') {
    const canvas = el('canvas', 'feature-canvas'); canvas.id = 'vector-output'; canvas.setAttribute('aria-label', 'Neuron activations'); root.append(canvas);
    const stats = el('dl', 'readings stats'); stats.id = 'stage-stats'; root.append(stats, el('p', 'note', 'One cell per neuron. Display brightness is scaled; statistics are measured before quantisation.'));
    if (layer.id === 'memory_context') {const context = el('dl', 'readings'); context.id = 'memory-readings'; root.append(context);}
  } else if (layer.id === 'scalars_in' || layer.id === 'opponents_in') {
    root.append(el('p', 'note', 'These are the normalised numbers the model receives. Indices refer to the original engine observation.'));
    const list = el('div', 'scalar-list'); list.id = 'scalar-list';
    layer.names.forEach((name, i) => {const row = el('div'); const index = layer.scalar_indices?.[i] ?? i + (layer.scalar_offset || 0); row.append(el('span', '', String(index)), el('span', '', name), el('span')); list.append(row);}); root.append(list);
  } else if (layer.kind === 'input') {
    const controls = el('div', 'observation-controls');
    const select = el('select', 'input-plane-select'); select.id = 'input-plane'; select.setAttribute('aria-label', 'Observation plane'); select.add(new Option('All planes', '-1'));
    layer.planes.forEach((p, i) => select.add(new Option(`Bit ${i} · ${p}`, String(i)))); select.addEventListener('change', () => {S.plane = Number(select.value); schedule();});
    controls.append(select);
    if (S.topology.architecture.history_frames === 2) {
      const time = el('select', 'input-plane-select'); time.id = 'input-frame'; time.setAttribute('aria-label', 'Observation time');
      time.add(new Option('Current frame', 'current')); time.add(new Option('Previous frame', 'previous'));
      time.addEventListener('change', () => {S.inputFrame = time.value; schedule();}); controls.append(time);
    }
    const canvas = el('canvas', 'large-map'); canvas.id = 'input-map'; canvas.setAttribute('aria-label', 'Selected input map'); root.append(controls, canvas);
    const m = MAPS.indexOf(layer.id), scales = ['2 m cells · 128 m window', '1 m cells · 64 m window', 'Whole arena at 64 × 64', 'Engine reachability and distance'];
    root.append(el('p', 'note', scales[m]));
  } else {
    const outputs = el('div', 'head-values'); outputs.id = 'head-output'; root.append(outputs);
  }
}
function renderStage(f) {
  if (!S.topology || !S.layer) return;
  const layer = S.topology.layers.find(l => l.id === S.layer); if (!layer) return;
  if (stageBuilt !== layer.id) buildStage(layer);
  $('attention-section').hidden = !layer.id.includes('_a') || !f?.attn;
  if (S.source === 'training') {
    $('activation-content').replaceChildren(el('p', 'note', 'Activations are not captured from live training. Choose Policy inspection to inspect a separate match.'));
    $('attention-section').hidden = true; return;
  }
  if (!f) return;
  const data = f.activations[layer.id], stats = f.stats?.[layer.id];
  if (layer.id === 'opponents_weights') renderHead(layer, f);
  else if (layer.kind === 'conv' && data) {
    drawMosaic($('feature-mosaic'), data, layer.shape, stats?.signed, S.page * 24, S.channel);
    drawChannel($('channel-large'), data, layer.shape, stats?.signed, S.channel);
    $('channels-range').textContent = `${S.page * 24}–${Math.min(layer.shape[0] - 1, S.page * 24 + 23)} of ${layer.shape[0]}`;
    $('channels-prev').disabled = S.page === 0; $('channels-next').disabled = (S.page + 1) * 24 >= layer.shape[0];
    if (document.activeElement !== $('channel')) $('channel').value = String(S.channel);
    statList(layer, stats);
  } else if (layer.kind === 'dense' && data) {drawVector($('vector-output'), data, stats?.signed); statList(layer, stats);}
  else if (layer.id === 'scalars_in' || layer.id === 'opponents_in') [...$('scalar-list').children].forEach((row, i) => {const index = layer.scalar_indices?.[i] ?? i + (layer.scalar_offset || 0); row.lastElementChild.textContent = fmt(f.numbers[index]); row.lastElementChild.title = String(f.numbers[index]);});
  else if (layer.kind === 'input') drawMap($('input-map'), (S.inputFrame === 'previous' && f.previous ? f.previous : f.maps)[layer.id], layer.id, S.plane);
  else if (['output', 'auxiliary'].includes(layer.kind)) renderHead(layer, f);
  if (layer.id === 'memory_context' && $('memory-readings')) {
    const root = $('memory-readings'); root.replaceChildren();
    const context = f.memory?.context;
    if (context) {const action = context.slice(0, 6).findIndex(value => value > .5); reading(root, 'Previous executed control', action >= 0 ? ACTIONS[action] : 'None'); reading(root, 'Elapsed game time', `${fmt(context[6] * .05, 3)} s`);}
  }
  if (!$('attention-section').hidden) renderAttention(layer, f);
}
function renderHead(layer, f) {
  const root = $('head-output'); root.replaceChildren(); const list = el('dl', 'readings');
  if (layer.id === 'policy') ACTIONS.slice(0, f.probs.length).forEach((a, i) => reading(list, a, f.mask[i] ? `${fmt(f.probs[i] * 100, 2)}%` : 'Masked'));
  if (layer.id === 'value') reading(list, 'Learned value', fmt(f.value));
  if (layer.id === 'aux' && f.aux) {reading(list, 'Territory in 2 s', percent(f.aux[0])); reading(list, 'Death within 2 s', percent(f.aux[1])); if (f.aux.length > 2) reading(list, 'Kill + survive in 2 s', percent(f.aux[2]));}
  if (layer.id === 'opponents_weights' && f.opponents) {
    f.opponents.forEach((weight, i) => reading(list, `${i + 1}${i === 0 ? 'st' : i === 1 ? 'nd' : 'rd'} nearest enemy`, f.numbers[64 + 8 * i + 7] > .5 ? percent(weight) : 'Absent'));
    root.append(el('p', 'note', 'Routing weights among present enemies; they are not kill probabilities.'));
  }
  root.append(list);
  if (layer.id === 'own' && f.own) {const canvas = el('canvas'); canvas.setAttribute('aria-label', 'Predicted future territory ownership'); root.append(canvas); drawOwnership(canvas, f.own); root.append(el('p', 'note', 'Cyan: mine · amber: enemy · dark: neither. Probabilities are mixed in colour, at the head’s native 16×16 resolution.'));}
}
let attentionCount = 0;
function renderAttention(layer, f) {
  const a = f.attn, layerIndex = Number(layer.id.split('_a')[1]) - 1, cells = a.side ** 2;
  const current = f.decision === latest()?.decision;
  $('follow-head').disabled = !current;
  $('attention-query').textContent = `Query cell ${a.cell % a.side}, ${Math.floor(a.cell / a.side)}. ` + (current
    ? `Click a head map to choose another cell${S.paused ? '' : '; pause to inspect it'}.`
    : 'Return to Live to change the query; this is a historical decision.');
  const root = $('attention-heads');
  if (attentionCount !== a.heads) {
    root.replaceChildren(); attentionCount = a.heads;
    for (let h = 0; h < a.heads; h++) {
      const label = el('label', '', `Head ${h + 1}`), canvas = el('canvas'); canvas.setAttribute('aria-label', `Attention head ${h + 1}`); label.append(canvas); root.append(label);
      canvas.addEventListener('click', e => {
        if (frame()?.decision !== latest()?.decision) {message('Return to Live to change the attention query.'); return;}
        const rect = canvas.getBoundingClientRect(), x = Math.min(a.side - 1, Math.floor((e.clientX - rect.left) / rect.width * a.side)), y = Math.min(a.side - 1, Math.floor((e.clientY - rect.top) / rect.height * a.side));
        control({cell: y * a.side + x});
      });
    }
  }
  [...root.querySelectorAll('canvas')].forEach((canvas, h) => {const start = (layerIndex * a.heads + h) * cells; drawHeatmap(canvas, a.rows.subarray(start, start + cells), a.side);});
}
let benchmarkRendered = null;
function resultCount(row) {return row?.rounds_completed ?? row?.rounds ?? row?.games ?? 0;}
function recordSummary(row) {
  if (!row) return 'No completed check yet';
  return `u${fmt(row.update, 0)} · ${row.wins ?? 0} wins / ${row.draws ?? 0} draws / ${row.losses ?? 0} losses · ${resultCount(row)}/${row.rounds_requested ?? resultCount(row)} games${row.partial ? ' · partial' : ''}`;
}
function renderBenchmarks() {
  const b = S.benchmark;
  if (!b) return;
  const p = b.progress;
  $('benchmark-status').textContent = b.running
    ? `${b.status === 'cancelling' ? 'Stopping check' : 'Checking frozen checkpoint'} · ${p?.update == null ? 'loading weights' : recordSummary(p)}`
    : p?.status === 'error' ? `Check failed: ${p.error || 'No final result saved.'}`
    : b.latest ? `${recordSummary(b.latest)}${b.latest.status === 'time_limit' ? ' · time budget reached' : b.latest.status === 'cancelled' ? ' · cancelled' : ''}`
    : 'No measured fixed-opponent results yet.';
  $('watch-benchmark').textContent = b.running ? `Fixed-opponent check running · ${resultCount(p)}/${p?.rounds_requested ?? 20} games. Open Progress.`
    : b.latest ? `Fixed-opponent check: ${recordSummary(b.latest)}. Open Progress.` : 'Fixed-opponent check: no result yet. Open Progress.';
  if (benchmarkRendered === b) return;
  benchmarkRendered = b;
  const root = $('benchmark-results'); root.replaceChildren();
  if (!b.results?.length) {root.append(el('p', 'note', 'Run a check to measure a frozen checkpoint against the same built-in opponent. Unfinished games are excluded.')); return;}
  const table = el('table', 'results-table'), head = el('thead'), tr = el('tr');
  ['Checkpoint', 'Opponent', 'Wins', 'Draws', 'Losses', 'Games'].forEach(x => tr.append(el('th', '', x)));
  head.append(tr); table.append(head); const body = el('tbody');
  for (const r of b.results.slice(0, 10)) {
    const row = el('tr');
    const opponent = r.opponent === 'builtin' ? 'Built-in AI' : String(r.opponent || 'Unknown').split('/').at(-1);
    [`u${fmt(r.update, 0)}`, opponent, r.wins, r.draws, r.losses, `${resultCount(r)}/${r.rounds_requested ?? resultCount(r)}${r.partial ? ' partial' : ''}`].forEach(v => row.append(el('td', '', String(v))));
    body.append(row);
    const detail = el('tr', 'result-detail'), cell = el('td'); cell.colSpan = 6;
    const policy = r.policy_mode === 'unknown' || r.sample == null ? 'policy mode not recorded' : r.sample ? 'sampled policy' : 'highest-probability policy';
    cell.textContent = `Real engine · ${policy} · no search · size ${r.size_factor ?? 'not recorded'} · ${r.time ? new Date(r.time * 1000).toLocaleTimeString() : 'time not recorded'}${r.status === 'time_limit' ? ' · time budget reached' : r.status === 'cancelled' ? ' · cancelled' : ''}`;
    detail.append(cell); body.append(detail);
  }
  table.append(body); root.append(table);
}
const METRICS = [
  ['sps', 'Decisions per second', 0], ['evals_s', 'Network evaluations per second', 0], ['ep_len', 'Round length · decisions', 1],
  ['pg', 'Policy cross-entropy', 4], ['vf', 'Value loss', 4], ['aux', 'Auxiliary loss', 4], ['safe_kill', 'Kill + survive prediction loss', 4], ['own', 'Ownership loss', 4],
  ['clip_fraction', 'Clipped fraction', 4], ['policy_kl', 'Policy change KL', 4], ['ent', 'Policy entropy · nats', 4], ['value_explained', 'Value explained variance', 4],
  ['grad_norm', 'Gradient norm · before clipping', 3], ['lr', 'Learning rate', 7], ['replay', 'Replay ratio', 2]
];
function metricRow(key, name, digits, ratio) {
  const value = S.metrics.last[key], row = el('div', 'metric-row'); row.append(el('span', 'metric-name', name), el('span', 'metric-number', ratio ? percent(value) : fmt(value, digits)));
  const canvas = el('canvas', 'metric-chart'); canvas.setAttribute('aria-label', `${name} history`); canvas._series = S.metrics.series[key] || []; canvas._ratio = ratio;
  row.append(canvas); return row;
}
function renderMetrics() {
  if (!S.metrics) return;
  const m = S.metrics.last, age = Math.max(0, Date.now() / 1000 - m.time), status = S.trainingStatus, active = status ? status.running : Number.isFinite(age) && age < Math.max(120, S.metrics.interval * 3);
  const warming = status?.phase === 'warming_up';
  $('training-state').textContent = `${warming ? 'Replay warmup' : active ? 'Training active' : 'Stopped / stale'} · last log ${age < 60 ? `${Math.round(age)} s` : age < 3600 ? `${Math.round(age / 60)} min` : `${Math.round(age / 3600)} h`} ago`;
  $('training-state').classList.toggle('training-state-active', active);
  $('training-summary').textContent = `Trainer update ${fmt(status?.update ?? m.update, 0)} · ${fmt(status?.steps ?? m.steps, 0)} decisions` + (warming ? ` · replay ${fmt(status.ready_samples, 0)} usable samples` : '') + `. Charts below are logged reports; latest report u${fmt(m.update, 0)} at ${new Date(m.time * 1000).toLocaleTimeString()}.`;
  $('training-empty').hidden = true;
  if (lastMetricRender !== S.metrics) {
    lastMetricRender = S.metrics; const root = $('training-metrics'); root.replaceChildren();
    const wins = Object.keys(m).filter(k => k.startsWith('win_')).sort();
    if (wins.length) {
      root.append(el('h3', 'metric-group', `Training matches · latest up to ${m.match_window || 400} per matchup`));
      root.append(el('p', 'note', 'Scores count draws as half a win. These matches include policy exploration.'));
      const names = {win_vs_hunter: 'Current policy vs space fighter', win_vs_past: 'Current policy vs frozen pool', win_exploiter: 'Exploiter vs frozen target'};
      for (const key of wins) {
        root.append(metricRow(key, names[key] || key.slice(4).replaceAll('_', ' '), 1, true));
        const counts = m.match_results?.[key];
        root.append(el('p', 'note outcome-counts', counts ? `${counts.wins} wins · ${counts.draws} draws · ${counts.losses} losses · ${counts.games} games` : 'Outcome counts were not recorded in this report.'));
      }
    }
    root.append(el('h3', 'metric-group', 'Learning and throughput'));
    for (const [key, name, digits] of METRICS) if (Number.isFinite(m[key])) root.append(metricRow(key, name, digits, false));
  }
  for (const canvas of document.querySelectorAll('.metric-chart')) drawSparkline(canvas, canvas._series, canvas._ratio);
}
function render() {
  updateControls(); renderBenchmarks(); const f = frame();
  if (S.view === 'decision') renderDecision(f);
  else if (S.view === 'network') renderStage(f);
  else renderMetrics();
}
const stream = new EventSource('/events');
stream.onopen = () => {S.connected = true; schedule();};
stream.onerror = () => {S.connected = false; S.probeBusy = null; schedule();};
stream.addEventListener('state', e => {
  const state = JSON.parse(e.data);
  if (state.source !== S.source || state.arena !== S.arena) resetHistory();
  Object.assign(S, {source: state.source || 'training', arena: state.arena ?? 0,
    selection: state.selection || 'sampled', paused: state.paused, speed: state.speed});
  syncSceneDisplay(); schedule();
});
stream.addEventListener('training_status', e => {S.trainingStatus = JSON.parse(e.data); schedule();});
stream.addEventListener('training_frame', e => {
  if (S.source !== 'training') return;
  const f = JSON.parse(e.data);
  if (f.arena !== S.arena || latest()?.timestamp === f.timestamp) return;
  if (latest()?.actor_pid && latest().actor_pid !== f.actor_pid) resetHistory();
  f.round = f.round_id; f.view_sample = (latest()?.view_sample || 0) + 1;
  S.frames.push(f); if (S.frames.length > 120) S.frames.shift(); schedule();
});
stream.addEventListener('benchmark', e => {S.benchmark = JSON.parse(e.data); schedule();});
stream.addEventListener('topology', e => {
  const t = JSON.parse(e.data);
  if (!t.architecture) {message('Restart the inspector server to load the new interface.'); return;}
  if (S.topology && (t.session !== S.topology.session || t.checkpoint !== S.topology.checkpoint || t.version !== S.topology.version || t.observation_version !== S.topology.observation_version)) {S.frames = []; S.selected = null; S.probe = null; S.probeBusy = null; probeRendered = null;}
  const structureChanged = !S.topology || t.version !== S.topology.version || t.observation_version !== S.topology.observation_version || t.session !== S.topology.session || t.checkpoint !== S.topology.checkpoint;
  S.topology = t;
  $('version').textContent = `V${t.version} architecture · observation ${t.observation_version}${t.architecture.memory?.persistent ? ` · ${t.architecture.memory.hidden}-unit memory` : ''}`; $('parameters').textContent = `${compact(t.architecture.parameters)} parameters`;
  $('parameters').title = `${fmt(t.architecture.parameters, 0)} trainable parameters`;
  $('checkpoint').textContent = t.checkpoint; $('checkpoint').title = t.checkpoint;
  $('update').textContent = `Update ${fmt(t.update, 0)}`;
  mapCanvases.forEach(map => {const read = t.reads.includes(map.id); map.button.classList.toggle('unread', !read); map.button.disabled = !read; map.button.title = read ? `Inspect ${MAP_NAMES[MAPS.indexOf(map.id)]} input` : `Not read by v${t.version}`;});
  if (!S.layer || !t.layers.some(l => l.id === S.layer)) S.layer = t.layers.some(l => l.id === 'arena_a1') ? 'arena_a1' : t.layers.find(l => l.kind === 'conv')?.id;
  if (structureChanged) {
    topologyRevision++; stageBuilt = null;
    const picker = $('scene-stage'); picker.replaceChildren();
    for (const layer of t.layers) picker.add(new Option(`${layer.name} · ${layer.shape.join(' × ')}`, layer.id));
    picker.disabled = false; buildGraph(); syncSceneDisplay();
  }
  schedule();
});
stream.addEventListener('frame', e => {
  if (!S.topology || S.source !== 'inspection') return;
  const f = prepareFrame(JSON.parse(e.data), S.topology);
  if (latest()?.decision === f.decision && latest()?.round === f.round && latest()?.update === f.update) return;
  S.frames.push(f); if (S.frames.length > 120) S.frames.shift(); schedule();
});
stream.addEventListener('probe', e => {S.probe = JSON.parse(e.data); S.probeBusy = null; probeRendered = null; schedule();});
stream.addEventListener('attention', e => {
  const result = JSON.parse(e.data), f = latest();
  if (f?.decision === result.decision) {f.attn = prepareAttention(result.attn); schedule();}
});
stream.addEventListener('metrics', e => {S.metrics = JSON.parse(e.data); schedule();});
setInterval(schedule, 1000);
readRoute();
