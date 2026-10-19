/* Scientific views are drawn from engine observations and measured tensors, never a UI image. */
const CYAN = [56, 216, 232], LILAC = [189, 154, 239], AMBER = [245, 172, 73];
export const MAPS = ['local_in', 'close_in', 'global_in', 'territory_in'];
export const MAP_NAMES = ['Nearby', 'Close', 'Arena', 'Territory'];
export const ACTIONS = ['Straight', 'Left', 'Right', 'Brake', 'Left + brake', 'Right + brake'];
export const ACTION_NAMES = ['Go straight', 'Turn left', 'Turn right', 'Brake', 'Turn left + brake', 'Turn right + brake'];
export function decode(s) { return Uint8Array.from(atob(s), c => c.charCodeAt(0)); }
export function prepareFrame(f, topology) {
  const bytes = decode(f.acts);
  f.activations = {};
  let offset = 0;
  for (const layer of topology.layers) {
    if (!['conv', 'dense'].includes(layer.kind)) continue;
    const n = layer.shape.reduce((a, b) => a * b, 1);
    f.activations[layer.id] = bytes.subarray(offset, offset + n);
    offset += n;
  }
  for (const key of Object.keys(f.maps)) f.maps[key] = decode(f.maps[key]);
  if (f.previous) for (const key of Object.keys(f.previous)) f.previous[key] = decode(f.previous[key]);
  if (f.own) f.own = decode(f.own);
  if (f.attn) prepareAttention(f.attn);
  delete f.acts;
  delete f.scalars;
  return f;
}
export function prepareAttention(a) {
  if (typeof a.rows === 'string') a.rows = decode(a.rows);
  if (typeof a.got === 'string') a.got = decode(a.got);
  return a;
}
function fit(canvas, height) {
  const w = Math.max(1, canvas.clientWidth || 280);
  const h = height ?? Math.max(1, canvas.clientHeight || w);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = false;
  return {ctx, w, h};
}
function raster(bytes, width, height, color) {
  const source = document.createElement('canvas');
  source.width = width; source.height = height;
  const ctx = source.getContext('2d'), image = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const c = color(bytes[i] || 0, i);
    image.data[i * 4] = c[0]; image.data[i * 4 + 1] = c[1]; image.data[i * 4 + 2] = c[2]; image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return source;
}
function intensity(v, signed) {
  const a = signed ? Math.min(1, Math.abs(v - 127.5) / 127.5) : v / 255;
  const c = signed && v < 127.5 ? LILAC : CYAN;
  return c.map((n, i) => Math.round([17, 24, 30][i] * (1 - a) + n * a));
}
export function drawMap(canvas, packed, id, plane = -1) {
  if (!packed) return;
  const {ctx, w, h} = fit(canvas);
  ctx.fillStyle = '#10171c'; ctx.fillRect(0, 0, w, h);
  const global = id === 'global_in';
  const image = raster(packed, 64, 64, v => {
    if (plane >= 0) return intensity(v & (1 << plane) ? 255 : 0, false);
    if (id === 'territory_in') {
      const strength = .35 + .65 * (1 - (v >> 2) / 63);
      const c = (v & 1) ? CYAN : ((v & 2) ? AMBER : [24, 32, 39]);
      return c.map(n => Math.round(n * strength));
    }
    let c = [17, 24, 30];
    if (v & (1 << (global ? 5 : 6))) c = [34, 43, 51];
    if (v & 1) c = [113, 130, 144];
    if (v & 2) c = CYAN;
    if (v & 4) c = AMBER;
    if (v & (1 << (global ? 6 : 3))) c = LILAC;
    if (!global && (v & 32)) c = [117, 88, 49];
    if (v & 16) c = [255, 208, 127];
    if (global && (v & 8)) c = [182, 249, 255];
    return c;
  });
  const side = Math.min(w, h);
  ctx.drawImage(image, (w - side) / 2, (h - side) / 2, side, side);
  if (!global && id !== 'territory_in' && plane < 0) {
    ctx.fillStyle = '#d9faff';
    const x = (w - side) / 2 + side * 32 / 64, y = (h - side) / 2 + side * 40 / 64;
    ctx.beginPath(); ctx.moveTo(x, y - side * .025); ctx.lineTo(x - side * .017, y + side * .015); ctx.lineTo(x + side * .017, y + side * .015); ctx.fill();
  }
}
export function drawArena(canvas, frame) {
  const {ctx, w, h} = fit(canvas);
  ctx.fillStyle = '#12191e'; ctx.fillRect(0, 0, w, h);
  const [lx, ly, hx, hy] = frame.bounds, pad = 11;
  const scale = Math.min((w - 2 * pad) / (hx - lx), (h - 2 * pad) / (hy - ly));
  const aw = (hx - lx) * scale, ah = (hy - ly) * scale, ox = (w - aw) / 2, oy = (h - ah) / 2;
  const X = x => ox + (x - lx) * scale, Y = y => oy + (hy - y) * scale;
  ctx.strokeStyle = '#2b363e'; ctx.lineWidth = .65;
  for (let k = 1; k < 20; k++) {
    ctx.beginPath(); ctx.moveTo(ox + aw * k / 20, oy); ctx.lineTo(ox + aw * k / 20, oy + ah); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox, oy + ah * k / 20); ctx.lineTo(ox + aw, oy + ah * k / 20); ctx.stroke();
  }
  ctx.strokeStyle = '#63717b'; ctx.lineWidth = 2; ctx.strokeRect(ox, oy, aw, ah);
  ctx.save(); ctx.beginPath(); ctx.rect(ox, oy, aw, ah); ctx.clip();
  for (const trail of frame.trails || []) {
    if (!trail.points?.length && !trail.position) continue;
    ctx.strokeStyle = trail.color; ctx.globalAlpha = trail.alive ? 1 : .35;
    ctx.lineWidth = trail.id === frame.me ? 2.7 : 2.1; ctx.lineJoin = 'miter'; ctx.lineCap = 'butt';
    ctx.beginPath();
    if (Array.isArray(trail.segments)) {
      for (const [x1, y1, x2, y2] of trail.segments) {ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2));}
    } else trail.points.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
    ctx.stroke();
    if (trail.alive) {
      const [x, y] = trail.position || trail.points[trail.points.length - 1], [dx, dy] = trail.heading;
      ctx.save(); ctx.translate(X(x), Y(y)); ctx.rotate(Math.atan2(-dy, dx)); ctx.fillStyle = trail.color;
      ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-5, -5); ctx.lineTo(-3, 0); ctx.lineTo(-5, 5); ctx.closePath(); ctx.fill(); ctx.restore();
    }
  }
  ctx.restore(); ctx.globalAlpha = 1;
}
export function drawMosaic(canvas, data, shape, signed, start = 0, selected = 0) {
  const channels = shape[0], side = shape[1] || 1;
  const cols = 6, rows = Math.ceil(Math.min(24, channels - start) / cols);
  const w = Math.max(1, canvas.clientWidth), gap = w > 450 ? 24 : 12;
  const tile = Math.min(60, (w - gap * (cols - 1)) / cols), pitch = tile + 18;
  const ox = Math.max(0, (w - cols * tile - (cols - 1) * gap) / 2);
  const height = Math.max(1, rows * pitch);
  canvas.style.height = `${height}px`;
  const {ctx} = fit(canvas, height);
  ctx.font = '10px SFMono-Regular, Consolas, monospace'; ctx.textAlign = 'center';
  for (let n = start; n < Math.min(start + 24, channels); n++) {
    const k = n - start, x = ox + (k % cols) * (tile + gap), y = Math.floor(k / cols) * pitch;
    ctx.drawImage(raster(data.subarray(n * side * side, (n + 1) * side * side), side, side, v => intensity(v, signed)), x, y, tile, tile);
    ctx.strokeStyle = n === selected ? '#bd9aef' : '#435462'; ctx.lineWidth = n === selected ? 2 : .7; ctx.strokeRect(x, y, tile, tile);
    ctx.fillStyle = n === selected ? '#bd9aef' : '#94a8ba'; ctx.fillText(String(n), x + tile / 2, y + tile + 13);
  }
  canvas._layout = {cols, tile, gap, pitch, start, channels, ox};
}
export function drawChannel(canvas, data, shape, signed, channel) {
  const side = shape[1] || 1;
  const {ctx, w, h} = fit(canvas);
  ctx.drawImage(raster(data.subarray(channel * side * side, (channel + 1) * side * side), side, side, v => intensity(v, signed)), 0, 0, w, h);
}
export function drawVector(canvas, bytes, signed = false) {
  const cols = Math.min(32, Math.ceil(Math.sqrt(bytes.length * 2))), rows = Math.ceil(bytes.length / cols);
  canvas.style.height = `${Math.max(45, canvas.clientWidth / cols * rows)}px`;
  const {ctx, w, h} = fit(canvas);
  ctx.drawImage(raster(bytes, cols, rows, v => intensity(v, signed)), 0, 0, w, h);
}
export function drawHeatmap(canvas, bytes, side = 16) {
  const {ctx, w, h} = fit(canvas);
  ctx.drawImage(raster(bytes, side, side, v => intensity(v, false)), 0, 0, w, h);
}
export function drawOwnership(canvas, bytes) {
  if (!bytes) return;
  const {ctx, w, h} = fit(canvas);
  const n = bytes.length / 2, side = Math.sqrt(n);
  const image = raster(bytes.subarray(0, n), side, side, (v, i) => {
    const mine = v / 255, enemy = bytes[n + i] / 255, nobody = Math.max(0, 1 - mine - enemy);
    return CYAN.map((c, k) => Math.round(c * mine + AMBER[k] * enemy + [19, 26, 32][k] * nobody));
  });
  ctx.drawImage(image, 0, 0, w, h);
}
export function drawSparkline(canvas, series, ratio = false) {
  const {ctx, w, h} = fit(canvas);
  const values = series.filter(Number.isFinite);
  if (!values.length) return;
  let min = ratio ? 0 : Math.min(...values), max = ratio ? 1 : Math.max(...values);
  if (max === min) {
    const padding = max === 0 ? .5 : Math.max(Math.abs(max) * .05, 1e-8);
    min -= padding; max += padding;
  }
  const x = i => 48 + i / Math.max(series.length - 1, 1) * Math.max(1, w - 57);
  const y = v => h - 6 - (v - min) / (max - min) * (h - 14);
  ctx.strokeStyle = '#30414d'; ctx.lineWidth = .7;
  ctx.beginPath(); ctx.moveTo(48, h - 6); ctx.lineTo(w, h - 6); ctx.stroke();
  ctx.font = '9px SFMono-Regular, Consolas, monospace'; ctx.fillStyle = '#8297a8'; ctx.textAlign = 'right';
  const label = v => ratio ? `${Math.round(v * 100)}%` : (v !== 0 && Math.abs(v) < .001 ? v.toExponential(1) : Math.abs(v) >= 1000 ? v.toPrecision(2) : v.toFixed(Math.abs(v) < .1 ? 3 : 1));
  ctx.fillText(label(max), 40, 11); ctx.fillText(label(min), 40, h - 3);
  ctx.strokeStyle = '#38d8e8'; ctx.lineWidth = 1.4; ctx.beginPath();
  let pen = false;
  series.forEach((v, i) => { if (!Number.isFinite(v)) {pen = false;return;} if (pen) ctx.lineTo(x(i), y(v)); else ctx.moveTo(x(i), y(v)); pen = true; });
  ctx.stroke();
  if (values.length === 1) {ctx.fillStyle='#38d8e8';ctx.beginPath();ctx.arc(x(0),y(values[0]),2,0,2*Math.PI);ctx.fill();}
}
