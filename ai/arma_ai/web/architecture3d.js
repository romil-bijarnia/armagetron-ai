import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const COLORS = {
  nearby: '#38d8e8', close: '#38d8e8', local: '#38d8e8', player: '#38d8e8',
  arena: '#bd9aef', global: '#bd9aef', territory: '#bd9aef',
  opponents: '#f5ac49', scalars: '#94a8ba', trunk: '#c6d5de', memory: '#74bfae', head: '#c6d5de',
};
const FIT_DIRECTION = new THREE.Vector3(0.42, 0.48, 1).normalize();
const FLOW_GAP = 1.2;
const LANE_GAP = 6.7;

function colourFor(group) { return COLORS[group] || '#a9becb'; }

function uniqueEdges(layers, architecture) {
  const ids = new Set(layers.map(layer => layer.id)), seen = new Set();
  return (architecture.edges || []).filter(edge => {
    if (!Array.isArray(edge) || edge.length !== 2 || !ids.has(edge[0]) || !ids.has(edge[1]) || edge[0] === edge[1]) return false;
    const key = JSON.stringify(edge);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Longest dependency depth, computed from the actual directed graph. */
function dependencyDepth(layers, edges) {
  const depth = new Map(), incoming = new Map(), outgoing = new Map();
  for (const layer of layers) {
    depth.set(layer.id, 0);
    incoming.set(layer.id, 0);
    outgoing.set(layer.id, []);
  }
  for (const [source, target] of edges) {
    incoming.set(target, incoming.get(target) + 1);
    outgoing.get(source).push(target);
  }
  const queue = layers.filter(layer => incoming.get(layer.id) === 0).map(layer => layer.id);
  for (let index = 0; index < queue.length; index++) {
    const source = queue[index];
    for (const target of outgoing.get(source)) {
      depth.set(target, Math.max(depth.get(target), depth.get(source) + 1));
      incoming.set(target, incoming.get(target) - 1);
      if (incoming.get(target) === 0) queue.push(target);
    }
  }
  if (queue.length !== layers.length) throw new Error('The measured architecture contains a dependency cycle.');
  return depth;
}

/** Cuboids encode tensor dimensions on a compressed scale, never neuron counts. */
function dimensions(layer) {
  const shape = layer.shape || [1];
  if (shape.length >= 2) {
    const rows = shape[shape.length - 2], columns = shape[shape.length - 1];
    const spatial = value => 0.8 + 1.6 * Math.sqrt(Math.max(1, value) / 64);
    return new THREE.Vector3(spatial(columns), shape.length === 3 ? 0.14 + Math.log2(shape[0] + 1) * 0.04 : 0.14, spatial(rows));
  }
  return new THREE.Vector3(0.75 + Math.log2(Math.max(1, shape[0]) + 1) * 0.15, 0.32, 0.55);
}

function boundsFor(nodes) {
  const bounds = new THREE.Box3();
  for (const node of nodes) {
    const half = node.size.clone().multiplyScalar(0.5);
    bounds.expandByPoint(node.mesh.position.clone().sub(half));
    bounds.expandByPoint(node.mesh.position.clone().add(half));
  }
  return bounds;
}

function disposeTree(object) {
  const geometries = new Set(), materials = new Set();
  object.traverse(child => {
    if (child.geometry) geometries.add(child.geometry);
    if (child.material) {
      for (const material of Array.isArray(child.material) ? child.material : [child.material]) materials.add(material);
    }
  });
  geometries.forEach(geometry => geometry.dispose());
  materials.forEach(material => material.dispose());
  object.clear();
}

/** Interactive view of the stages and dependencies reported by the loaded model. */
export class ArchitectureScene {
  constructor(container, onSelect = () => {}) {
    if (!(container instanceof HTMLElement)) throw new TypeError('ArchitectureScene needs a container element.');
    this.container = container;
    this.onSelect = onSelect;
    this.nodes = new Map();
    this.branchLabels = [];
    this.edges = [];
    this.selectedId = null;
    this.topology = null;
    this.structure = null;
    this.visible = true;
    this.disposed = false;
    this.frame = null;
    this.width = 0;
    this.height = 0;
    this.pendingFit = false;
    this.neighbours = new Set();
    this.pointerIds = new Set();
    this.pointerStart = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.projected = new THREE.Vector3();
    this.abort = new AbortController();

    // Create WebGL before attaching DOM; callers can show a fallback if unavailable.
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor('#101316', 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setSize(1, 1, false);
    this.canvas = this.renderer.domElement;
    this.canvas.className = 'architecture-canvas';
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', 'Interactive 3D network architecture. Drag to orbit, scroll to zoom, right-drag or arrow keys to pan. Home fits the model; F focuses the selected stage.');
    this.canvas.style.display = 'block';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.touchAction = 'none';

    this.labels = document.createElement('div');
    this.labels.className = 'scene-labels';
    Object.assign(this.labels.style, { position: 'absolute', inset: '0', overflow: 'hidden', pointerEvents: 'none' });
    container.append(this.canvas, this.labels);

    this.scene = new THREE.Scene();
    this.model = new THREE.Group();
    this.scene.add(this.model);
    this.scene.add(new THREE.HemisphereLight('#d6e5ed', '#26333c', 2.4));
    const keyLight = new THREE.DirectionalLight('#ffffff', 3);
    keyLight.position.set(-12, 25, 24);
    this.scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight('#87a6b8', 1.2);
    fillLight.position.set(16, -12, -10);
    this.scene.add(fillLight);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 2000);
    this.camera.position.copy(FIT_DIRECTION).multiplyScalar(45);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.controls.enableDamping = !this.motionPreference.matches;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.controls.rotateSpeed = 0.75;
    this.controls.zoomSpeed = 0.9;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 500;
    this.controls.minPolarAngle = 0.04;
    this.controls.maxPolarAngle = Math.PI - 0.04;
    this.controlChange = () => this.requestRender();
    this.controls.addEventListener('change', this.controlChange);

    const listenerOptions = { signal: this.abort.signal };
    this.motionPreference.addEventListener('change', event => {
      this.controls.enableDamping = !event.matches;
      this.requestRender();
    }, listenerOptions);
    this.canvas.addEventListener('keydown', event => this.handleKey(event), listenerOptions);
    this.canvas.addEventListener('pointerdown', event => {
      this.pointerIds.add(event.pointerId);
      if (this.pointerIds.size > 1) { this.pointerStart = null; return; }
      this.pointerStart = event.button === 0 ? { x: event.clientX, y: event.clientY, id: event.pointerId, time: performance.now(), moved: false } : null;
      this.canvas.focus({ preventScroll: true });
    }, listenerOptions);
    this.canvas.addEventListener('pointermove', event => {
      if (this.pointerStart && Math.hypot(event.clientX - this.pointerStart.x, event.clientY - this.pointerStart.y) > 5) this.pointerStart.moved = true;
      if (!event.buttons && event.pointerType !== 'touch') this.canvas.style.cursor = this.pick(event.clientX, event.clientY) ? 'pointer' : 'grab';
    }, listenerOptions);
    this.canvas.addEventListener('pointerup', event => {
      const start = this.pointerStart;
      this.pointerIds.delete(event.pointerId);
      this.pointerStart = null;
      if (!start || start.id !== event.pointerId || start.moved || performance.now() - start.time > 1000) return;
      const id = this.pick(event.clientX, event.clientY);
      if (id) this.activate(id);
    }, listenerOptions);
    this.canvas.addEventListener('pointercancel', event => {
      this.pointerIds.delete(event.pointerId);
      this.pointerStart = null;
    }, listenerOptions);
    this.canvas.addEventListener('lostpointercapture', event => {
      this.pointerIds.delete(event.pointerId);
    }, listenerOptions);
    this.labels.addEventListener('click', event => {
      const button = event.target.closest('button[data-layer]');
      if (button && this.labels.contains(button)) this.activate(button.dataset.layer);
    }, listenerOptions);
    this.labels.addEventListener('focusin', () => this.requestRender(), listenerOptions);
    this.labels.addEventListener('keydown', event => this.handleKey(event), listenerOptions);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  setTopology(topology) {
    if (this.disposed) return;
    const layers = topology?.layers || [], architecture = topology?.architecture || {};
    const edges = uniqueEdges(layers, architecture);
    const signature = JSON.stringify({
      layers: layers.map(({ id, group, kind, shape, short, name }) => ({ id, group, kind, shape, short, name })),
      branches: architecture.branches, context: architecture.context_stages, edges,
    });
    this.topology = topology;
    if (signature === this.structure) return;
    const depth = dependencyDepth(layers, edges);
    this.structure = signature;
    disposeTree(this.model);
    this.nodes.clear();
    this.branchLabels = [];
    this.edges = [];
    this.labels.replaceChildren();
    const branches = architecture.branches || [], branchFor = new Map(), laneFor = new Map();
    branches.forEach((branch, index) => {
      const lane = { x: (index - (branches.length - 1) / 2) * LANE_GAP, z: branch.id === 'arena' || branch.id === 'global' ? -1.6 : branch.id === 'scalars' ? -2 : index % 2 ? 0.7 : 1.2 };
      laneFor.set(branch.id, lane);
      for (const id of [...(branch.inputs || []), ...(branch.stages || [])]) branchFor.set(id, branch.id);
    });
    const contextStages = new Set(architecture.context_stages || []);
    const maxDepth = Math.max(0, ...depth.values());
    const fused = group => group === 'trunk' || group === 'memory';
    const heads = layers.filter(layer => layer.group === 'head' && !edges.some(([source, target]) => target === layer.id && !fused(layers.find(candidate => candidate.id === source)?.group)));
    const ownershipSources = new Map(edges.filter(([source, target]) => layers.find(layer => layer.id === target)?.kind === 'auxiliary' && !fused(layers.find(layer => layer.id === source)?.group)).map(([source, target]) => [target, source]));

    for (const layer of layers) {
      const branch = branchFor.get(layer.id) || layer.group;
      const lane = laneFor.get(branch) || { x: 0, z: 0 };
      const position = new THREE.Vector3(lane.x, (maxDepth / 2 - depth.get(layer.id)) * FLOW_GAP, lane.z);
      const branchDefinition = branches.find(candidate => candidate.id === branch);
      const inputIndex = branchDefinition?.inputs.indexOf(layer.id) ?? -1;
      if (inputIndex >= 0 && branchDefinition.inputs.length > 1) position.x += (inputIndex - (branchDefinition.inputs.length - 1) / 2) * 2.8;
      if (contextStages.has(layer.id)) {
        position.x += branch === 'opponents' ? -2.3 : 2.2;
        position.z += 2.7;
      }
      if (fused(layer.group)) position.set(0, position.y, 1.1);
      if (layer.id === 'memory_in' || layer.id === 'memory_context') {
        position.set(layer.id === 'memory_in' ? -3 : 3,
          (maxDepth / 2 - depth.get('memory_state') + 1) * FLOW_GAP, 3.1);
      }
      const headIndex = heads.findIndex(candidate => candidate.id === layer.id);
      if (headIndex >= 0) position.set((headIndex - (heads.length - 1) / 2) * 4.6, position.y, 1.1);
      const ownershipSource = ownershipSources.get(layer.id);
      if (ownershipSource) {
        const sourceLayer = layers.find(candidate => candidate.id === ownershipSource);
        const sourceLane = laneFor.get(branchFor.get(ownershipSource) || sourceLayer.group) || { x: 0, z: 0 };
        position.set(sourceLane.x + 3.3, (maxDepth / 2 - depth.get(ownershipSource)) * FLOW_GAP, sourceLane.z - 3.4);
      }
      const size = dimensions(layer), color = colourFor(ownershipSource ? 'arena' : branch);
      const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
      const material = new THREE.MeshStandardMaterial({ color, roughness: 0.63, metalness: 0.05, emissive: color, emissiveIntensity: 0.025 });
      material.color.multiplyScalar(0.48);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(position);
      mesh.userData.layerId = layer.id;
      const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }));
      mesh.add(outline);
      this.model.add(mesh);
      const label = document.createElement('button');
      label.type = 'button';
      label.className = 'scene-stage-label';
      label.dataset.layer = layer.id;
      label.style.setProperty('--stage-color', color);
      Object.assign(label.style, { position: 'absolute', left: '0', top: '0', pointerEvents: 'auto', visibility: 'hidden' });
      label.setAttribute('aria-pressed', 'false');
      label.title = `${layer.id}: ${layer.name || layer.short || layer.id}\n${layer.operation || ''}\nTensor: ${(layer.shape || []).join(' × ')}`;
      const name = document.createElement('span'); name.className = 'scene-stage-name'; name.textContent = layer.short || layer.name || layer.id;
      const shape = document.createElement('small'); shape.className = 'scene-stage-shape'; shape.textContent = (layer.shape || []).join(' × ');
      label.append(name, shape);
      label.setAttribute('aria-label', `${layer.name || layer.id}; tensor ${(layer.shape || []).join(' by ')}; select stage`);
      this.labels.append(label);
      const priority = layer.kind === 'input' ? 8 : layer.group === 'head' ? 10 : contextStages.has(layer.id) ? 7 : layer.id.endsWith('_fc') || fused(layer.group) ? 6 : layer.operation?.startsWith('Self-attention') ? 5 : 1;
      this.nodes.set(layer.id, { layer, branch, mesh, outline, label, size, color, priority, depth: depth.get(layer.id) });
    }

    for (const branch of branches) {
      const inputs = (branch.inputs || []).map(id => this.nodes.get(id)).filter(Boolean);
      if (!inputs.length) continue;
      const position = new THREE.Vector3();
      inputs.forEach(input => position.add(input.mesh.position));
      position.divideScalar(inputs.length);
      position.y += 1.5;
      const label = document.createElement('span');
      label.className = 'scene-branch-label';
      label.textContent = branch.name || branch.id;
      Object.assign(label.style, {
        position: 'absolute', left: '0', top: '0', visibility: 'hidden',
        color: colourFor(branch.id), fontFamily: 'var(--mono, monospace)',
        fontSize: '11px', fontWeight: '650', lineHeight: '1.3', whiteSpace: 'nowrap',
        padding: '4px 6px', background: '#101316e8', borderBottom: `1px solid ${colourFor(branch.id)}`,
        pointerEvents: 'none',
      });
      this.labels.append(label);
      this.branchLabels.push({ label, position });
    }

    for (const [sourceId, targetId] of edges) this.addEdge(sourceId, targetId);
    this.bounds = boundsFor(this.nodes.values());
    this.canvas.dataset.stageCount = String(this.nodes.size);
    this.canvas.dataset.edgeCount = String(this.edges.length);
    this.canvas.dataset.architectureVersion = String(topology?.version ?? '');
    this.canvas.dataset.observationVersion = String(topology?.observation_version ?? '');
    this.select(this.nodes.has(this.selectedId) ? this.selectedId : null);
    this.pendingFit = true;
    this.fit();
  }

  addEdge(sourceId, targetId) {
    const source = this.nodes.get(sourceId), target = this.nodes.get(targetId);
    const start = source.mesh.position.clone(), end = target.mesh.position.clone();
    const sideLink = Math.abs(start.y - end.y) < 0.2;
    if (sideLink) {
      const sign = Math.sign(end.x - start.x) || 1;
      start.x += source.size.x * 0.5 * sign;
      end.x -= target.size.x * 0.5 * sign;
    } else {
      start.y -= source.size.y * 0.5;
      end.y += target.size.y * 0.5;
    }
    const difference = end.clone().sub(start), c1 = start.clone(), c2 = end.clone();
    if (sideLink) {
      c1.x += difference.x * 0.45;
      c2.x -= difference.x * 0.45;
    } else {
      const bend = Math.max(0.25, Math.abs(difference.y) * 0.45);
      c1.y -= bend;
      c2.y += bend;
      if (source.branch !== target.branch) { c1.z += 0.35; c2.z += 0.35; }
    }
    const curve = new THREE.CubicBezierCurve3(start, c1, c2, end);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(32)), new THREE.LineBasicMaterial({ color: source.color, transparent: true, opacity: 0.4 }));
    const highlight = new THREE.Mesh(new THREE.TubeGeometry(curve, 32, 0.024, 5, false), new THREE.MeshBasicMaterial({ color: source.color, transparent: true, opacity: 0.95 }));
    highlight.visible = false;
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.064, 0.21, 6), new THREE.MeshBasicMaterial({ color: source.color, transparent: true, opacity: 0.7 }));
    arrow.position.copy(curve.getPoint(0.94));
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(0.94).normalize());
    this.model.add(line, highlight, arrow);
    this.edges.push({ source: sourceId, target: targetId, line, highlight, arrow, color: source.color });
  }

  activate(id) {
    if (!this.nodes.has(id)) return;
    this.select(id);
    this.onSelect(id);
  }

  select(stageId) {
    if (this.disposed) return;
    this.selectedId = this.nodes.has(stageId) ? stageId : null;
    this.neighbours.clear();
    for (const edge of this.edges) {
      const related = this.selectedId !== null && (edge.source === this.selectedId || edge.target === this.selectedId);
      if (related) {
        this.neighbours.add(edge.source);
        this.neighbours.add(edge.target);
      }
      edge.highlight.visible = related;
      edge.line.material.opacity = related ? 0.95 : this.selectedId ? 0.2 : 0.4;
      edge.arrow.material.opacity = related ? 1 : this.selectedId ? 0.35 : 0.7;
      const color = related ? this.nodes.get(this.selectedId).color : edge.color;
      edge.highlight.material.color.set(color);
      edge.arrow.material.color.set(color);
    }
    for (const [id, node] of this.nodes) {
      const selected = id === this.selectedId, related = this.neighbours.has(id);
      node.mesh.material.color.set(node.color).multiplyScalar(selected ? 0.85 : related ? 0.6 : 0.48);
      node.mesh.material.emissiveIntensity = selected ? 0.18 : 0.025;
      node.outline.material.color.set(selected ? '#eef8ff' : node.color);
      node.outline.material.opacity = selected ? 1 : related ? 0.95 : 0.8;
      node.label.classList.toggle('selected', selected);
      node.label.classList.toggle('related', related && !selected);
      node.label.setAttribute('aria-pressed', String(selected));
    }
    this.canvas.dataset.selectedLayer = this.selectedId || '';
    this.requestRender();
  }

  setVisible(visible) {
    if (this.disposed) return;
    this.visible = Boolean(visible);
    this.controls.enabled = this.visible;
    if (!this.visible) {
      if (this.frame !== null) cancelAnimationFrame(this.frame);
      this.frame = null;
      return;
    }
    this.resize();
    if (this.pendingFit) this.fit();
    this.requestRender();
  }

  resize() {
    if (this.disposed) return;
    const width = this.container.clientWidth, height = this.container.clientHeight;
    if (width < 1 || height < 1) return;
    const firstSize = this.width === 0 || this.height === 0;
    this.width = width;
    this.height = height;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (this.pendingFit || firstSize && this.nodes.size) this.fit();
    this.requestRender();
  }

  fit() {
    if (this.disposed || !this.nodes.size) return;
    if (this.container.clientWidth < 1 || this.container.clientHeight < 1) { this.pendingFit = true; return; }
    this.pendingFit = false;
    this.frameBounds(this.bounds, FIT_DIRECTION, 1.045, [...this.nodes.values()]);
  }

  focus(stageId = this.selectedId) {
    if (this.disposed || !this.nodes.has(stageId)) return;
    const ids = new Set([stageId]);
    for (const edge of this.edges) {
      if (edge.source === stageId) ids.add(edge.target);
      if (edge.target === stageId) ids.add(edge.source);
    }
    const bounds = boundsFor([...ids].map(id => this.nodes.get(id)));
    bounds.expandByScalar(0.35);
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    this.frameBounds(bounds, direction, 1.14);
  }

  frameBounds(bounds, direction, margin, nodes = null) {
    if (bounds.isEmpty() || !this.width || !this.height) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(this.camera.up, direction).normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize();
    // Leave room around the model for screen-sized labels, including on mobile.
    const vertical = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 0.9;
    const horizontal = vertical * this.camera.aspect * (this.width < 600 ? 0.8 : 0.91);
    const points = [];
    // Fitting the enclosing box invents empty corners around the branching DAG.
    // Actual cuboid corners give the whole model more useful screen space.
    for (const box of nodes ? nodes.map(node => boundsFor([node])) : [bounds]) {
      for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) points.push(new THREE.Vector3(x, y, z));
    }
    if (nodes) this.branchLabels.forEach(branch => points.push(branch.position));
    let distance = 1;
    for (const point of points) {
      const relative = point.clone().sub(center);
      const towardCamera = relative.dot(direction);
      distance = Math.max(distance, towardCamera + Math.abs(relative.dot(up)) / vertical, towardCamera + Math.abs(relative.dot(right)) / horizontal);
    }
    distance *= margin;
    // Clear residual orbit momentum before explicit Fit/Focus positioning.
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, distance);
    this.camera.lookAt(center);
    this.controls.update();
    this.controls.enableDamping = damping;
    const modelSize = this.bounds.getSize(new THREE.Vector3()).length();
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = Math.max(100, modelSize * 12);
    this.requestRender();
  }

  pick(clientX, clientY) {
    if (!this.visible || !this.nodes.size) return null;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    this.pointer.set((clientX - rect.left) / rect.width * 2 - 1, 1 - (clientY - rect.top) / rect.height * 2);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    return this.raycaster.intersectObjects([...this.nodes.values()].map(node => node.mesh), false)[0]?.object.userData.layerId || null;
  }

  handleKey(event) {
    if (!this.visible || event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === 'Home') this.fit();
    else if (event.key.toLowerCase() === 'f') this.focus();
    else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
      const amount = this.camera.position.distanceTo(this.controls.target) * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * (event.shiftKey ? 0.15 : 0.05);
      const translation = new THREE.Vector3();
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') translation.setFromMatrixColumn(this.camera.matrix, 0).multiplyScalar(event.key === 'ArrowLeft' ? -amount : amount);
      else translation.setFromMatrixColumn(this.camera.matrix, 1).multiplyScalar(event.key === 'ArrowDown' ? -amount : amount);
      this.camera.position.add(translation);
      this.controls.target.add(translation);
      this.controls.update();
      this.requestRender();
    } else return;
    event.preventDefault();
    event.stopPropagation();
  }

  updateLabels() {
    const candidates = [];
    for (const branch of this.branchLabels) {
      this.projected.copy(branch.position).project(this.camera);
      const x = (this.projected.x * 0.5 + 0.5) * this.width;
      const y = (0.5 - this.projected.y * 0.5) * this.height;
      const label = branch.label;
      if (this.projected.z < -1 || this.projected.z > 1 || x < 0 || x > this.width || y < 0 || y > this.height) {
        label.style.visibility = 'hidden'; continue;
      }
      const width = label.offsetWidth || 90, height = label.offsetHeight || 24;
      candidates.push({ label, left: Math.max(5, Math.min(this.width - width - 5, x - width / 2)), top: Math.max(5, Math.min(this.height - height - 5, y - height)), width, height, priority: 200, depth: this.projected.z });
    }
    for (const [id, node] of this.nodes) {
      this.projected.copy(node.mesh.position).project(this.camera);
      const x = (this.projected.x * 0.5 + 0.5) * this.width;
      const y = (0.5 - this.projected.y * 0.5) * this.height;
      const selected = id === this.selectedId, focused = document.activeElement === node.label;
      if (this.projected.z < -1 || this.projected.z > 1 || x < 0 || x > this.width || y < 0 || y > this.height) {
        node.label.style.visibility = 'hidden'; node.label.tabIndex = -1; continue;
      }
      // Labels keep their true CSS size while the scene zooms. They never obscure
      // a selected label, and omitted labels return when the user moves closer.
      const width = node.label.offsetWidth || 138, height = node.label.offsetHeight || 36;
      const left = Math.max(5, Math.min(this.width - width - 5, x + 8));
      const top = Math.max(5, Math.min(this.height - height - 5, y + 4));
      const priority = focused ? 1000 : selected ? 900 : this.neighbours.has(id) ? 100 : node.priority;
      candidates.push({ label: node.label, node, left, top, width, height, priority, depth: this.projected.z, selected, focused });
    }
    candidates.sort((a, b) => b.priority - a.priority || a.depth - b.depth);
    const placed = [];
    for (const candidate of candidates) {
      const { label, node, left, top, width, height, selected, focused } = candidate;
      const overlaps = placed.some(other => left < other.left + other.width + 5 && left + width + 5 > other.left && top < other.top + other.height + 4 && top + height + 4 > other.top);
      const visible = selected || focused || !overlaps;
      label.style.visibility = visible ? 'visible' : 'hidden';
      if (node) label.tabIndex = visible ? 0 : -1;
      if (!visible) continue;
      label.style.transform = `translate3d(${Math.round(left)}px, ${Math.round(top)}px, 0)`;
      label.style.zIndex = selected || focused ? '3' : node && this.neighbours.has(node.layer.id) ? '2' : '1';
      placed.push(candidate);
    }
  }

  requestRender() {
    if (this.disposed || !this.visible || !this.width || !this.height || this.frame !== null) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = null;
      if (this.disposed || !this.visible) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.updateLabels();
    });
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.abort.abort();
    this.resizeObserver.disconnect();
    this.controls.removeEventListener('change', this.controlChange);
    this.controls.dispose();
    disposeTree(this.model);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.canvas.remove();
    this.labels.remove();
    this.nodes.clear();
    this.branchLabels = [];
    this.edges = [];
  }
}
