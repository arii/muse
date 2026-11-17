import * as THREE from 'three';
import { generateLevel, movePlayer, collectPickups } from './logic.mjs';

const EYE_HEIGHT = 1.55;
const TRANSITION_SECONDS = 0.4;
const LOOK_SENSITIVITY = 0.0025;
const MAX_PITCH = Math.PI / 2 - 0.08;
const AERIAL_HEIGHT = 24;
const canvas = document.querySelector('#scene');
const mode = document.querySelector('#mode');
const toggle = document.querySelector('#toggle');
const counter = document.querySelector('#counter strong');
const hint = document.querySelector('#hint');
const win = document.querySelector('#win');
const joystick = document.querySelector('#joystick');
const stick = document.querySelector('#stick');
const lookZone = document.querySelector('#look-zone');
const keys = new Set();
const touchMove = { right: 0, forward: 0 };
const level = generateLevel();
let player = { ...level.spawn, yaw: 0 };
let remaining = [...level.pickups];
let pitch = 0;
let aerial = false;
let blend = 0;
let completed = false;

if (matchMedia('(pointer: coarse)').matches) {
  hint.textContent = 'Left stick to move. Drag right to look. Tap Aerial view to see above.';
}

function firstInput() { hint.hidden = true; }
function clearInput() {
  keys.clear();
  if (movePointer !== null && joystick.hasPointerCapture(movePointer)) joystick.releasePointerCapture(movePointer);
  if (lookPointer !== null && lookZone.hasPointerCapture(lookPointer)) lookZone.releasePointerCapture(lookPointer);
  movePointer = lookPointer = null;
  touchMove.right = touchMove.forward = 0;
  stick.style.transform = '';
}
function look(dx, dy) {
  player.yaw -= dx * LOOK_SENSITIVITY;
  pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch - dy * LOOK_SENSITIVITY));
}
function switchView() {
  firstInput();
  aerial = !aerial;
  if (document.pointerLockElement) document.exitPointerLock();
  mode.textContent = aerial ? 'Aerial' : 'First person';
  toggle.innerHTML = `${aerial ? 'First person' : 'Aerial view'} <kbd>V</kbd>`;
  document.querySelector('#crosshair').hidden = aerial;
}
toggle.addEventListener('click', switchView);
document.addEventListener('keydown', (event) => {
  if (!['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight', 'KeyV'].includes(event.code)) return;
  event.preventDefault();
  firstInput();
  keys.add(event.code);
  if (event.code === 'KeyV' && !event.repeat) switchView();
});
document.addEventListener('keyup', (event) => keys.delete(event.code));
window.addEventListener('blur', clearInput);
document.addEventListener('visibilitychange', clearInput);
canvas.addEventListener('click', async () => {
  firstInput();
  if (!aerial && !completed && matchMedia('(pointer: fine)').matches) {
    try { await canvas.requestPointerLock(); }
    catch (error) { hint.hidden = false; hint.textContent = `Mouse capture failed: ${error.message}. Try clicking the arena again.`; }
  }
});
document.addEventListener('mousemove', (event) => {
  if (document.pointerLockElement === canvas) look(event.movementX, event.movementY);
});

let movePointer = null;
function updateJoystick(event) {
  const rect = joystick.getBoundingClientRect();
  const dx = event.clientX - rect.left - rect.width / 2;
  const dy = event.clientY - rect.top - rect.height / 2;
  const reach = rect.width * 0.32;
  const scale = Math.max(1, Math.hypot(dx, dy) / reach);
  touchMove.right = dx / scale / reach;
  touchMove.forward = -dy / scale / reach;
  stick.style.transform = `translate(${dx / scale}px, ${dy / scale}px)`;
}
joystick.addEventListener('pointerdown', (event) => {
  if (movePointer !== null) return;
  firstInput();
  movePointer = event.pointerId;
  joystick.setPointerCapture(event.pointerId);
  updateJoystick(event);
});
joystick.addEventListener('pointermove', (event) => { if (event.pointerId === movePointer) updateJoystick(event); });
joystick.addEventListener('lostpointercapture', () => {
  movePointer = null;
  touchMove.right = touchMove.forward = 0;
  stick.style.transform = '';
});
let lookPointer = null;
let lastLook = { x: 0, y: 0 };
lookZone.addEventListener('pointerdown', (event) => {
  if (lookPointer !== null) return;
  firstInput();
  lookPointer = event.pointerId;
  lastLook = { x: event.clientX, y: event.clientY };
  lookZone.setPointerCapture(event.pointerId);
});
lookZone.addEventListener('pointermove', (event) => {
  if (event.pointerId !== lookPointer) return;
  look(event.clientX - lastLook.x, event.clientY - lastLook.y);
  lastLook = { x: event.clientX, y: event.clientY };
});
lookZone.addEventListener('lostpointercapture', () => { lookPointer = null; });

function startGame() {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setClearColor(0x101c29);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x101c29, 16, 46);
  scene.add(new THREE.HemisphereLight(0xcbe9ff, 0x384252, 2.5));
  const sun = new THREE.DirectionalLight(0xffe0ad, 3);
  sun.position.set(-5, 12, 8);
  scene.add(sun);
  const camera = new THREE.PerspectiveCamera(68, 1, 0.05, 100);
  const material = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.85 });
  const floorMaterial = material(0x22394a);
  const wallMaterial = material(0x3b5968);
  const boxMaterial = material(0x8b9b9c);
  function box(width, height, depth, x, z, surface) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), surface);
    mesh.position.set(x, height / 2, z);
    scene.add(mesh);
    return mesh;
  }
  box(20, 0.1, 20, 0, 0, floorMaterial).position.y = -0.08;
  scene.add(new THREE.GridHelper(20, 20, 0x7095a6, 0x385362));
  for (const x of [-10, 10]) box(1, 2.8, 21, x, 0, wallMaterial);
  for (const z of [-10, 10]) box(21, 2.8, 1, 0, z, wallMaterial);
  for (const obstacle of level.boxes) {
    box(obstacle.maxX - obstacle.minX, obstacle.height, obstacle.maxZ - obstacle.minZ,
      (obstacle.minX + obstacle.maxX) / 2, (obstacle.minZ + obstacle.maxZ) / 2, boxMaterial);
  }
  const pickupMaterial = new THREE.MeshStandardMaterial({ color: 0xffbc57, emissive: 0xf89b32, emissiveIntensity: 0.8, roughness: 0.3 });
  const pickupGeometry = new THREE.OctahedronGeometry(0.28);
  const pickupMeshes = new Map();
  for (const p of level.pickups) {
    const mesh = new THREE.Mesh(pickupGeometry, pickupMaterial);
    mesh.position.set(p.x, 0.7, p.z);
    scene.add(mesh);
    pickupMeshes.set(p.id, mesh);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.44, 32), new THREE.MeshBasicMaterial({ color: 0xe6ab54, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(p.x, 0.015, p.z);
    mesh.userData.ring = ring;
    scene.add(ring);
  }
  const marker = new THREE.Group();
  const markerMaterial = new THREE.MeshBasicMaterial({ color: 0x7df6e0, depthTest: false });
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.08, 24), markerMaterial);
  marker.add(disc);
  const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.65, 3), markerMaterial);
  arrow.rotation.x = -Math.PI / 2;
  arrow.position.z = -0.5;
  marker.add(arrow);
  marker.renderOrder = 10;
  scene.add(marker);
  const firstRotation = new THREE.Quaternion();
  const topRotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0, 'YXZ'));
  const firstPosition = new THREE.Vector3();
  const topPosition = new THREE.Vector3();
  const euler = new THREE.Euler(0, 0, 0, 'YXZ');
  function resize() {
    renderer.setSize(innerWidth, innerHeight, false);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();
  document.querySelector('#restart').addEventListener('click', () => {
    player = { ...level.spawn, yaw: 0 };
    pitch = 0;
    remaining = [...level.pickups];
    completed = false;
    win.hidden = true;
    counter.textContent = '0 / 8';
    clearInput();
    for (const mesh of pickupMeshes.values()) mesh.visible = mesh.userData.ring.visible = true;
    canvas.focus();
  });
  let previous = performance.now();
  function frame(now) {
    const dt = Math.max(0, (now - previous) / 1000);
    previous = now;
    if (!completed) {
      const forward = Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown')) + touchMove.forward;
      const right = Number(keys.has('KeyD') || keys.has('ArrowRight')) - Number(keys.has('KeyA') || keys.has('ArrowLeft')) + touchMove.right;
      player = movePlayer(player, { forward, right, sprint: keys.has('ShiftLeft') || keys.has('ShiftRight') }, dt, level);
      const next = collectPickups(player, remaining);
      if (next.length !== remaining.length) {
        remaining = next;
        const ids = new Set(remaining.map((p) => p.id));
        for (const [id, mesh] of pickupMeshes) mesh.visible = mesh.userData.ring.visible = ids.has(id);
        counter.textContent = `${level.pickups.length - remaining.length} / ${level.pickups.length}`;
        if (remaining.length === 0) {
          completed = true;
          clearInput();
          document.exitPointerLock();
          win.hidden = false;
          document.querySelector('#restart').focus();
        }
      }
    }
    const target = aerial ? 1 : 0;
    blend += Math.sign(target - blend) * Math.min(Math.abs(target - blend), Math.min(dt, 0.1) / TRANSITION_SECONDS);
    const eased = blend * blend * (3 - 2 * blend);
    firstPosition.set(player.x, EYE_HEIGHT, player.z);
    topPosition.set(player.x, AERIAL_HEIGHT, player.z);
    camera.position.lerpVectors(firstPosition, topPosition, eased);
    euler.set(pitch, player.yaw, 0);
    firstRotation.setFromEuler(euler);
    camera.quaternion.slerpQuaternions(firstRotation, topRotation, eased);
    marker.position.set(player.x, 0.15, player.z);
    marker.rotation.y = player.yaw;
    marker.visible = blend > 0.3;
    for (const mesh of pickupMeshes.values()) {
      mesh.rotation.y = now / 900;
      mesh.position.y = 0.7 + Math.sin(now / 450 + mesh.id) * 0.08;
    }
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

try { startGame(); }
catch (error) {
  const message = document.querySelector('#error');
  message.textContent = `Could not start the arena: ${error.message}. Enable WebGL and reload.`;
  message.hidden = false;
  console.error(error);
}
