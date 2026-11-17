import { test } from 'node:test';
import assert from 'node:assert/strict';
import { generateLevel, canStand, movePlayer, collectPickups, MAX_DT } from './logic.mjs';

const arena = { limit: 10, boxes: [{ minX: 1, maxX: 2, minZ: -2, maxZ: 2 }] };
const input = { forward: 0, right: 1, sprint: false };

test('fixed seed reproduces eight reachable pickups and clear spawn', () => {
  const level = generateLevel();
  assert.deepEqual(level, generateLevel(23));
  assert.notDeepEqual(level, generateLevel(24));
  assert.equal(level.pickups.length, 8);
  assert.equal(new Set(level.pickups.map((p) => `${p.x},${p.z}`)).size, 8);
  assert.ok(canStand(level.spawn.x, level.spawn.z, level));
  for (const pickup of level.pickups) {
    assert.ok(canStand(pickup.x, pickup.z, level));
    // Pickup grid lines form clear corridors through every obstacle row.
    for (let z = -8; z <= 8; z += 0.25) assert.ok(canStand(pickup.x, z, level));
  }
});

test('axis separation slides along box and arena walls', () => {
  let p = { x: 0.65, z: 0, yaw: 0 };
  for (let i = 0; i < 5; i++) p = movePlayer(p, { ...input, forward: 1 }, 0.1, arena);
  assert.ok(p.x <= 0.7);
  assert.ok(p.z < -1);
  const wall = movePlayer({ x: 9.7, z: 0, yaw: 0 }, { ...input, forward: 1 }, 0.1, arena);
  assert.equal(wall.x, 9.7);
  assert.ok(wall.z < 0);
});

test('inside and outside corners remain collision-free', () => {
  for (const start of [{ x: 0.6, z: -2.4, yaw: 0 }, { x: 9.65, z: 9.65, yaw: 0 }]) {
    let p = start;
    for (let i = 0; i < 40; i++) {
      p = movePlayer(p, { ...input, forward: -1, sprint: true }, 0.1, arena);
      assert.ok(canStand(p.x, p.z, arena));
    }
  }
});

test('large sprint dt is clamped and substeps prevent tunneling through thin boxes', () => {
  const thin = { limit: 10, boxes: [{ minX: 0.45, maxX: 0.46, minZ: -2, maxZ: 2 }] };
  const p = { x: 0, z: 0, yaw: 0 };
  const sprint = { ...input, sprint: true };
  const next = movePlayer(p, sprint, 100, thin);
  assert.deepEqual(next, movePlayer(p, sprint, MAX_DT, thin));
  assert.ok(next.x <= 0.15);
  assert.ok(canStand(next.x, next.z, thin));
});

test('diagonal speed is normalized and yaw rotates movement', () => {
  const p = { x: 0, z: 0, yaw: 0 };
  const empty = { limit: 10, boxes: [] };
  const next = movePlayer(p, { ...input, forward: 1 }, 0.1, empty);
  assert.ok(Math.abs(Math.hypot(next.x, next.z) - 0.4) < 1e-9);
  const turned = movePlayer({ ...p, yaw: Math.PI / 2 }, { ...input, right: 0, forward: 1 }, 0.1, empty);
  assert.ok(Math.abs(turned.x + 0.4) < 1e-9);
});

test('empty input, invalid dt and negative dt leave position unchanged', () => {
  const p = { x: 0, z: 0, yaw: 0 };
  for (const dt of [0, -1, NaN, Infinity]) assert.deepEqual(movePlayer(p, input, dt, arena), p);
  assert.deepEqual(movePlayer(p, { ...input, right: 0 }, 0.1, arena), p);
});

test('pickup radius includes boundary, excludes points outside, and preserves input', () => {
  const pickups = [{ x: 0.7, z: 0 }, { x: 0.701, z: 0 }, { x: 0, z: 0 }];
  assert.deepEqual(collectPickups({ x: 0, z: 0 }, pickups), [pickups[1]]);
  assert.equal(pickups.length, 3);
  assert.deepEqual(collectPickups({ x: 0, z: 0 }, []), []);
});
