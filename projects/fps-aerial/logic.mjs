export const PLAYER_RADIUS = 0.3;
export const PICKUP_RADIUS = 0.7;
export const WALK_SPEED = 4;
export const SPRINT_SPEED = 7;
export const MAX_DT = 0.1;
const COLLISION_STEP = PLAYER_RADIUS / 2;

export function generateLevel(seed = 23) {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const shuffle = (items) => {
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  };
  const cells = [];
  for (const x of [-6, -2, 2, 6]) {
    for (const z of [-6, -2, 2, 6]) cells.push({ x, z });
  }
  const boxes = shuffle(cells).slice(0, 10).map(({ x, z }) => ({
    minX: x - 1, maxX: x + 1, minZ: z - 1, maxZ: z + 1,
    height: 1.2 + random() * 1.4,
  }));
  const spots = [];
  for (const x of [-8, -4, 0, 4, 8]) {
    for (const z of [-8, -4, 0, 4, 8]) {
      if (x !== 0 || z !== 8) spots.push({ x, z });
    }
  }
  return {
    limit: 9.5, boxes,
    pickups: shuffle(spots).slice(0, 8).map((p, id) => ({ ...p, id })),
    spawn: { x: 0, z: 8 },
  };
}

export function canStand(x, z, level, radius = PLAYER_RADIUS) {
  if (Math.abs(x) > level.limit - radius || Math.abs(z) > level.limit - radius) return false;
  return level.boxes.every((box) => {
    const closestX = Math.max(box.minX, Math.min(x, box.maxX));
    const closestZ = Math.max(box.minZ, Math.min(z, box.maxZ));
    return Math.hypot(x - closestX, z - closestZ) >= radius;
  });
}

export function movePlayer(player, input, dt, level) {
  const elapsed = Number.isFinite(dt) ? Math.max(0, Math.min(dt, MAX_DT)) : 0;
  const length = Math.max(1, Math.hypot(input.forward, input.right));
  const speed = input.sprint ? SPRINT_SPEED : WALK_SPEED;
  const sin = Math.sin(player.yaw), cos = Math.cos(player.yaw);
  const dx = (input.right * cos - input.forward * sin) / length * speed * elapsed;
  const dz = (-input.forward * cos - input.right * sin) / length * speed * elapsed;
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dz) / COLLISION_STEP));
  let { x, z } = player;
  for (let i = 0; i < steps; i++) {
    if (canStand(x + dx / steps, z, level)) x += dx / steps;
    if (canStand(x, z + dz / steps, level)) z += dz / steps;
  }
  return { ...player, x, z };
}

export function collectPickups(player, pickups, radius = PICKUP_RADIUS) {
  return pickups.filter((p) => Math.hypot(player.x - p.x, player.z - p.z) > radius);
}
