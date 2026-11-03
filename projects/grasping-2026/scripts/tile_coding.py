"""Sutton-style tile coding over continuous states (dependency-free).

One TileCoder maps a bounded float vector to one active tile per
tiling; linear SARSA keeps one weight vector per discrete action
and sums active tiles for Q(s, a). (Tiling, coords) pairs hash
into a fixed memory (Sutton & Barto 12.7), so resolution is set
by tiles/tilings while memory stays flat; collisions are extra
(random) generalization. Offsets are deterministic and asymmetric
per dimension; selected dims wrap (angles). The hash is a fixed
FNV-1a mix (never Python hash(): salted per process).

Thesis use: tile (q0, q1, q2, target_x, target_y, target_yaw) for
the 27-action ({-0.1, 0, +0.1}^3) SARSA port of Anders (2014).
"""
import itertools

import numpy as np


class TileCoder:
    """Hashed tilings with per-dim offsets and optional wrap."""

    def __init__(self, lows, highs, tiles, tilings=8, wrap=(),
                 memory=16384):
        """lows/highs/tiles: per-dim sequences; wrap: dim indices;
        memory: hash table size (one weight vector per action)."""
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.tiles = np.asarray(tiles, dtype=int)
        self.tilings = int(tilings)
        if not (len(self.lows) == len(self.highs) == len(self.tiles)):
            raise ValueError("lows/highs/tiles must agree in length")
        if np.any(self.highs <= self.lows) or np.any(self.tiles < 1):
            raise ValueError("need highs > lows and tiles >= 1")
        self.wrap = frozenset(wrap)
        if any(w < 0 or w >= len(self.tiles) for w in self.wrap):
            raise ValueError(f"wrap dims out of range: {wrap}")
        self.memory = int(memory)
        if self.memory < 1:
            raise ValueError("memory must be >= 1")
        self._widths = self.highs - self.lows
        # Deterministic asymmetric offsets: tiling i shifts dim d by
        # a distinct fraction (prime steps mod a large prime, so no
        # two dims share an offset pattern whatever the tiling count).
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        while len(primes) < len(self.tiles):
            primes.append(primes[-1] + 2)
        self._offsets = np.array(
            [[((i * primes[d]) % 1009) / 1009.0
              for d in range(len(self.tiles))]
             for i in range(self.tilings)])

    @property
    def n_tiles(self):
        """Hash memory size (one weight vector per action)."""
        return self.memory

    @staticmethod
    def _hash(tiling, coords, memory):
        """FNV-1a mix of (tiling, coords) into [0, memory)."""
        h = 2166136261
        h ^= tiling + 0x9e3779b9
        h = (h * 16777619) & 0xFFFFFFFF
        for c in coords:
            h ^= (int(c) + 0x9e3779b9) & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFF
        return h % memory

    def tiles_for(self, state):
        """Active tile index per tiling (int array, len tilings)."""
        x = np.asarray(state, dtype=float)
        if x.shape != self.lows.shape:
            raise ValueError(f"state shape {x.shape} != {self.lows.shape}")
        out = np.empty(self.tilings, dtype=int)
        for i in range(self.tilings):
            coords = []
            for d in range(len(self.tiles)):
                v = x[d]
                if d in self.wrap:
                    span = self._widths[d]
                    v = self.lows[d] + ((v - self.lows[d]) % span)
                else:
                    v = min(max(v, self.lows[d]), self.highs[d] - 1e-9)
                q = (v - self.lows[d]) / self._widths[d]
                c = int((q + self._offsets[i][d]) * self.tiles[d])
                # Wrap dims tile the circle; plain dims let offset
                # tilings overflow past tiles[d]-1 into distinct hash
                # buckets (clamping would collapse edge tiles and
                # destroy offset diversity there).
                coords.append(c % self.tiles[d] if d in self.wrap
                              else c)
            out[i] = self._hash(i, coords, self.memory)
        return out

    def all_actions(self):
        """27 joint-increment actions, {-0.1, 0, +0.1}^3 (thesis)."""
        return np.array(list(itertools.product((-0.1, 0.0, 0.1),
                                               repeat=3)))
