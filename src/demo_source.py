# demo_source.py
#
# Simulated cart scan over a beam with stirrups. Used for
#   * `python dashboard.py --demo`  (booth fallback / rehearsal without hardware)
#   * the engine's validation tests (known ground truth).
#
# This is SIMULATED data -- the dashboard labels it as such. The signal model is
# a Gaussian bump per rebar in Bx (like the LIS2 Bx trace in your screenshot),
# smaller opposite-sign bumps in By/Bz, an offset + slow drift, white noise, and
# a much lower relative noise on the quantum channel.

from __future__ import annotations

import math

import numpy as np


class DemoScan:
    def __init__(self, seed=0, length_cm=480.0, spacing_cm=50.0, first_cm=60.0,
                 jitter_cm=4.0, v_cm_s=22.0, hz=15.0, artifacts=False,
                 lis_noise=0.02, quantum_noise=0.4, quantum_online=True, rebars=None):
        self.rng = np.random.default_rng(seed)
        self.length, self.v, self.hz = length_cm, v_cm_s, hz
        self.artifacts = artifacts
        self.lis_noise, self.quantum_noise = lis_noise, quantum_noise
        self.quantum_online = quantum_online
        if rebars is None:
            n = int((length_cm - first_cm - 25) // spacing_cm) + 1
            rebars = first_cm + spacing_cm * np.arange(n) + self.rng.uniform(-jitter_cm, jitter_cm, n)
        self.truth = np.sort(np.asarray(rebars, dtype=float))
        self.amp = self.rng.uniform(0.8, 1.2, len(self.truth))
        self.width = self.rng.uniform(4.5, 6.5, len(self.truth))    # gaussian sigma [cm]
        self.drift_phase = self.rng.uniform(0, 6.28, 3)

    def _bump(self, x):
        g = np.zeros_like(np.asarray(x, dtype=float))
        for p, a, w in zip(self.truth, self.amp, self.width):
            g = g + a * np.exp(-0.5 * ((x - p) / w) ** 2)
        return g

    def field(self, x):
        """Noise-free (lis Bx, By, Bz) [G] and quantum (Bx, By, Bz) [a.u.] at position x."""
        b = self._bump(x)
        drift = 0.03 * math.sin(x / 90.0 + self.drift_phase[0])
        lis = (-0.9 + 0.5 * b + drift, 0.6 - 0.32 * b, 0.4 - 0.2 * b)
        quantum = (20.0 + 60.0 * b + 30 * drift, -8.0 - 35.0 * b, 5.0 + 18.0 * b)
        return lis, quantum

    def samples(self):
        """Yield dicts: pos, lis2, lis3, quantum (tuples), fresh (bool). Chronological."""
        rng = self.rng
        x = 0.0
        dt = 1.0 / self.hz
        k = 0
        spike_at = set(rng.integers(40, 300, 6).tolist()) if self.artifacts else set()
        while x < self.length:
            v = self.v * (1.0 + 0.15 * math.sin(k / 37.0)) * rng.uniform(0.9, 1.1)
            x += v * dt
            k += 1
            if self.artifacts:
                # startup: cart position only reported sparsely for the first ~130 cm
                if x < 130 and k % 6:
                    continue
                # BLE dropout: ~35 cm without samples
                if 300 < x < 335:
                    continue
            pos = round(x + rng.normal(0, 0.15), 2)
            lis, q = self.field(x)
            lis2 = tuple(c + rng.normal(0, self.lis_noise) for c in lis)
            lis3 = tuple(c + rng.normal(0, self.lis_noise * 1.25) + off
                         for c, off in zip(lis, (0.3, -0.2, 0.1)))
            qs = tuple(c + rng.normal(0, self.quantum_noise) for c in q)
            if k in spike_at:
                lis2 = (lis2[0] + rng.choice([-1, 1]) * 1.5, lis2[1], lis2[2])
                lis3 = (lis3[0], lis3[1] + rng.choice([-1, 1]) * 1.5, lis3[2])
            yield {"pos": pos, "lis2": lis2, "lis3": lis3, "quantum": qs,
                   "fresh": self.quantum_online}


def as_snapshot(scan):
    """Materialise a whole scan into the engine's snapshot format."""
    pos, l2, l3, q = [], [], [], []
    for s in scan.samples():
        pos.append(s["pos"]); l2.append(s["lis2"]); l3.append(s["lis3"])
        q.append(s["quantum"] if s["fresh"] else (np.nan,) * 3)
    l2, l3, q = np.array(l2), np.array(l3), np.array(q)
    mk = lambda a: {"x": a[:, 0], "y": a[:, 1], "z": a[:, 2]}
    return {"pos": np.array(pos), "lis2": mk(l2), "lis3": mk(l3), "quantum": mk(q)}
