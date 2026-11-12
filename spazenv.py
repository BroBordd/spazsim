"""Training environment on top of the standalone spaz math core (no graphics)."""
import numpy as np
from spazsim import Sim

def make_map(name="holes"):
    """Maps are lists of axis-aligned boxes (minx,miny,minz,maxx,maxy,maxz); floor top is y=0."""
    if name == "pad":
        return [[-10, -2, -10, 10, 0, 10]]
    if name == "holes":
        L = 12.0
        b = [[-L, -2, -L, -2, 0, L], [2, -2, -L, L, 0, L], [-2, -2, -L, 2, 0, -2], [-2, -2, 2, 2, 0, L]]
        return b  # a 4x4 pit in the middle of a 24x24 platform
    raise ValueError(name)

N_PROBE_DIRS, PROBE_R = 16, (1.5, 3.0)
OBS_DIM, ACT_DIM = 3 + 3 + 3 + 3 + 5 + 4 + N_PROBE_DIRS * len(PROBE_R) + 2, 4

class SpazEnv:
    def __init__(self, map_name="holes", repeat=4, max_steps=600, seed=0):
        self.boxes = np.array(make_map(map_name), np.float32)
        self.sim = Sim(self.boxes); self.repeat, self.max_steps = repeat, max_steps
        self.rng = np.random.default_rng(seed)
        ang = np.linspace(0, 2 * np.pi, N_PROBE_DIRS, endpoint=False)
        self.pd = np.stack([np.cos(ang), np.sin(ang)], 1)
        self.extent = float(np.abs(self.boxes[:, [0, 2, 3, 5]]).max())
        self.reset()

    def on_floor(self, x, z):
        b = self.boxes
        return np.any((x >= b[:, 0]) & (x <= b[:, 3]) & (z >= b[:, 2]) & (z <= b[:, 5]) & (b[:, 4] >= -0.5), axis=-1)

    def reset(self):
        while True:  # spawn somewhere solid, away from edges
            x, z = self.rng.uniform(-self.extent + 1, self.extent - 1, 2)
            if all(self.on_floor(x + dx, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)): break
        self.sim.reset(float(x), -0.5, float(z), float(self.rng.uniform(0, 360)))
        self.t = 0; self.visited = set(); self.prev_act = np.zeros(4, np.float32)
        self.last = self.sim.state(); return self._obs(self.last)

    def _obs(self, o):
        rp = o[12:15]
        probes = [self.on_floor(rp[0] + d[0] * r, rp[2] + d[1] * r) for r in PROBE_R for d in self.pd]
        return np.concatenate([o[9:12], o[3:6] / 5, o[6:9] / 5, (o[12:15] - o[0:3]),
                               [o[21], min(o[22], 2) / 2, float(o[23] > 0), o[24], o[25]], self.prev_act,
                               np.array(probes, np.float32) * 2 - 1, [o[0] / self.extent, o[2] / self.extent]]).astype(np.float32)

    def step(self, a):
        a = np.asarray(a, np.float32); lr, ud = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
        run, jump = float(np.clip(a[2], 0, 1)), a[3] > 0.5
        for _ in range(self.repeat): self.sim.step(lr, ud, run, jump)
        o = self.sim.state(); self.last = o; self.t += 1; self.prev_act = np.array([lr, ud, run, float(jump)], np.float32)
        r = 0.02; done = False; info = {}
        cell = (int(np.floor(o[0])), int(np.floor(o[2])))
        if cell not in self.visited: self.visited.add(cell); r += 0.5
        r += 0.01 * min(np.hypot(o[3], o[5]), 5.0)
        if o[1] < -1.5: r -= 20.0; done = True; info["fell"] = True          # left the map
        elif o[23] > 0 or o[10] < 0.4: r -= 10.0; done = True; info["downed"] = True  # knocked out / toppled
        trunc = (not done) and self.t >= self.max_steps
        info["cells"] = len(self.visited)
        return self._obs(o), r, done, trunc, info
