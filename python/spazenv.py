"""Training environment on top of the standalone spaz math core (no graphics).

Observations use ONLY what the live game exposes on a spaz node (position, torso_position,
position_center, position_forward, velocity, knockout) so a trained brain can drive a real
in-game spaz. Maps are lists of boxes in *game xz coordinates*, y relative to the floor top (0)."""
import numpy as np
from spazsim import Sim

# game_map = bascenev1 map name used by the plugin. boxes: (minx,miny,minz,maxx,maxy,maxz).
MAPS = {
    # Footprint approximated from The Pad's area_of_interest bounds (real collide mesh isn't in the repo).
    "the_pad": dict(game_map="The Pad", boxes=[[-7.5, -2, -11.3, 8.2, 0, 6.2]]),
    # training-only maps (no in-game counterpart)
    "holes": dict(game_map=None, boxes=[[-12, -2, -12, -2, 0, 12], [2, -2, -12, 12, 0, 12], [-2, -2, -12, 2, 0, -2], [-2, -2, 2, 2, 0, 12]]),
}
N_PROBE_DIRS, PROBE_R = 16, (1.5, 3.0)
OBS_DIM, ACT_DIM, DT = 58, 4, 0.032
PD = np.stack([np.cos(np.linspace(0, 2 * np.pi, N_PROBE_DIRS, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, N_PROBE_DIRS, endpoint=False))], 1)

def map_geometry(boxes):
    b = np.array(boxes, np.float32)
    cx, cz = (b[:, 0].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 5].max()) / 2
    ext = max(b[:, 3].max() - b[:, 0].min(), b[:, 5].max() - b[:, 2].min()) / 2
    return b, float(cx), float(cz), float(ext)

def on_floor(b, x, z):
    return np.any((x >= b[:, 0]) & (x <= b[:, 3]) & (z >= b[:, 2]) & (z <= b[:, 5]) & (b[:, 4] >= -0.5), axis=-1)

def build_obs(T, R, C, F, V, knock, prev_up, prev_R, prev_act, floor_y, boxes, cx, cz, ext):
    """Shared observation definition (plugin re-implements this in pure python; tests compare them)."""
    T, R, C, F, V = (np.asarray(v, np.float64) for v in (T, R, C, F, V))
    H = (C - 0.7 * T) / 0.3
    up = H - T; n = np.linalg.norm(up); up = up / n if n > 1e-6 else np.array([0, 1.0, 0])
    fw = (F - T) / 0.2
    probes = [on_floor(boxes, R[0] + d[0] * r, R[2] + d[1] * r) for r in PROBE_R for d in PD]
    return np.concatenate([up, fw, V / 5, R - T, (up - prev_up) * 10, (R - prev_R) / DT / 5,
                           [T[1] - floor_y, float(knock > 0)], prev_act, np.array(probes) * 2.0 - 1.0,
                           [(R[0] - cx) / ext, (R[2] - cz) / ext]]).astype(np.float32), up

class SpazEnv:
    def __init__(self, map_name="the_pad", repeat=4, max_steps=600, seed=0):
        self.map_name = map_name; self.boxes, self.cx, self.cz, self.ext = map_geometry(MAPS[map_name]["boxes"])
        self.sim = Sim(self.boxes); self.repeat, self.max_steps = repeat, max_steps
        self.rng = np.random.default_rng(seed); self.reset()

    def reset(self):
        b = self.boxes
        while True:  # spawn somewhere solid, away from edges
            x = self.rng.uniform(self.cx - self.ext, self.cx + self.ext); z = self.rng.uniform(self.cz - self.ext, self.cz + self.ext)
            if all(on_floor(b, x + dx, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)): break
        self.sim.reset(float(x), -0.5, float(z), float(self.rng.uniform(0, 360)))
        self.t = 0; self.visited = set(); self.prev_act = np.zeros(4, np.float32)
        o = self.sim.state(); self.last = o; self.prev_up = None; self.prev_R = o[12:15].astype(np.float64)
        return self._obs(o)

    def _obs(self, o):
        if self.prev_up is None:  # first observation of an episode: zero finite differences
            H = (o[29:32].astype(np.float64) - 0.7 * o[0:3]) / 0.3; u = H - o[0:3]; self.prev_up = u / np.linalg.norm(u)
        ob, up = build_obs(o[0:3], o[12:15], o[29:32], o[26:29], o[3:6], o[23], self.prev_up, self.prev_R, self.prev_act, 0.0,
                           self.boxes, self.cx, self.cz, self.ext)
        self.prev_up = up; self.prev_R = o[12:15].astype(np.float64); return ob

    def step(self, a):
        a = np.asarray(a, np.float32); lr, ud = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
        run, jump = float(np.clip(a[2], 0, 1)), a[3] > 0.5
        for _ in range(self.repeat): self.sim.step(lr, ud, run, jump)
        o = self.sim.state(); self.last = o; self.t += 1; self.prev_act = np.array([lr, ud, run, float(jump)], np.float32)
        r = 0.02; done = False; info = {}
        cell = (int(np.floor(o[0])), int(np.floor(o[2])))
        if cell not in self.visited: self.visited.add(cell); r += 0.5
        r += 0.01 * min(np.hypot(o[3], o[5]), 5.0)
        if o[1] < -1.5: r -= 20.0; done = True; info["fell"] = True
        elif o[23] > 0 or o[10] < 0.4: r -= 10.0; done = True; info["downed"] = True
        trunc = (not done) and self.t >= self.max_steps
        info["cells"] = len(self.visited)
        return self._obs(o), r, done, trunc, info
