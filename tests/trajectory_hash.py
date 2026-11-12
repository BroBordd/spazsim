"""Bit-exact fingerprint of the physics core: scripted inputs, hash of every state. Must never change under refactors."""
import os, sys, hashlib, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from spazsim import Sim
from spazenv import MAPS
h = hashlib.sha256()
for name in ("the_pad", "holes"):
    s = Sim(np.array(MAPS[name]["boxes"], np.float32)); rng = np.random.default_rng(42)
    for ep in range(6):
        s.reset(float(rng.uniform(-3, 3)), -0.5, float(rng.uniform(-3, 3)), float(rng.uniform(0, 360)))
        for t in range(1500):
            if t % 20 == 0: a = rng.uniform(-1, 1, 3).astype(np.float32); j = int(rng.random() < 0.05); run = float(abs(a[2]))
            s.step(float(a[0]), float(a[1]), run, j); h.update(s.state().tobytes())
print(h.hexdigest())
