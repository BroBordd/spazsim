"""C++ pool == single-env python reference (obs, rewards, flags), and identical for any thread count."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "python"))
from spazenv import SpazEnv; from spazpool import SpazPool, max_threads
rng = np.random.default_rng(1)
# 1) reference equivalence with a forced spawn
env = SpazEnv("the_pad", seed=0); spawn = (0.5, -3.0, 123.0)
o_ref = env.reset(spawn=spawn); pool = SpazPool("the_pad", 3, seed=5, threads=1); o_p = pool.reset_one(0, *spawn)
mo = mr = 0.0; ended = None
for t in range(400):
    a = np.clip(rng.normal(0, 0.8, 4), -1, 1).astype(np.float32); a[2] = abs(a[2]); a[3] = 0.0
    o_ref, r_ref, d, tr, info = env.step(a)
    acts = np.zeros((3, 4), np.float32); acts[0] = a
    o_p, r_p, fl, _, fin = pool.step(acts)
    if d or tr or fl[0]:
        ended = (t, d, tr, int(fl[0])); mo = max(mo, float(np.abs(fin[0] - o_ref).max())); mr = max(mr, abs(r_p[0] - r_ref)); break
    mo = max(mo, float(np.abs(o_p[0] - o_ref).max())); mr = max(mr, abs(float(r_p[0]) - r_ref))
print(f"reference check over {t+1} steps: max|obs diff|={mo:.2e}  max|reward diff|={mr:.2e}  end={ended}")
assert mo < 1e-4 and mr < 1e-4
# 2) thread-count invariance (random spawns, random actions, many auto-resets)
def run(threads):
    p = SpazPool("the_pad", 16, seed=9, threads=threads); o = p.reset(); r2 = np.random.default_rng(3); out = [o.copy()]; rs = 0.0; ends = 0
    for _ in range(600):
        o, r, fl, c, _ = p.step(r2.normal(0, 1, (16, 4)).astype(np.float32)); rs += r.sum(); ends += int((fl > 0).sum())
    return o, rs, ends
a = run(1); b = run(max(2, max_threads()))
print("threads=1 vs threads=%d: obs identical=%s reward sum %.4f vs %.4f, episodes ended %d vs %d" % (max(2, max_threads()), np.array_equal(a[0], b[0]), a[1], b[1], a[2], b[2]))
assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
print("OK")
