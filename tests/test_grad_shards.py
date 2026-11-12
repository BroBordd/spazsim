"""Sharded (multi-thread) PPO gradients == single-shard gradients."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "python"))
from concurrent.futures import ThreadPoolExecutor
import train
from spazenv import OBS_DIM, ACT_DIM
rng = np.random.default_rng(0)
pi = train.init_mlp(rng, [OBS_DIM, 128, 128, ACT_DIM], 0.3); vf = train.init_mlp(rng, [OBS_DIM, 128, 128, 1], 0.3); ls = np.full(ACT_DIM, -0.5, np.float32)
m = 1024; x = rng.normal(0, 1, (m, OBS_DIM)).astype(np.float32); ac = rng.normal(0, 1, (m, ACT_DIM)).astype(np.float32)
olp = rng.normal(-4, .3, m).astype(np.float32); ad = rng.normal(0, 1, m).astype(np.float32); ret = rng.normal(0, 1, m).astype(np.float32)
with ThreadPoolExecutor(8) as ex:
    g1 = train.parallel_grads(ex, 1, pi, vf, ls, x, ac, olp, ad, ret); g8 = train.parallel_grads(ex, 8, pi, vf, ls, x, ac, olp, ad, ret)
err = max(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-12)) for a, b in zip(g1, g8))
print(f"max relative gradient difference, 1 shard vs 8 shards: {err:.2e}"); assert err < 1e-4; print("OK")
