"""PPO trainer (numpy only). Continuously checkpoints to a 'brain' file; rerun to resume, like an LLM run."""
import argparse, os, signal, time, json, numpy as np
from spazenv import SpazEnv, OBS_DIM, ACT_DIM, MAPS, PROBE_R, N_PROBE_DIRS, DT

H = 128
def init_mlp(rng, sizes, last_scale=0.01):
    P = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        sc = last_scale if i == len(sizes) - 2 else np.sqrt(2.0 / a)
        P += [(rng.standard_normal((a, b)) * sc).astype(np.float32), np.zeros(b, np.float32)]
    return P

def mlp_fwd(P, x):
    acts = [x]; h = x
    for i in range(0, len(P), 2):
        h = h @ P[i] + P[i + 1]
        if i < len(P) - 2: h = np.tanh(h)
        acts.append(h)
    return h, acts

def mlp_bwd(P, acts, gout):
    G = [None] * len(P); g = gout
    for i in range(len(P) - 2, -1, -2):
        a = acts[i // 2]; G[i] = a.T @ g; G[i + 1] = g.sum(0)
        if i > 0: g = (g @ P[i].T) * (1 - acts[i // 2] ** 2)
    return G

class Adam:
    def __init__(self, params, lr): self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0; self.lr = lr
    def step(self, params, grads, clip=0.5):
        n = np.sqrt(sum((g ** 2).sum() for g in grads)); s = min(1.0, clip / (n + 1e-8)); self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            g = g * s; m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

class RunNorm:
    def __init__(self, d): self.mean = np.zeros(d); self.var = np.ones(d); self.n = 1e-4
    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), len(x); d = bm - self.mean; tot = self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot; self.n = tot
    def __call__(self, x): return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

def make_meta(map_name):
    m = MAPS[map_name]
    return dict(format=2, map=map_name, game_map=m["game_map"], boxes=m["boxes"], floor_top=0.0, repeat=4, dt=DT, max_steps=600,
                obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=H, probe_dirs=N_PROBE_DIRS, probe_radii=list(PROBE_R), sim="spazsim",
                note="boxes are game-xz coordinates, y relative to floor top; observations use only in-game node attributes")

def peek_meta(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(bytes(z["meta"]).decode()) if "meta" in z.files else None

def save_brain(path, st):
    tmp = path + ".tmp.npz"
    d = {"cfg": json.dumps(st["cfg"]), "meta": np.frombuffer(json.dumps(st["meta"]).encode(), np.uint8), "iter": st["iter"], "steps": st["steps"], "best": st["best"], "log_std": st["log_std"],
         "norm_mean": st["norm"].mean, "norm_var": st["norm"].var, "norm_n": st["norm"].n, "adam_t": st["opt"].t,
         "rng": json.dumps(st["rng"].bit_generator.state)}
    for tag, P in (("pi", st["pi"]), ("vf", st["vf"])):
        for i, p in enumerate(P): d[f"{tag}{i}"] = p
    allp = st["pi"] + st["vf"] + [st["log_std"]]
    for i, (m, v) in enumerate(zip(st["opt"].m, st["opt"].v)): d[f"m{i}"] = m; d[f"v{i}"] = v
    np.savez(tmp, **d); os.replace(tmp, path)

def load_brain(path, st):
    z = np.load(path, allow_pickle=False)
    st["iter"], st["steps"], st["best"] = int(z["iter"]), int(z["steps"]), float(z["best"]); st["log_std"][:] = z["log_std"]
    for tag, P in (("pi", st["pi"]), ("vf", st["vf"])):
        for i in range(len(P)): P[i][:] = z[f"{tag}{i}"]
    st["norm"].mean, st["norm"].var, st["norm"].n = z["norm_mean"], z["norm_var"], float(z["norm_n"])
    st["opt"].t = int(z["adam_t"])
    for i in range(len(st["opt"].m)): st["opt"].m[i][:] = z[f"m{i}"]; st["opt"].v[i][:] = z[f"v{i}"]
    st["rng"].bit_generator.state = json.loads(str(z["rng"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brains", "the_pad.npz")); ap.add_argument("--map", default=None, help="training map (default: the brain's map, else the_pad)")
    ap.add_argument("--envs", type=int, default=32); ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--iters", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    old = peek_meta(a.brain) if os.path.exists(a.brain) else None
    if os.path.exists(a.brain) and old is None: raise SystemExit("brain has no metadata (old format); delete it or retrain")
    if old and a.map and a.map != old["map"]: raise SystemExit(f"brain was trained on '{old['map']}', refusing to continue on '{a.map}'")
    a.map = (old["map"] if old else a.map or "the_pad")
    pi = init_mlp(rng, [OBS_DIM, H, H, ACT_DIM]); vf = init_mlp(rng, [OBS_DIM, H, H, 1], 1.0)
    log_std = np.full(ACT_DIM, -0.5, np.float32)
    params = pi + vf + [log_std]; opt = Adam(params, a.lr)
    st = dict(cfg=vars(a), meta=make_meta(a.map), iter=0, steps=0, best=-1e9, pi=pi, vf=vf, log_std=log_std, norm=RunNorm(OBS_DIM), opt=opt, rng=rng)
    if os.path.exists(a.brain): load_brain(a.brain, st); print(f"resumed {a.brain}: iter {st['iter']}, {st['steps']} env steps")
    else: print("fresh brain")
    envs = [SpazEnv(a.map, seed=a.seed * 1000 + i + st["iter"]) for i in range(a.envs)]
    obs = np.stack([e.reset() for e in envs]); stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    ep_ret = np.zeros(a.envs); recent = []; falls = []; cells = []
    while not stop[0]:
        T, N = a.rollout, a.envs; t0 = time.time()
        O = np.zeros((T, N, OBS_DIM), np.float32); A = np.zeros((T, N, ACT_DIM), np.float32); LP = np.zeros((T, N), np.float32)
        V = np.zeros((T + 1, N), np.float32); Rw = np.zeros((T, N), np.float32); D = np.zeros((T, N), np.float32); TR = np.zeros((T, N), np.float32)
        raw_obs = []
        for t in range(T):
            raw_obs.append(obs); n = st["norm"](obs); mu, _ = mlp_fwd(pi, n); v, _ = mlp_fwd(vf, n)
            std = np.exp(log_std); act = mu + std * st["rng"].standard_normal(mu.shape).astype(np.float32)
            LP[t] = (-0.5 * ((act - mu) / std) ** 2 - log_std - 0.9189385).sum(1); O[t], A[t], V[t] = n, act, v[:, 0]
            nobs = np.empty_like(obs)
            for i, e in enumerate(envs):
                o, r, d, tr, info = e.step(act[i]); Rw[t, i] = r; ep_ret[i] += r
                if tr: TR[t, i] = 1; Rw[t, i] += 0.0
                if d or tr:
                    if tr:  # bootstrap truncated episodes with the value of the last state
                        vv, _ = mlp_fwd(vf, st["norm"](o[None])); Rw[t, i] += 0.99 * vv[0, 0]
                    D[t, i] = 1; recent.append(ep_ret[i]); falls.append(1.0 if info.get("fell") or info.get("downed") else 0.0); cells.append(info["cells"]); ep_ret[i] = 0; o = e.reset()
                nobs[i] = o
            obs = nobs
        st["norm"].update(np.concatenate(raw_obs)); st["steps"] += T * N
        vlast, _ = mlp_fwd(vf, st["norm"](obs)); V[T] = vlast[:, 0]
        adv = np.zeros((T, N), np.float32); g = 0
        for t in reversed(range(T)):
            nonterm = 1.0 - D[t]; delta = Rw[t] + 0.99 * V[t + 1] * nonterm - V[t]; g = delta + 0.99 * 0.95 * nonterm * g; adv[t] = g
        ret = adv + V[:T]
        Of, Af, LPf, advf, retf = O.reshape(-1, OBS_DIM), A.reshape(-1, ACT_DIM), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        advf = (advf - advf.mean()) / (advf.std() + 1e-8); B = len(Of)
        for ep in range(4):
            perm = st["rng"].permutation(B)
            for s in range(0, B, 1024):
                ix = perm[s:s + 1024]; x, ac = Of[ix], Af[ix]
                mu, ca = mlp_fwd(pi, x); std = np.exp(log_std)
                lp = (-0.5 * ((ac - mu) / std) ** 2 - log_std - 0.9189385).sum(1); ratio = np.exp(lp - LPf[ix]); ad = advf[ix]
                clipped = ((ad > 0) & (ratio > 1.2)) | ((ad < 0) & (ratio < 0.8)); w = np.where(clipped, 0.0, ratio * ad)[:, None]
                gmu = -w * (ac - mu) / std ** 2 / len(ix)
                gls = -(w * (((ac - mu) / std) ** 2 - 1.0)).sum(0) / len(ix) - 0.001
                Gp = mlp_bwd(pi, ca, gmu.astype(np.float32))
                v, cv = mlp_fwd(vf, x); gv = (0.5 * (v[:, 0] - retf[ix]))[:, None] * 2 / len(ix) * 0.5
                Gv = mlp_bwd(vf, cv, gv.astype(np.float32))
                opt.step(params, Gp + Gv + [gls.astype(np.float32)]); np.clip(log_std, -2.5, 0.5, out=log_std)
        st["iter"] += 1; mr = float(np.mean(recent[-50:])) if recent else 0.0
        if recent and mr > st["best"]: st["best"] = mr
        print(f"iter {st['iter']:5d} steps {st['steps']:9d} ret {mr:7.2f} fail {np.mean(falls[-50:]) if falls else 0:4.2f} cells {np.mean(cells[-50:]) if cells else 0:5.1f} std {np.exp(log_std).mean():.2f} {T*N/(time.time()-t0):6.0f} sps", flush=True)
        if st["iter"] % a.save_every == 0: save_brain(a.brain, st)
        if a.iters and st["iter"] >= a.iters: break
    save_brain(a.brain, st); print("saved", a.brain)

if __name__ == "__main__": main()
