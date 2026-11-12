"""Offline checks for brainbot: pure-python brain == numpy brain, plugin obs == trainer obs, closed-loop play."""
import sys, collections, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("tests/stubs", "python", "plugin"): sys.path.insert(0, os.path.join(ROOT, p))  # stubs stand in for babase/bascenev1
import brainbot
from spazenv import SpazEnv
from train import init_mlp, mlp_fwd, H
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "brains", "the_pad.npz")
brain = brainbot.Brain(path); print("meta:", {k: brain.meta[k] for k in ("map", "game_map", "repeat", "obs_dim")})
z = np.load(path); pi = [z[f"pi{i}"] for i in range(6)]
mean, var = z["norm_mean"], z["norm_var"]
def np_act(obs): return mlp_fwd(pi, np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32))[0]

class FakeNode:  # exposes exactly the node attributes the plugin reads, backed by the math sim
    def __init__(s, sim): s.sim = sim; s.cmd = (0, 0, 0, 0)
    def _o(s): return s.sim.state()
    torso_position = property(lambda s: tuple(float(x) for x in s._o()[0:3]))
    position = property(lambda s: tuple(float(x) for x in s._o()[12:15]))
    position_center = property(lambda s: tuple(float(x) for x in s._o()[29:32]))
    position_forward = property(lambda s: tuple(float(x) for x in s._o()[26:29]))
    velocity = property(lambda s: tuple(float(x) for x in s._o()[3:6]))
    knockout = property(lambda s: float(s._o()[23]))
    def __setattr__(s, k, v):
        if k in ("move_left_right", "move_up_down", "run", "jump_pressed"):
            c = list(s.__dict__["cmd"]); c[("move_left_right", "move_up_down", "run", "jump_pressed").index(k)] = v; s.__dict__["cmd"] = tuple(c)
        else: object.__setattr__(s, k, v)

env = SpazEnv("the_pad", seed=7); drv = brainbot.Driver(brain)
maxobs = maxact = 0.0; n = 0
obs_env = env.reset(); drv.reset_episode(); node = FakeNode(env.sim)
for step in range(300):
    cap = {}
    orig = drv.brain.act
    drv.brain.act = lambda o: (cap.__setitem__("o", np.array(o, np.float32)), orig(o))[1]
    reason = drv.think(node); drv.brain.act = orig
    maxobs = max(maxobs, float(np.abs(cap["o"] - obs_env).max()))
    a_np = np_act(obs_env); a_py = np.array(orig(list(cap["o"])))
    maxact = max(maxact, float(np.abs(a_np - a_py).max())); n += 1
    obs_env, r, d, tr, info = env.step(np.array(drv.prev_act[:3] + [0.0] if False else [node.cmd[0], node.cmd[1], node.cmd[2], 1.0 if node.cmd[3] else 0.0]))
    if d or tr or reason: break
print(f"steps compared {n}: max |obs_plugin - obs_trainer| = {maxobs:.2e}, max |act_py - act_numpy| = {maxact:.2e}")

# closed loop, plugin controller only
res = collections.Counter(); steps = []
for ep in range(20):
    env = SpazEnv("the_pad", seed=100 + ep); env.reset(); drv = brainbot.Driver(brain); node = FakeNode(env.sim)
    while True:
        reason = drv.think(node)
        for _ in range(4): env.sim.step(*node.cmd[:3], 1 if node.cmd[3] else 0)
        if reason: break
    res[reason] += 1; steps.append(drv.t)
print("closed-loop episodes (20):", dict(res), "mean steps", np.mean(steps))
