"""Batched, multithreaded environment pool (all stepping/obs/rewards/resets run in C++)."""
import ctypes, os, numpy as np
from spazsim import _lib
from spazenv import MAPS, OBS_DIM, ACT_DIM

fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
_lib.sp_create.restype = ctypes.c_void_p
_lib.sp_create.argtypes = [ctypes.c_int, fp, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.sp_destroy.argtypes = [ctypes.c_void_p]; _lib.sp_destroy.restype = None
_lib.sp_reset_all.argtypes = [ctypes.c_void_p, fp]; _lib.sp_reset_all.restype = None
_lib.sp_reset_one.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, fp]; _lib.sp_reset_one.restype = None
_lib.sp_step.argtypes = [ctypes.c_void_p, fp, fp, fp, ip, ip, fp]; _lib.sp_step.restype = None
_lib.sp_max_threads.restype = ctypes.c_int

def max_threads(): return _lib.sp_max_threads()
_P = lambda a, t: a.ctypes.data_as(t)

class SpazPool:
    """flags: 0 running, 1 fell, 2 downed (knocked out/toppled), 3 time-limit truncation."""
    def __init__(self, map_name, n_envs, seed=0, threads=0, repeat=4, max_steps=600):
        self.n = n_envs; self.threads = threads or max_threads()
        b = np.ascontiguousarray(MAPS[map_name]["boxes"], np.float32).reshape(-1)
        self.h = ctypes.c_void_p(_lib.sp_create(n_envs, _P(b, fp), len(b) // 6, seed, repeat, max_steps, self.threads))
        self.obs = np.zeros((n_envs, OBS_DIM), np.float32); self.rew = np.zeros(n_envs, np.float32)
        self.flags = np.zeros(n_envs, np.int32); self.cells = np.zeros(n_envs, np.int32); self.final = np.zeros((n_envs, OBS_DIM), np.float32)
    def reset(self): _lib.sp_reset_all(self.h, _P(self.obs, fp)); return self.obs.copy()
    def reset_one(self, i, x, z, ang):
        o = np.zeros(OBS_DIM, np.float32); _lib.sp_reset_one(self.h, i, x, z, ang, _P(o, fp)); self.obs[i] = o; return o
    def step(self, act):
        a = np.ascontiguousarray(act, np.float32)
        _lib.sp_step(self.h, _P(a, fp), _P(self.obs, fp), _P(self.rew, fp), _P(self.flags, ip), _P(self.cells, ip), _P(self.final, fp))
        return self.obs.copy(), self.rew.copy(), self.flags.copy(), self.cells.copy(), self.final
    def __del__(self):
        try: _lib.sp_destroy(self.h)
        except Exception: pass
