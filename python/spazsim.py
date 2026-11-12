import ctypes, os, numpy as np
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libspazsim.so"))
_lib.ss_create.restype = ctypes.c_void_p
for n in ("ss_destroy","ss_set_map","ss_reset","ss_step","ss_state"): getattr(_lib,n).restype=None
fp = ctypes.POINTER(ctypes.c_float)
c_f, c_i, c_v = ctypes.c_float, ctypes.c_int, ctypes.c_void_p
_lib.ss_destroy.argtypes=[c_v]; _lib.ss_set_map.argtypes=[c_v,fp,c_i]
_lib.ss_reset.argtypes=[c_v,c_f,c_f,c_f,c_f]; _lib.ss_step.argtypes=[c_v,c_f,c_f,c_f,c_i]; _lib.ss_state.argtypes=[c_v,fp]
class Sim:
    def __init__(self, boxes):
        self.h = ctypes.c_void_p(_lib.ss_create()); self.set_map(boxes)
    def set_map(self, boxes):
        b = np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1)
        _lib.ss_set_map(self.h, b.ctypes.data_as(fp), len(b)//6)
    def reset(self, x, y, z, ang=0.0): _lib.ss_reset(self.h, x, y, z, ang)
    def step(self, lr, ud, run=0.0, jump=0): _lib.ss_step(self.h, lr, ud, run, int(jump))
    def state(self):
        o = np.zeros(32, np.float32); _lib.ss_state(self.h, o.ctypes.data_as(fp)); return o
    def __del__(self):
        try: _lib.ss_destroy(self.h)
        except Exception: pass
