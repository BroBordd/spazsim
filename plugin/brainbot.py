# ba_meta require api 9
"""brainbot - plays a trained *spazsim* brain inside a real BombSquad FFA session.

Usage (from the in-game python console / any script)::

    import brainbot; brainbot.start("path/to/brain.npz")
    brainbot.stop()

What it does
  * reads the brain file (pure python, no numpy needed) and its metadata (which map it was trained on),
  * starts a fresh Free-for-All session on that map (same path as a normal FFA start, join screen skipped),
  * spawns a bot spaz and drives it with the brain (policy runs every 4 game steps = 32 ms, as in training),
  * when the bot falls / is knocked down / the episode times out, it respawns and goes again.

The brain only sees what a live spaz node exposes (position, torso_position, position_center,
position_forward, velocity, knockout), so the same file that was trained headless drives the real game.
"""
# pylint: disable=too-many-lines,import-error
from __future__ import annotations

import array
import ast
import json
import logging
import math
import os
import random
import struct
import sys
import zipfile
from operator import mul
from typing import Any

import babase
import bascenev1 as bs
from bascenev1lib.actor.spaz import Spaz
from bascenev1lib.game.deathmatch import DeathMatchGame

__all__ = ['start', 'stop', 'Brain', 'BrainGame']

STEP_SECONDS = 0.032  # 4 game steps (8 ms each); must match training ('repeat' x kGameStepSeconds)
ROLLER_RADIUS = 0.3  # spaz roller ball radius at ball_size 1
_log = logging.getLogger('brainbot')


# --------------------------------------------------------------- brain file ----
_DTYPES = {'<f4': 'f', '<f8': 'd', '|u1': 'B', '<i8': 'q', '<i4': 'i', '<u4': 'I'}


def _read_npy(buf: bytes) -> Any:
    """Minimal .npy reader -> (array.array | list, shape)."""
    if buf[:6] != b'\x93NUMPY':
        raise ValueError('not an npy member')
    major = buf[6]
    if major == 1:
        (hlen,) = struct.unpack('<H', buf[8:10])
        off = 10
    else:
        (hlen,) = struct.unpack('<I', buf[8:12])
        off = 12
    header = ast.literal_eval(buf[off:off + hlen].decode('latin1'))
    descr, shape = header['descr'], tuple(header['shape'])
    if header.get('fortran_order'):
        raise ValueError('fortran-order arrays unsupported')
    if descr not in _DTYPES:
        raise ValueError(f'unsupported dtype {descr}')
    arr = array.array(_DTYPES[descr])
    arr.frombytes(buf[off + hlen:])
    if sys.byteorder == 'big' and descr != '|u1':
        arr.byteswap()
    return arr, shape


class Brain:
    """A trained policy (deterministic mean action) evaluated in pure python."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with zipfile.ZipFile(path) as zf:
            files = {n[:-4]: zf.read(n) for n in zf.namelist() if n.endswith('.npy')}
        if 'meta' not in files:
            raise ValueError('brain file has no metadata (old format); retrain with the current trainer')
        meta_arr, _ = _read_npy(files['meta'])
        self.meta: dict[str, Any] = json.loads(bytes(meta_arr).decode())
        self.mean = list(_read_npy(files['norm_mean'])[0])
        self.var = list(_read_npy(files['norm_var'])[0])
        self.layers: list[tuple[list[list[float]], list[float]]] = []
        i = 0
        while f'pi{2 * i}' in files:
            w, (n_in, n_out) = _read_npy(files[f'pi{2 * i}'])
            b, _ = _read_npy(files[f'pi{2 * i + 1}'])
            rows = [list(w[r * n_out:(r + 1) * n_out]) for r in range(n_in)]
            cols = [list(c) for c in zip(*rows)]  # one list per output unit
            self.layers.append((cols, list(b)))
            i += 1
        if not self.layers or len(self.mean) != self.meta['obs_dim']:
            raise ValueError('brain file is inconsistent')

    def act(self, obs: list[float]) -> list[float]:
        x = [max(-10.0, min(10.0, (o - m) / math.sqrt(v + 1e-8)))
             for o, m, v in zip(obs, self.mean, self.var)]
        last = len(self.layers) - 1
        for li, (cols, bias) in enumerate(self.layers):
            x = [sum(map(mul, x, c)) + b for c, b in zip(cols, bias)]
            if li != last:
                x = [math.tanh(v) for v in x]
        return x


# ----------------------------------------------------------- observation ----
class Driver:
    """Turns spaz-node attributes into the brain's observation and back into controls.

    Mirrors spazenv.build_obs (numpy) exactly; keep the two in sync."""

    def __init__(self, brain: Brain) -> None:
        m = brain.meta
        self.brain = brain
        self.boxes: list[list[float]] = m['boxes']
        xs0 = min(b[0] for b in self.boxes)
        xs1 = max(b[3] for b in self.boxes)
        zs0 = min(b[2] for b in self.boxes)
        zs1 = max(b[5] for b in self.boxes)
        self.cx, self.cz = (xs0 + xs1) / 2, (zs0 + zs1) / 2
        self.ext = max(xs1 - xs0, zs1 - zs0) / 2
        n = m['probe_dirs']
        self.dirs = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]
        self.radii: list[float] = m['probe_radii']
        self.dt: float = m['dt']
        self.max_steps: int = m['max_steps']
        self.floor_y = float(m.get('floor_top', 0.0))
        self.reset_episode()

    def reset_episode(self) -> None:
        self.t = 0
        self.prev_up: list[float] | None = None
        self.prev_r: list[float] | None = None
        self.prev_act = [0.0, 0.0, 0.0, 0.0]

    def on_floor(self, x: float, z: float) -> bool:
        for b in self.boxes:
            if b[0] <= x <= b[3] and b[2] <= z <= b[5] and b[4] >= -0.5:
                return True
        return False

    def observe(self, node: Any) -> tuple[list[float], float, float]:
        """Returns (obs, up_y, torso_height_above_floor)."""
        t = list(node.torso_position)
        r = list(node.position)
        c = list(node.position_center)
        f = list(node.position_forward)
        v = list(node.velocity)
        knock = float(node.knockout)
        h = [(c[i] - 0.7 * t[i]) / 0.3 for i in range(3)]
        up = [h[i] - t[i] for i in range(3)]
        n = math.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2)
        up = [u / n for u in up] if n > 1e-6 else [0.0, 1.0, 0.0]
        if self.prev_up is None:
            self.prev_up = list(up)
        if self.prev_r is None:
            self.prev_r = list(r)
        fw = [(f[i] - t[i]) / 0.2 for i in range(3)]
        o: list[float] = []
        o += up
        o += fw
        o += [x / 5 for x in v]
        o += [r[i] - t[i] for i in range(3)]
        o += [(up[i] - self.prev_up[i]) * 10 for i in range(3)]
        o += [(r[i] - self.prev_r[i]) / self.dt / 5 for i in range(3)]
        o += [t[1] - self.floor_y, 1.0 if knock > 0 else 0.0]
        o += self.prev_act
        for rad in self.radii:
            for dx, dz in self.dirs:
                o.append(1.0 if self.on_floor(r[0] + dx * rad, r[2] + dz * rad) else -1.0)
        o += [(r[0] - self.cx) / self.ext, (r[2] - self.cz) / self.ext]
        self.prev_up, self.prev_r = up, r
        return o, up[1], t[1] - self.floor_y

    def think(self, node: Any, apply: bool = True) -> str | None:
        """One policy step. Returns an episode-end reason or None."""
        obs, up_y, height = self.observe(node)
        a = self.brain.act(obs)
        lr = max(-1.0, min(1.0, a[0]))
        ud = max(-1.0, min(1.0, a[1]))
        run = max(0.0, min(1.0, a[2]))
        jump = a[3] > 0.5
        if apply:
            node.move_left_right = lr
            node.move_up_down = ud
            node.run = run
            node.jump_pressed = bool(jump)
        self.prev_act = [lr, ud, run, 1.0 if jump else 0.0]
        self.t += 1
        if height < -1.5:
            return 'fell'
        if float(node.knockout) > 0 or up_y < 0.4:
            return 'downed'
        if self.t >= self.max_steps:
            return 'timeout'
        return None


# ------------------------------------------------------------ the activity ----
class _Run:
    """State of the current playback (module singleton)."""

    brain: Brain
    driver: Driver
    game_map: str
    episodes = 0
    stopped = False
    calibrated = False


_RUN: _Run | None = None


class BrainGame(DeathMatchGame):
    """A Free-for-All game with no human players: one brain-driven spaz."""

    name = 'Brain Bot'
    description = 'Watch a trained brain move around.'
    announce_player_deaths = False

    @classmethod
    def get_supported_maps(cls, sessiontype: type[bs.Session]) -> list[str]:
        if _RUN is not None:
            return [_RUN.game_map]
        return super().get_supported_maps(sessiontype)

    def __init__(self, settings: dict):
        super().__init__(settings)
        self._spaz: Spaz | None = None
        self._think_timer: Any = None
        self._calibrating = False
        self._cal_samples: list[float] = []
        self._cal_tries = 0

    def on_begin(self) -> None:
        # Skip DeathMatchGame.on_begin: it assumes at least one team/player exists.
        bs.TeamGameActivity.on_begin(self)
        assert _RUN is not None
        if not _RUN.calibrated:
            self._start_calibration()
        else:
            self._spawn()

    # -- spawning ---------------------------------------------------------------
    def _new_spaz(self) -> Spaz:
        if self._spaz is not None and self._spaz.node:
            self._spaz.handlemessage(bs.DieMessage(immediate=True))
        spaz = Spaz(color=(0.3, 0.7, 1.0), highlight=(1.0, 1.0, 1.0), character='Spaz', source_player=None,
                    start_invincible=False, can_accept_powerups=False, demo_mode=False)
        try:
            spaz.node.is_area_of_interest = True
        except Exception:  # pylint: disable=broad-except
            pass
        self._spaz = spaz
        return spaz

    def _start_calibration(self) -> None:
        """Drop the bot at a map spawn point and measure the floor height once."""
        self._calibrating = True
        self._cal_samples = []
        self._cal_tries += 1
        pos = self.map.get_ffa_start_position([])
        spaz = self._new_spaz()
        spaz.handlemessage(bs.StandMessage(tuple(pos), random.uniform(0, 360)))
        self._restart_timer()

    def _random_spawn(self) -> tuple[float, float]:
        d = _RUN.driver  # type: ignore[union-attr]
        for _ in range(200):
            x = random.uniform(d.cx - d.ext, d.cx + d.ext)
            z = random.uniform(d.cz - d.ext, d.cz + d.ext)
            if all(d.on_floor(x + dx, z + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
                return x, z
        return d.cx, d.cz

    def _spawn(self) -> None:
        assert _RUN is not None
        if _RUN.stopped:
            return
        d = _RUN.driver
        x, z = self._random_spawn()
        spaz = self._new_spaz()
        # same pose training used: roller bottom 0.1 above the floor (Stand y = floor - 0.5)
        spaz.handlemessage(bs.StandMessage((x, d.floor_y - 0.5, z), random.uniform(0, 360)))
        d.reset_episode()
        self._calibrating = False
        self._restart_timer()
        self._think()  # first observation happens right at spawn, as in training

    def _restart_timer(self) -> None:
        self._think_timer = bs.Timer(STEP_SECONDS, self._think, repeat=True)

    # -- control loop -----------------------------------------------------------
    def _think(self) -> None:
        run = _RUN
        if run is None or run.stopped:
            self._think_timer = None
            return
        spaz = self._spaz
        node = spaz.node if spaz is not None else None
        if not node or not spaz.is_alive():  # type: ignore[union-attr]
            self._end_episode('died')
            return
        try:
            if self._calibrating:
                self._calibrate(node)
                return
            reason = run.driver.think(node)
        except Exception:  # pylint: disable=broad-except
            logging.exception('brainbot: think failed')
            self._end_episode('error')
            return
        if reason:
            self._end_episode(reason)

    def _calibrate(self, node: Any) -> None:
        run = _RUN
        assert run is not None
        self._cal_samples.append(float(node.position[1]))
        if len(self._cal_samples) < 60:  # ~1.9 s: landed and standing
            return
        tail = sorted(self._cal_samples[-20:])
        run.driver.floor_y = tail[len(tail) // 2] - ROLLER_RADIUS
        run.calibrated = True
        _say(f'brainbot: floor height measured ({run.driver.floor_y:.2f}), starting')
        self._think_timer = None
        bs.timer(0.05, self._spawn_cb)

    def _spawn_cb(self) -> None:
        self._spawn()

    def _end_episode(self, reason: str) -> None:
        run = _RUN
        assert run is not None
        self._think_timer = None
        if self._calibrating:  # fell during calibration: try again
            if self._cal_tries < 5:
                bs.timer(0.5, self._start_calibration)
            else:
                _say('brainbot: could not measure floor height', error=True)
            return
        run.episodes += 1
        _say(f'brainbot: episode {run.episodes} ended ({reason}) after {run.driver.t} steps')
        if self._spaz is not None and self._spaz.node:
            self._spaz.handlemessage(bs.DieMessage(immediate=True))
        bs.timer(1.0, self._spawn_cb)

    # DeathMatch bookkeeping that assumes players: make them harmless.
    def end_game(self) -> None:  # noqa: D102
        pass

    def handlemessage(self, msg: Any) -> Any:  # noqa: D102
        if isinstance(msg, bs.PlayerDiedMessage):
            return None
        return super().handlemessage(msg)


def _say(msg: str, error: bool = False) -> None:
    (_log.error if error else _log.info)(msg)
    try:
        babase.screenmessage(msg, color=(1, 0.4, 0.4) if error else (0.6, 1, 0.6))
    except Exception:  # pylint: disable=broad-except
        pass


# ------------------------------------------------------------ session start ----
class BrainSession(bs.FreeForAllSession):
    """A normal FFA session whose playlist is the single brain game."""

    def get_fixed_playlist(self) -> list[dict[str, Any]] | None:  # newer engines
        return _playlist()

    def allows_tutorial(self) -> bool:  # newer engines
        return False


def _playlist() -> list[dict[str, Any]]:
    assert _RUN is not None
    return [{
        'type': f'{__name__}.BrainGame',
        'settings': {'map': _RUN.game_map, 'Epic Mode': False, 'Kills to Win Per Player': 10,
                     'Time Limit': 0, 'Respawn Times': 1.0},
    }]


_CFG_KEYS = ('Free-for-All Playlists', 'Free-for-All Playlist Selection', 'Free-for-All Playlist Randomize',
             'Show Tutorial')
_MISSING = object()


def _launch() -> None:
    cfg = babase.app.config
    saved = {k: cfg.get(k, _MISSING) for k in _CFG_KEYS}
    try:  # older engines have no fixed-playlist hook: inject a temporary playlist (in memory only)
        playlists = dict(cfg.get('Free-for-All Playlists', {}))
        playlists['brainbot'] = _playlist()
        cfg['Free-for-All Playlists'] = playlists
        cfg['Free-for-All Playlist Selection'] = 'brainbot'
        cfg['Free-for-All Playlist Randomize'] = False
        cfg['Show Tutorial'] = False
        bs.new_host_session(BrainSession)
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                cfg.pop(k, None)
            else:
                cfg[k] = v
    _skip_join(0)


def _skip_join(tries: int) -> None:
    """The session opens on the join screen; end it so the game starts, as if a player had readied up."""
    try:
        sess = bs.get_foreground_host_session()
        act = sess.getactivity() if sess is not None else None
    except Exception:  # pylint: disable=broad-except
        act = None
    if act is not None and isinstance(act, bs.JoinActivity):
        act.end()
        return
    if isinstance(act, BrainGame) or tries > 80:
        return
    babase.apptimer(0.25, lambda: _skip_join(tries + 1))


# ---------------------------------------------------------------- public API ----
def start(brain_path: str) -> None:
    """Load a brain file and start playing it in a new free-for-all session."""
    global _RUN  # pylint: disable=global-statement
    brain = Brain(os.path.expanduser(brain_path))
    game_map = brain.meta.get('game_map')
    if not game_map:
        raise ValueError(f"brain was trained on '{brain.meta.get('map')}', which has no in-game map "
                         '(metadata game_map is empty)')
    run = _Run()
    run.brain, run.driver, run.game_map = brain, Driver(brain), game_map
    _RUN = run
    _say(f"brainbot: brain trained on '{brain.meta['map']}' -> starting FFA on '{game_map}'")
    babase.pushcall(_launch)


def stop() -> None:
    """Stop driving the bot (the session keeps running; leave it from the menu)."""
    if _RUN is not None:
        _RUN.stopped = True


# ba_meta export babase.Plugin
class BrainBotPlugin(babase.Plugin):
    """Registers the module as a plugin; use start()/stop() from the console."""

    def on_app_running(self) -> None:
        pass
