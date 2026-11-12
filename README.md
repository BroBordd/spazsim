# spazsim - headless spaz math + trainer

## Layout
    cpp/      spazsim.cpp (physics math core), precalc_rands.h + gen_tables.py (game's baked random tables)
    python/   spazsim.py (ctypes wrapper), spazenv.py (maps, observations, rewards), train.py (PPO, numpy)
    plugin/   brainbot.py  (BombSquad api 9 plugin; copy to the mods folder)
    brains/   the_pad.npz  (trained brain + metadata: map, game_map, geometry)
    tests/    test_plugin_equiv.py + stub game modules (offline checks, no game needed)

`make` builds the core, `make test` runs the offline checks, `make train` trains/resumes `brains/the_pad.npz`
(Ctrl-C saves; rerun resumes weights, Adam state, obs-normalizer, RNG).

## What is a math port (spazsim.cpp, float32, operand order kept, no FMA)
- ODE quickstep: PGS/SOR (w=1.3, 10 iters), shuffled row order from dRand seed 5432 (saved/restored per solve),
  island DFS body/joint ordering, finite-step integrator, 125 Hz (8 ms), gravity -20, global erp/cfm.
- Joints: soft fixed (JointFixedEF), amotor (roller drive + brakes), contact rows with friction cone.
- SpazNode logic: balance, stand joint, roller suspension/jump, motor + gears + run gas, brakes,
  dizzy->knockout, neck pose, collide callback, bg-limbs tuning, SPAZ_RIG sizes/masses, 51 ms spawn no-collide.

## Not identical (be aware)
- Map collision meshes (.cob) are not in the repo; terrain is axis-aligned boxes. Sphere-vs-box is ODE's algorithm,
  pelvis uses ODE box-plane contact selection on box tops; the game's sphere-vs-trimesh may differ at edges.
- Python-layer damage/impact knockouts, head-torso self collision, limbs and idle shoulder cosmetics are omitted.
- No bit-for-bit comparison against the C++ engine was run (engine not built here).

## BombSquad plugin (brainbot.py, api 9)
Copy `plugin/brainbot.py` into the game's mods folder, then in the in-game python console:

    import brainbot; brainbot.start("path/to/brain.npz")

It reads the brain's metadata (`meta.game_map`, the map it was trained on), starts a fresh Free-for-All session on
that map (join screen skipped), measures the floor height once, then spawns a bot spaz and drives it every 32 ms
(4 game steps, as in training); on fall / knock-down / 19 s timeout it respawns. `brainbot.stop()` stops driving.
Pure python (no numpy). The brain sees only node attributes (position, torso_position, position_center,
position_forward, velocity, knockout), so the headless-trained file drives the real game.

Metadata: the trainer writes `meta` (map, game_map, boxes, repeat, dt, obs/act sizes) into brain.npz and refuses to
continue a brain on a different map. Maps live in `spazenv.MAPS`; `the_pad` is an approximate footprint of The Pad
(real collision meshes are not in the repo), `holes` is training-only. Tested offline (`test_plugin_equiv.py`):
plugin obs/actions == trainer to ~1e-6, 20/20 closed-loop episodes survive. NOT tested inside a running game here.
