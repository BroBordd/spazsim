# spazsim - headless spaz math + trainer

Build: `g++ -O2 -ffp-contract=off -shared -fPIC -std=c++17 -o libspazsim.so spazsim.cpp`
Train / resume: `python3 train.py --brain brain.npz` (Ctrl-C saves; rerun resumes weights, Adam state, obs-normalizer, RNG).

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
