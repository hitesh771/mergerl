# MergeRL: learning safe highway on-ramp merging

Reinforcement-learning mini-project (IIT Jodhpur, AI course).
Author: Hitesh Saini (B24CM1029).

An ego car starts on an on-ramp and has to merge into a single-lane highway
with IDM-driven traffic before its acceleration lane ends. The agent only
controls its longitudinal acceleration; the lane change itself happens
automatically when the gaps allow it. The real difficulty is timing: the
agent has to pick a speed profile that lines it up with a usable gap, and
it has to merge so that the car behind it (the "lag" car) does not need to
brake hard. Everything (simulator, PPO, DQN, baselines) is written from
scratch in numpy / PyTorch.

## MDP

| | |
|---|---|
| Simulation | 1-D longitudinal, dt = 0.1 s, agent acts every 2 sub-steps (5 Hz) |
| Road | highway 0–700 m; acceleration lane with merge zone 200–320 m and a hard end at 320 m |
| Traffic | IDM cars, per-car driver parameters between "cautious" and "aggressive", density ρ ~ U(10, 45) veh/km per episode, steady inflow upstream |
| Action | a ∈ [-1, 1] → acceleration in [-3, +3] m/s² (DQN: 7 levels -3, -2, …, +3) |
| Merge | automatic as soon as the ego is in the zone and both highway gaps are ≥ 2 m. This is deliberately permissive, so bad cut-ins are possible. Highway IDM braking is capped at 8 m/s², so a lag car can crash into the ego |
| Termination | collision, ramp end or 40 s timeout without merging (failure), or 8 s after the merge without a crash (success) |

### State (13 dims, all roughly in [-1, 1])

| idx | feature | normalisation |
|---|---|---|
| 0 | ego speed | v / 30 |
| 1 | ego position relative to merge zone | (x - 200) / 120 |
| 2 | distance to ramp end (0 once merged) | (320 - x) / 100 |
| 3 | merged flag | {0, 1} |
| 4, 5 | highway lead gap, relative speed | gap / 50, Δv / 10 |
| 6, 7 | highway lag gap, relative speed | gap / 50, Δv / 10 |
| 8 | TTC of the lag car to the ego | clip(0, 10 s) / 5 − 1 |
| 9 | lag-car speed | v / 30 |
| 10 | traffic density | ρ / 50 |
| 11 | previous ego acceleration | a / 3 |
| 12 | lag-car time headway | clip(gap / v_lag, 0, 4 s) / 2 − 1 |

A negative gap means a highway car is alongside the ego.

### Reward (weights in `configs/default.yaml`)

| term | weight | why |
|---|---|---|
| speed v / 35 | +0.05 / step | keep moving, do not block the ramp |
| \|Δa\| / 6 (jerk) | −0.02 / step | comfort |
| merged and TTC (lead or lag) < 2 s | −0.3 / step | unsafe proximity after merging |
| lag car decelerates harder than 4 m/s² | −1.0 / step | forcing others to brake hard is the main safety cost |
| on the ramp with v < 2 m/s | −0.1 / step | stops "wait forever" reward hacking |
| time | −0.02 / step | merge sooner |
| merge | +5 | |
| survive 8 s after merging | +5 | the agent pays for post-merge consequences |
| collision | −20 | |
| ramp end / timeout unmerged | −10 | smaller than a collision, so failing to merge is better than crashing |

Training uses a density curriculum: 10–20 veh/km for the first 20 % of
steps, 10–35 up to 50 %, then 10–45. Evaluation also covers 55 and 70
veh/km, which the agent never sees in training (generalisation).

## Methods

* `agents/ppo.py`: PPO with GAE(λ = 0.95), clipped surrogate (ε = 0.2),
  clipped value loss, advantage normalisation, running observation
  normalisation (stored in the checkpoint), gradient clipping, linear LR
  decay and 16 environments run in lock-step.
* `agents/dqn.py`: Double DQN with replay buffer, hard target updates,
  Huber loss and linear ε-greedy decay on the 7 discrete accelerations.
* `agents/baselines.py` (comparisons only):
  * `GapAcceptanceAgent` plans a constant acceleration that brings the ego
    to the zone entry at the same time as a gap ≥ 10 m. Among the feasible
    gaps it picks the one with the smallest speed mismatch. It has full
    simulator state.
  * `IDMEgoAgent` drives IDM towards a virtual stop at the ramp end and
    relies on the auto-merge.
  * `RandomAgent` picks uniform random accelerations.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Commands

```bash
# tests
.venv/bin/python -m pytest -q tests

# train (logs to results/runs/<name>/log.csv, model to checkpoints/<name>.pt)
.venv/bin/python train.py --algo ppo --config configs/default.yaml --seed 0 --steps 3000000 --out results/runs/ppo_s0
.venv/bin/python train.py --algo dqn --seed 0 --steps 1000000 --out results/runs/dqn_s0
.venv/bin/python train.py --algo ppo --config configs/ablation_no_ttc.yaml --seed 0 --out results/runs/abl_no_ttc_s0

# evaluate over densities (writes results/eval/<name>.csv)
.venv/bin/python eval.py --agent ppo --ckpt checkpoints/ppo_s0.pt --name ppo_s0 --densities 10,25,40,55,70 --episodes 200 --seed 1000
.venv/bin/python eval.py --agent gap        # also: idm, random, dqn

# figures and GIFs
.venv/bin/python plots.py
.venv/bin/python make_gif.py --agent ppo --ckpt checkpoints/ppo_s0.pt --out results/figures/ppo_merge.gif

# everything (3 seeds PPO + DQN, 4 ablations x 3 seeds, eval, plots, GIFs)
scripts/run_all.sh
# quick pipeline check
STEPS_PPO=5000 STEPS_DQN=6000 EPISODES=5 scripts/run_all.sh
```

Seeds: training seed k resets its environments with seeds 100k + i (i =
env index). Evaluation uses scenario seeds 1000, 1001, … and the same
scenarios for every agent, so training and evaluation never overlap.

## Results

Fill in from `results/eval/*.csv` after `scripts/run_all.sh`. Mean over 3
seeds, 200 episodes per density.

| agent | success ↑ | collision ↓ | lag hard-brake rate ↓ | mean \|jerk\| ↓ | time to merge |
|---|---|---|---|---|---|
| PPO | | | | | |
| DQN | | | | | |
| Gap acceptance | | | | | |
| IDM ego | | | | | |
| Random | | | | | |

Figures: `results/figures/learning_curves.pdf`, `density_sweep.pdf`,
`ablation.pdf`, `generalisation.pdf`, `trace.pdf`, plus `*_merge.gif`.

## Repository layout

```
env/idm.py           IDM car following (+ constant-acceleration cut-in correction)
env/traffic.py       highway lane, spawning / recycling, ego as leader after merge
env/merge_env.py     gymnasium MergeEnv, reward, metrics, discrete-action wrapper
env/render.py        matplotlib top-down frames and GIF export
agents/networks.py   MLP actor (Gaussian), critic, Q-network
agents/ppo.py        PPO, GAE, running obs normalisation, vector env loop
agents/dqn.py        Double DQN and replay buffer
agents/baselines.py  gap acceptance, IDM ego, random
config.py            YAML loader with `base:` inheritance, curriculum schedule
configs/             default config and four reward ablations
train.py eval.py plots.py make_gif.py
scripts/run_all.sh   full reproduction pipeline
tests/               pytest unit tests (IDM, env, PPO/GAE, DQN)
```
