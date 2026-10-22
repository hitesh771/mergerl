#!/usr/bin/env bash
# Full reproduction pipeline: train, evaluate, plot, render.
#
# Usage:
#   scripts/run_all.sh                 # full run (long)
#   STEPS_PPO=5000 STEPS_DQN=5000 EPISODES=5 scripts/run_all.sh   # quick check
#
# Training runs of one stage are started in parallel (one CPU core each),
# capped at JOBS processes at a time.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
STEPS_PPO="${STEPS_PPO:-3000000}"
STEPS_DQN="${STEPS_DQN:-1000000}"
STEPS_ABL="${STEPS_ABL:-$STEPS_PPO}"
SEEDS="${SEEDS:-0 1 2}"
ABL_SEEDS="${ABL_SEEDS:-0 1 2}"
EPISODES="${EPISODES:-200}"
DENSITIES="${DENSITIES:-10,25,40,55,70}"
JOBS="${JOBS:-10}"
LOGS=results/logs
mkdir -p "$LOGS" results/runs results/eval results/figures checkpoints

# Run "$@" in the background, waiting first if JOBS processes are running.
run_bg() {
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n
    done
    "$@" &
}

wait_all() {
    # `wait` on each pid so that a failing job makes the script fail.
    local pid
    for pid in $(jobs -rp); do
        wait "$pid"
    done
    wait
}

echo "== tests"
"$PY" -m pytest -q tests

echo "== training (PPO + DQN seeds: $SEEDS; ablation seeds: $ABL_SEEDS)"
for s in $SEEDS; do
    run_bg "$PY" train.py --algo ppo --seed "$s" --steps "$STEPS_PPO" \
        --out "results/runs/ppo_s$s" > "$LOGS/train_ppo_s$s.log" 2>&1
    run_bg "$PY" train.py --algo dqn --seed "$s" --steps "$STEPS_DQN" \
        --out "results/runs/dqn_s$s" > "$LOGS/train_dqn_s$s.log" 2>&1
done
for term in jerk ttc speed brake; do
    for s in $ABL_SEEDS; do
        run_bg "$PY" train.py --algo ppo --seed "$s" --steps "$STEPS_ABL" \
            --config "configs/ablation_no_$term.yaml" \
            --out "results/runs/abl_no_${term}_s$s" > "$LOGS/train_abl_no_${term}_s$s.log" 2>&1
    done
done
wait_all

echo "== evaluation (densities $DENSITIES, $EPISODES episodes each)"
EVAL=(--densities "$DENSITIES" --episodes "$EPISODES" --seed 1000)
for s in $SEEDS; do
    run_bg "$PY" eval.py --agent ppo --ckpt "checkpoints/ppo_s$s.pt" --name "ppo_s$s" "${EVAL[@]}" \
        > "$LOGS/eval_ppo_s$s.log" 2>&1
    run_bg "$PY" eval.py --agent dqn --ckpt "checkpoints/dqn_s$s.pt" --name "dqn_s$s" "${EVAL[@]}" \
        > "$LOGS/eval_dqn_s$s.log" 2>&1
done
for term in jerk ttc speed brake; do
    for s in $ABL_SEEDS; do
        run_bg "$PY" eval.py --agent ppo --ckpt "checkpoints/abl_no_${term}_s$s.pt" \
            --config "configs/ablation_no_$term.yaml" --name "abl_no_${term}_s$s" "${EVAL[@]}" \
            > "$LOGS/eval_abl_no_${term}_s$s.log" 2>&1
    done
done
for agent in gap idm random; do
    run_bg "$PY" eval.py --agent "$agent" "${EVAL[@]}" > "$LOGS/eval_$agent.log" 2>&1
done
wait_all

echo "== figures"
"$PY" plots.py
"$PY" make_gif.py --agent ppo --ckpt checkpoints/ppo_s0.pt --out results/figures/ppo_merge.gif
"$PY" make_gif.py --agent gap --out results/figures/gap_merge.gif
echo "done; see results/figures and results/eval"
