"""Evaluate a trained agent or a baseline over a sweep of traffic densities.

Example:
    python eval.py --agent ppo --ckpt checkpoints/ppo_s0.pt \
        --densities 10,25,40,55,70 --episodes 200 --seed 1000

Episode seeds are seed, seed+1, ... (default 1000+). Training resets its
envs with seeds seed*100 + env index (< 1000 for training seeds 0-9), so
evaluation scenarios are never used for training. Every agent faces exactly
the same scenarios.

Densities above the training range (45 veh/km) measure generalisation.

Output: results/eval/<name>.csv (name = --name or the agent type) with one
row per density and the columns listed in METRICS.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import load_config
from env.merge_env import INF_TTC, MergeEnv

METRICS = ["success_rate", "collision_rate", "fail_rate", "mean_speed", "time_to_merge",
           "mean_abs_jerk", "ego_hard_brakes", "lag_hard_brakes", "hard_brake_rate",
           "min_ttc", "mean_return"]


def make_policy(agent: str, ckpt: str | None, seed: int):
    """Return a function (obs, env) -> continuous action in [-1, 1]."""
    if agent == "ppo":
        from agents.ppo import PPO
        model = PPO.load(ckpt, MergeEnv.OBS_DIM)
        return lambda obs, env: model.act(obs, deterministic=True)
    if agent == "dqn":
        from agents.dqn import DQN
        model = DQN.load(ckpt, MergeEnv.OBS_DIM)
        levels = np.linspace(-1.0, 1.0, model.n_actions)
        return lambda obs, env: np.array([levels[model.act(obs)]])
    from agents.baselines import GapAcceptanceAgent, IDMEgoAgent, RandomAgent
    baseline = {"gap": GapAcceptanceAgent(), "idm": IDMEgoAgent(),
                "random": RandomAgent(seed)}[agent]
    return baseline.act


def run_episode(env: MergeEnv, policy, seed: int, density: float, record: bool = False):
    """Run one episode; return a dict of episode metrics (and a trace if record)."""
    obs, _ = env.reset(seed=seed, options={"density": density})
    done, ret = False, 0.0
    speeds, jerks, trace = [], [], []
    ego_hb = lag_hb = 0
    min_ttc, merge_time = INF_TTC, np.nan
    hard = env.cfg["hard_brake"]
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        done = term or trunc
        ret += r
        speeds.append(info["speed"])
        jerks.append(info["jerk"])
        ego_hb += info["accel"] < hard
        lag_hb += info["lag_decel"] < hard
        min_ttc = min(min_ttc, info["ttc"])
        if info["just_merged"]:
            merge_time = info["time"]
        if record:
            trace.append({k: info[k] for k in ["time", "x", "speed", "accel", "ttc",
                                               "lag_decel", "merged"]})
    result = {
        "success": float(info["success"]), "collision": float(info["collision"]),
        "failed": float(info["failed"]), "speed": float(np.mean(speeds)),
        "merge_time": merge_time, "jerk": float(np.mean(jerks)),
        "ego_hb": ego_hb, "lag_hb": lag_hb, "any_lag_hb": float(lag_hb > 0),
        "min_ttc": min(min_ttc, 10.0), "return": ret,
    }
    return (result, trace) if record else result


def evaluate(policy, env: MergeEnv, density: float, episodes: int, seed: int) -> dict:
    """Aggregate metrics over `episodes` episodes at a fixed density."""
    eps = [run_episode(env, policy, seed + i, density) for i in range(episodes)]
    get = lambda k: np.array([e[k] for e in eps], dtype=float)
    merged = ~np.isnan(get("merge_time"))
    return {
        "success_rate": get("success").mean(),
        "collision_rate": get("collision").mean(),
        "fail_rate": get("failed").mean(),
        "mean_speed": get("speed").mean(),
        "time_to_merge": get("merge_time")[merged].mean() if merged.any() else np.nan,
        "mean_abs_jerk": get("jerk").mean(),
        # Hard-brake events = decision steps with accel < -4 m/s^2, per episode.
        "ego_hard_brakes": get("ego_hb").mean(),
        "lag_hard_brakes": get("lag_hb").mean(),
        # Fraction of episodes in which the ego forced the lag car to brake hard.
        "hard_brake_rate": get("any_lag_hb").mean(),
        # Minimum TTC after merging, averaged over merged episodes (capped 10 s).
        "min_ttc": get("min_ttc")[merged].mean() if merged.any() else np.nan,
        "mean_return": get("return").mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", choices=["ppo", "dqn", "gap", "idm", "random"], required=True)
    parser.add_argument("--ckpt", default=None, help="checkpoint for ppo / dqn")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--densities", default="10,25,40,55,70")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--name", default=None, help="output name (default: agent)")
    parser.add_argument("--out_dir", default="results/eval")
    args = parser.parse_args()
    if args.agent in ("ppo", "dqn") and not args.ckpt:
        parser.error("--ckpt is required for ppo / dqn")

    torch.set_num_threads(1)
    env = MergeEnv(load_config(args.config)["env"])
    policy = make_policy(args.agent, args.ckpt, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name or args.agent}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["density"] + METRICS)
        writer.writeheader()
        for d in [float(x) for x in args.densities.split(",")]:
            row = evaluate(policy, env, d, args.episodes, args.seed)
            writer.writerow({"density": d, **{k: round(float(v), 4) for k, v in row.items()}})
            print(f"density={d:5.1f}  " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()),
                  flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
