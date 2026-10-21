"""Train PPO or DQN on the merge environment.

Example:
    python train.py --algo ppo --config configs/default.yaml --seed 0 \
        --steps 2000000 --out results/runs/ppo_s0

Writes <out>/log.csv with one row per log interval:
    step, mean_return, success_rate, collision_rate, fail_rate, density_hi
(statistics over the episodes finished since the previous row) and saves the
final model to checkpoints/<name>.pt, where <name> is the last part of --out.
Env seeds used in training are seed*100 + env index (< 1000 for seeds 0-9);
evaluation uses env seeds >= 1000 (see eval.py).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from config import density_for_progress, load_config
from env.merge_env import MergeEnv


def episode_stats(episodes: list) -> dict:
    """Mean return and outcome rates over a list of finished-episode infos."""
    if not episodes:
        return None
    return {
        "mean_return": float(np.mean([e["return"] for e in episodes])),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "collision_rate": float(np.mean([e["collision"] for e in episodes])),
        "fail_rate": float(np.mean([e["failed"] for e in episodes])),
    }


class CsvLogger:
    FIELDS = ["step", "mean_return", "success_rate", "collision_rate", "fail_rate",
              "density_hi", "steps_per_s"]

    def __init__(self, path: Path):
        self.file = path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def log(self, row: dict) -> None:
        self.writer.writerow({k: row.get(k, "") for k in self.FIELDS})
        self.file.flush()
        print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()), flush=True)


def set_density(envs: list, density_range: list) -> None:
    for env in envs:
        env.unwrapped.cfg["density_range"] = list(density_range)


def train_ppo(cfg: dict, seed: int, total_steps: int, logger: CsvLogger, ckpt: Path) -> None:
    from agents.ppo import PPO, VecEnv
    pc = cfg["ppo"]
    venv = VecEnv(lambda: MergeEnv(cfg["env"]), pc["n_envs"], seed=seed * 100)
    agent = PPO(MergeEnv.OBS_DIM, 1, pc)
    batch_steps = pc["n_envs"] * pc["n_steps"]
    n_iters = max(total_steps // batch_steps, 1)
    step, t0 = 0, time.time()
    for it in range(n_iters):
        progress = it / n_iters
        dens = density_for_progress(cfg, progress)
        set_density(venv.envs, dens)
        batch, episodes = agent.collect(venv, pc["n_steps"])
        lr = pc["lr"] * (1.0 - progress)          # linear decay to 0
        agent.update(batch, lr)
        step += batch_steps
        stats = episode_stats(episodes)
        if stats:
            logger.log({"step": step, **stats, "density_hi": dens[1],
                        "steps_per_s": round(step / (time.time() - t0))})
    agent.save(str(ckpt), {"seed": seed, "steps": step})


def train_dqn(cfg: dict, seed: int, total_steps: int, logger: CsvLogger, ckpt: Path) -> None:
    from agents.dqn import DQN
    from env.merge_env import DiscreteActionWrapper
    dc = cfg["dqn"]
    env = DiscreteActionWrapper(MergeEnv(cfg["env"]), dc["n_actions"])
    agent = DQN(MergeEnv.OBS_DIM, dc["n_actions"], dc, seed=seed)
    obs, _ = env.reset(seed=seed * 100)
    ep_return, episodes, t0 = 0.0, [], time.time()
    log_every = max(total_steps // 100, 1000)
    for step in range(1, total_steps + 1):
        progress = step / total_steps
        dens = density_for_progress(cfg, progress)
        set_density([env], dens)
        action = agent.act(obs, agent.epsilon(progress))
        next_obs, r, term, trunc, info = env.step(action)
        # Only true terminations stop bootstrapping (time-outs are also
        # terminal in this env, so trunc is always False here).
        agent.buffer.add(obs, action, r, next_obs, float(term))
        ep_return += r
        obs = next_obs
        if term or trunc:
            episodes.append({**info, "return": ep_return})
            ep_return = 0.0
            obs, _ = env.reset()
        if step >= dc["learning_starts"] and step % dc["train_every"] == 0:
            agent.train_step()
        if step % dc["target_update"] == 0:
            agent.update_target()
        if step % log_every == 0:
            stats = episode_stats(episodes)
            if stats:
                logger.log({"step": step, **stats, "density_hi": dens[1],
                            "steps_per_s": round(step / (time.time() - t0))})
            episodes = []
    agent.save(str(ckpt), {"seed": seed, "steps": total_steps})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--out", required=True, help="run directory, e.g. results/runs/ppo_s0")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"{out.name}.pt"
    logger = CsvLogger(out / "log.csv")
    print(f"training {args.algo} seed={args.seed} steps={args.steps} config={args.config}")
    if args.algo == "ppo":
        train_ppo(cfg, args.seed, args.steps, logger, ckpt)
    else:
        train_dqn(cfg, args.seed, args.steps, logger, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
