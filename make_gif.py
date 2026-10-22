"""Render one evaluation episode of an agent to a GIF.

Example:
    python make_gif.py --agent ppo --ckpt checkpoints/ppo_s0.pt \
        --density 40 --seed 1003 --out results/figures/ppo_merge.gif
"""
import argparse
from pathlib import Path

import torch

from env.merge_env import MergeEnv
from env.render import save_gif
from eval import make_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--agent", choices=["ppo", "dqn", "gap", "idm", "random"], required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--density", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    env = MergeEnv()
    policy = make_policy(args.agent, args.ckpt, args.seed)
    obs, _ = env.reset(seed=args.seed, options={"density": args.density})
    frames, done = [env.render()], False
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        frames.append(env.render())
        done = term or trunc
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # One frame per 0.2 s decision step -> fps 5 is real time.
    save_gif(frames, args.out, fps=5)
    outcome = "success" if info["success"] else "collision" if info["collision"] else "failed"
    print(f"wrote {args.out} ({len(frames)} frames, {outcome})")


if __name__ == "__main__":
    main()
