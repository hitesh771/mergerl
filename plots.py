"""Make all report figures from the results directory.

Expected layout (produced by scripts/run_all.sh):
    results/runs/<algo>_s<seed>/log.csv          training logs (ppo, dqn)
    results/eval/<name>.csv                       eval.py output, where name is
        ppo_s<seed>, dqn_s<seed>, gap, idm, random, abl_no_<term>_s<seed>

Figures written to results/figures/:
    learning_curves.pdf    success rate and return vs steps, mean +- std over seeds
    density_sweep.pdf      success / collision / hard-brake rate vs density, all agents
    ablation.pdf           reward ablations at the heaviest training density
    generalisation.pdf     success and hard-brake rate, seen vs unseen densities
    trace.pdf              speed / accel / TTC of one episode, PPO vs gap baseline
Missing inputs are skipped with a message, so partial results can be plotted.
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Two-column paper style: 3.4 in wide, 8 pt fonts.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 300,
})
WIDTH = 3.4
COLORS = {"ppo": "#1f77b4", "dqn": "#ff7f0e", "gap": "#2ca02c", "idm": "#9467bd",
          "random": "#7f7f7f"}
LABELS = {"ppo": "PPO", "dqn": "DQN", "gap": "Gap acceptance", "idm": "IDM ego",
          "random": "Random"}
TRAIN_MAX_DENSITY = 45.0


def load_eval_group(eval_dir: Path, prefix: str) -> pd.DataFrame | None:
    """Concatenate eval CSVs <prefix>.csv or <prefix>_s<k>.csv (one per seed)."""
    files = [f for f in glob.glob(str(eval_dir / f"{prefix}*.csv"))
             if re.fullmatch(rf"{re.escape(prefix)}(_s\d+)?\.csv", Path(f).name)]
    if not files:
        return None
    return pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(sorted(files))])


def mean_std(df: pd.DataFrame, col: str):
    g = df.groupby("density")[col]
    return g.mean(), g.std().fillna(0.0)


def plot_learning_curves(runs_dir: Path, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 2, 2.0))
    found = False
    for algo in ["ppo", "dqn"]:
        logs = [pd.read_csv(f) for f in sorted(glob.glob(str(runs_dir / f"{algo}_s*/log.csv")))]
        logs = [l for l in logs if len(l)]
        if not logs:
            continue
        found = True
        # Interpolate every seed onto a common step grid before averaging.
        grid = np.linspace(0, min(l["step"].max() for l in logs), 100)
        for ax, col, smooth in [(axes[0], "success_rate", 5), (axes[1], "mean_return", 5)]:
            ys = np.stack([np.interp(grid, l["step"],
                                     l[col].rolling(smooth, min_periods=1).mean()) for l in logs])
            m, s = ys.mean(0), ys.std(0)
            ax.plot(grid, m, color=COLORS[algo], label=f"{LABELS[algo]} ({len(logs)} seeds)")
            ax.fill_between(grid, m - s, m + s, color=COLORS[algo], alpha=0.2, lw=0)
    if not found:
        print("no training logs, skipping learning curves")
        plt.close(fig)
        return
    axes[0].set_ylabel("training success rate")
    axes[1].set_ylabel("mean episode return")
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="lower right")
    fig.savefig(fig_dir / "learning_curves.pdf")
    plt.close(fig)


def plot_density_sweep(evals: dict, fig_dir: Path) -> None:
    panels = [("success_rate", "success rate"), ("collision_rate", "collision rate"),
              ("hard_brake_rate", "lag hard-brake rate")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH * 2, 1.9))
    for name, df in evals.items():
        for ax, (col, label) in zip(axes, panels):
            m, s = mean_std(df, col)
            ax.errorbar(m.index, m.values, yerr=s.values, color=COLORS[name], marker="o",
                        ms=3, capsize=2, label=LABELS[name])
            ax.set_ylabel(label)
    for ax in axes:
        ax.set_xlabel("density [veh/km]")
        ax.axvspan(TRAIN_MAX_DENSITY, ax.get_xlim()[1], color="0.9", zorder=0)
        ax.set_ylim(-0.03, 1.03)
    # Legend above the panels so it never hides data; grey band = unseen densities.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(fig_dir / "density_sweep.pdf")
    plt.close(fig)


def plot_generalisation(evals: dict, fig_dir: Path) -> None:
    """Bars: mean over training densities vs mean over unseen densities."""
    names = [n for n in ["ppo", "dqn", "gap", "idm"] if n in evals]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.7))
    x = np.arange(len(names))
    for ax, col, label in [(axes[0], "success_rate", "success rate"),
                           (axes[1], "hard_brake_rate", "lag hard-brake rate")]:
        for k, (subset, hatch) in enumerate([("seen", ""), ("unseen", "///")]):
            vals = []
            for n in names:
                df = evals[n]
                mask = df["density"] <= TRAIN_MAX_DENSITY
                vals.append(df[mask if subset == "seen" else ~mask][col].mean())
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=[COLORS[n] for n in names],
                   hatch=hatch, edgecolor="k", lw=0.4, label=f"{subset} densities")
        ax.set_xticks(x, [LABELS[n].split()[0] for n in names])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(fig_dir / "generalisation.pdf")
    plt.close(fig)


def plot_ablation(eval_dir: Path, fig_dir: Path) -> None:
    variants = [("ppo", "full reward")] + [(f"abl_no_{t}", f"no {t}")
                                           for t in ["jerk", "ttc", "speed", "brake"]]
    rows = []
    for prefix, label in variants:
        df = load_eval_group(eval_dir, prefix)
        if df is None:
            continue
        df = df[df["density"] <= TRAIN_MAX_DENSITY]
        per_seed = df.groupby("seed")[["success_rate", "collision_rate", "hard_brake_rate",
                                       "mean_abs_jerk"]].mean()
        rows.append((label, per_seed.mean(), per_seed.std().fillna(0.0)))
    if len(rows) < 2:
        print("no ablation results, skipping ablation plot")
        return
    metrics = [("success_rate", "success"), ("collision_rate", "collision"),
               ("hard_brake_rate", "lag hard brake"), ("mean_abs_jerk", "|jerk| [m/s³]")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH * 2, 1.8))
    x = np.arange(len(rows))
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(x, [r[1][col] for r in rows], yerr=[r[2][col] for r in rows], capsize=2,
               color=["#1f77b4"] + ["#aec7e8"] * (len(rows) - 1), edgecolor="k", lw=0.4)
        ax.set_xticks(x, [r[0] for r in rows], rotation=45, ha="right")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation.pdf")
    plt.close(fig)


def plot_trace(ckpt: str, fig_dir: Path, density: float, seed: int) -> None:
    """Speed, acceleration and lag-car deceleration of one episode, PPO vs gap."""
    from eval import make_policy, run_episode
    from env.merge_env import MergeEnv
    if not Path(ckpt).exists():
        print(f"{ckpt} missing, skipping trace plot")
        return
    env = MergeEnv()
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.4), sharex=True)
    for name, policy in [("ppo", make_policy("ppo", ckpt, seed)),
                         ("gap", make_policy("gap", None, seed))]:
        _, trace = run_episode(env, policy, seed, density, record=True)
        tr = pd.DataFrame(trace)
        axes[0].plot(tr["time"], tr["speed"], color=COLORS[name], label=LABELS[name])
        axes[1].plot(tr["time"], tr["accel"], color=COLORS[name])
        axes[2].plot(tr["time"], tr["lag_decel"], color=COLORS[name])
        merged = tr[tr["merged"]]
        if len(merged):
            for ax in axes:
                ax.axvline(merged["time"].iloc[0], color=COLORS[name], ls=":", lw=0.8)
    axes[0].set_ylabel("ego speed [m/s]")
    axes[1].set_ylabel("ego accel [m/s²]")
    axes[2].set_ylabel("lag-car min\naccel [m/s²]")
    axes[2].axhline(-4.0, color="r", ls="--", lw=0.8)
    axes[2].set_xlabel("time [s]")
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"density {density:.0f} veh/km, scenario seed {seed} (dotted: merge)")
    fig.savefig(fig_dir / "trace.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--trace_ckpt", default="checkpoints/ppo_s0.pt")
    parser.add_argument("--trace_density", type=float, default=40.0)
    parser.add_argument("--trace_seed", type=int, default=1003)
    args = parser.parse_args()
    res = Path(args.results)
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(res / "runs", fig_dir)
    evals = {n: df for n in ["ppo", "dqn", "gap", "idm", "random"]
             if (df := load_eval_group(res / "eval", n)) is not None}
    if evals:
        plot_density_sweep(evals, fig_dir)
        plot_generalisation(evals, fig_dir)
    else:
        print("no eval results, skipping density sweep and generalisation")
    plot_ablation(res / "eval", fig_dir)
    plot_trace(args.trace_ckpt, fig_dir, args.trace_density, args.trace_seed)
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
