"""Make all report figures from the results directory.

Expected layout (produced by scripts/run_all.sh):
    results/runs/<algo>_s<seed>/log.csv          training logs (ppo, dqn)
    results/eval/<name>.csv                       eval.py output, where name is
        ppo_s<seed>, dqn_s<seed>, gap, idm, random, abl_no_<term>_s<seed>

Figures written to results/figures/ (each as .pdf and .png):
    learning_curves     training success rate and return vs steps, mean +- std over seeds
    density_sweep       success / collision / lag hard-brake rate / |jerk| vs density
    ablation            reward ablations, averaged over the training densities
    generalisation      success and lag hard-brake rate, seen vs unseen densities
    trace               speed / accel / lag-car accel of one episode, PPO vs gap baseline
    merge_snapshot      a few frames of one PPO episode (environment illustration)
Missing inputs are skipped with a message, so partial results can be plotted.

Metric definitions used in the axis labels:
    success rate         fraction of episodes that merge and survive 8 s afterwards
    collision rate       fraction of episodes that end in a crash
    fail rate            fraction of episodes that reach the ramp end / time out unmerged
    lag hard-brake rate  fraction of episodes in which the highway car behind the ego
                         (the "lag" car) brakes harder than 4 m/s^2 at least once
    mean |jerk|          mean absolute change of ego acceleration per second [m/s^3]
Error bars and bands are +-1 standard deviation over training seeds (3 seeds for
PPO, DQN and the ablations). The rule-based baselines are deterministic given the
scenario seeds, so they have a single evaluation run and no error bar.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Two-column IEEE style: 3.4 in single column, 7.0 in double column.
# Every font is 8 pt because figures are included at their native size.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.handlelength": 2.5, "pdf.fonttype": 42,
})
WIDTH = 3.4          # single column [in]
WIDTH2 = 7.0         # double column [in]
PNG_DPI = 200

# Okabe-Ito colour-blind safe palette; each method also has its own marker and
# line style so the plots stay readable in greyscale.
COLORS = {"ppo": "#0072B2", "dqn": "#E69F00", "gap": "#009E73", "idm": "#CC79A7",
          "random": "black"}
MARKERS = {"ppo": "o", "dqn": "s", "gap": "^", "idm": "D", "random": "x"}
LINESTYLES = {"ppo": "-", "dqn": (0, (5, 1, 1, 1, 1, 1)), "gap": "-.", "idm": ":",
              "random": "--"}
LABELS = {"ppo": "PPO", "dqn": "DQN", "gap": "Gap acceptance", "idm": "IDM ego",
          "random": "Random"}

TRAIN_MAX_DENSITY = 55.0          # highest density seen in training [veh/km]
CURRICULUM_STAGE_ENDS = [0.2, 0.5]  # fractions of training (configs/default.yaml)
SMOOTH_STEPS = 50_000             # learning-curve rolling-mean window [env steps]
HARD_BRAKE = -4.0                 # hard-brake threshold [m/s^2] (env DEFAULT_CONFIG)
A_MAX = 3.0                       # ego action limit [m/s^2] (env A_MAX)


def save(fig, fig_dir: Path, name: str) -> None:
    """Save a figure as vector PDF (for LaTeX) and PNG (for previews)."""
    fig.savefig(fig_dir / f"{name}.pdf")
    fig.savefig(fig_dir / f"{name}.png", dpi=PNG_DPI)
    plt.close(fig)


def load_eval_group(eval_dir: Path, prefix: str) -> pd.DataFrame | None:
    """Concatenate eval CSVs <prefix>.csv or <prefix>_s<k>.csv (one per seed)."""
    files = [f for f in glob.glob(str(eval_dir / f"{prefix}*.csv"))
             if re.fullmatch(rf"{re.escape(prefix)}(_s\d+)?\.csv", Path(f).name)]
    if not files:
        return None
    return pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(sorted(files))])


def mean_std(df: pd.DataFrame, col: str):
    """Mean and std over seeds of `col` at every density (std 0 for one seed)."""
    g = df.groupby("density")[col]
    return g.mean(), g.std().fillna(0.0)


def clipped_yerr(mean, std, lo: float = 0.0, hi: float | None = 1.0) -> np.ndarray:
    """Asymmetric error bars mean +- std, cut so they stay inside [lo, hi]."""
    mean, std = np.asarray(mean, float), np.asarray(std, float)
    lower = np.minimum(std, mean - lo)
    upper = std if hi is None else np.minimum(std, hi - mean)
    return np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])


def method_handle(name: str) -> Line2D:
    """Legend entry showing colour, marker and line style of one method."""
    return Line2D([], [], color=COLORS[name], marker=MARKERS[name], ls=LINESTYLES[name],
                  ms=4, label=LABELS[name])


# --------------------------------------------------------------------------- #
# Learning curves
# --------------------------------------------------------------------------- #
def plot_learning_curves(runs_dir: Path, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH2, 2.0))
    handles, ppo_len = [], None
    for algo in ["ppo", "dqn"]:
        logs = [pd.read_csv(f) for f in sorted(glob.glob(str(runs_dir / f"{algo}_s*/log.csv")))]
        logs = [l for l in logs if len(l)]
        if not logs:
            continue
        # The rolling window is given in env steps; convert it to log rows
        # because PPO logs every 4096 steps and DQN every 10k steps.
        log_every = np.median(np.diff(logs[0]["step"]))
        window = max(int(round(SMOOTH_STEPS / log_every)), 1)
        # Interpolate every seed onto a common step grid before averaging.
        total = min(l["step"].max() for l in logs)
        grid = np.linspace(logs[0]["step"].iloc[0], total, 200)
        if algo == "ppo":
            ppo_len = total
        for ax, col in [(axes[0], "success_rate"), (axes[1], "mean_return")]:
            ys = np.stack([np.interp(grid, l["step"],
                                     l[col].rolling(window, min_periods=1).mean()) for l in logs])
            m, s = ys.mean(0), ys.std(0)
            lo, hi = m - s, m + s
            if col == "success_rate":
                lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
            ax.plot(grid / 1e6, m, color=COLORS[algo], ls=LINESTYLES[algo])
            ax.fill_between(grid / 1e6, lo, hi, color=COLORS[algo], alpha=0.2, lw=0)
        handles.append(Line2D([], [], color=COLORS[algo], ls=LINESTYLES[algo],
                              label=f"{LABELS[algo]} ({len(logs)} seeds)"))
    if not handles:
        print("no training logs, skipping learning curves")
        plt.close(fig)
        return

    axes[0].set_ylabel("training success rate ↑")
    axes[0].set_ylim(None, 1.02)
    axes[1].set_ylabel("mean episode return ↑")
    for ax, tag in zip(axes, "ab"):
        ax.set_xlabel("environment steps (×10⁶)")
        ax.set_title(f"({tag})", loc="left")
        # Curriculum stage boundaries of the PPO runs (DQN has a shorter run,
        # so its boundaries would be at different x and are not drawn).
        if ppo_len is not None:
            for k, frac in enumerate(CURRICULUM_STAGE_ENDS):
                x = frac * ppo_len / 1e6
                ax.axvline(x, color="0.5", ls=":", lw=0.8, zorder=0)
                ax.text(x, 0.02, f" stage {k + 2}", transform=ax.get_xaxis_transform(),
                        ha="left", va="bottom", color="0.35")
    handles.append(Line2D([], [], color="0.5", ls=":", lw=0.8,
                          label="PPO curriculum stage start"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.02, f"line = mean, band = ±1 std over seeds; rolling mean over "
                         f"{SMOOTH_STEPS // 1000}k environment steps",
             ha="center", va="top", color="0.3")
    fig.tight_layout()
    save(fig, fig_dir, "learning_curves")


# --------------------------------------------------------------------------- #
# Density sweep
# --------------------------------------------------------------------------- #
def plot_density_sweep(evals: dict, fig_dir: Path) -> None:
    # (column, y label, is the metric a rate in [0, 1]?)
    panels = [("success_rate", "success rate ↑", True),
              ("collision_rate", "collision rate ↓", True),
              ("hard_brake_rate", "lag hard-brake rate ↓", True),
              ("mean_abs_jerk", "mean |jerk| [m/s³] ↓", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH2, 1.9))
    # Small horizontal shift per method so error bars at the same density
    # do not sit on top of each other.
    offsets = np.linspace(-2.0, 2.0, len(evals))
    for off, (name, df) in zip(offsets, evals.items()):
        for ax, (col, _, is_rate) in zip(axes, panels):
            m, s = mean_std(df, col)
            yerr = clipped_yerr(m.values, s.values, hi=1.0 if is_rate else None)
            ax.errorbar(m.index + off, m.values, yerr=yerr, color=COLORS[name],
                        marker=MARKERS[name], ls=LINESTYLES[name], ms=3.5, capsize=1.5,
                        elinewidth=0.8, lw=1.0)
    densities = sorted(next(iter(evals.values()))["density"].unique())
    x_lo, x_hi = densities[0] - 5, densities[-1] + 5
    for ax, (_, label, is_rate), tag in zip(axes, panels, "abcd"):
        ax.set_xlabel("density [veh/km]")
        ax.set_ylabel(label)
        ax.set_title(f"({tag})", loc="left")
        ax.set_xlim(x_lo, x_hi)
        ax.set_xticks(densities)
        ax.axvspan(TRAIN_MAX_DENSITY + 2.5, x_hi, color="0.88", lw=0, zorder=0)
        if is_rate:
            ax.set_ylim(-0.03, 1.03)
        else:
            # Random's jerk is ~10x the others, so a log axis keeps all visible.
            ax.set_yscale("log")
    handles = [method_handle(n) for n in evals]
    handles.append(Patch(color="0.88", label=f"unseen in training (> {TRAIN_MAX_DENSITY:.0f})"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, fig_dir, "density_sweep")


# --------------------------------------------------------------------------- #
# Seen vs unseen densities
# --------------------------------------------------------------------------- #
def seed_means(df: pd.DataFrame, mask, col: str):
    """Average `col` over the densities in `mask` per seed, then mean / std over seeds."""
    per_seed = df[mask].groupby("seed")[col].mean()
    return per_seed.mean(), (per_seed.std() if len(per_seed) > 1 else 0.0)


def plot_generalisation(evals: dict, fig_dir: Path) -> None:
    """Bars: mean over training densities vs mean over unseen densities."""
    names = [n for n in ["ppo", "dqn", "gap", "idm"] if n in evals]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9))
    x = np.arange(len(names))
    bar_w = 0.38
    subsets = [("seen", ""), ("unseen", "////")]
    for ax, col, label, tag in [(axes[0], "success_rate", "success rate ↑", "a"),
                                (axes[1], "hard_brake_rate", "lag hard-brake rate ↓", "b")]:
        for k, (subset, hatch) in enumerate(subsets):
            means, stds = [], []
            for n in names:
                df = evals[n]
                seen = df["density"] <= TRAIN_MAX_DENSITY
                m, s = seed_means(df, seen if subset == "seen" else ~seen, col)
                means.append(m)
                stds.append(s)
            xs = x + (k - 0.5) * bar_w
            ax.bar(xs, means, bar_w, yerr=clipped_yerr(means, stds), capsize=1.5,
                   error_kw={"elinewidth": 0.7}, color=[COLORS[n] for n in names],
                   hatch=hatch, edgecolor="k", lw=0.4)
            # Value labels, so that bars of height ~0 are still readable.
            for xi, m, s in zip(xs, means, stds):
                ax.text(xi, min(m + s, 1.0) + 0.03, f"{m:.2f}", ha="center", va="bottom",
                        rotation=90)
        ax.set_xticks(x, [LABELS[n].split()[0] for n in names])
        ax.set_ylabel(label)
        ax.set_title(f"({tag})", loc="left")
        ax.set_ylim(0, 1.35)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor="white", edgecolor="k", lw=0.4, hatch=h,
                     label=f"{s} densities ({'≤' if s == 'seen' else '>'} "
                           f"{TRAIN_MAX_DENSITY:.0f})")
               for s, h in subsets]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, fig_dir, "generalisation")


# --------------------------------------------------------------------------- #
# Reward ablation
# --------------------------------------------------------------------------- #
def plot_ablation(eval_dir: Path, fig_dir: Path) -> None:
    variants = [("ppo", "full reward")] + [(f"abl_no_{t}", f"no {t}")
                                           for t in ["jerk", "ttc", "speed", "brake"]]
    # (column, title, value format for the bar labels)
    metrics = [("collision_rate", "collision rate ↓", "{:.3f}"),
               ("fail_rate", "fail rate ↓", "{:.3f}"),
               ("hard_brake_rate", "lag hard-brake rate ↓", "{:.2f}"),
               ("mean_abs_jerk", "mean |jerk| [m/s³] ↓", "{:.2f}")]
    rows, n_seeds = [], 0
    for prefix, label in variants:
        df = load_eval_group(eval_dir, prefix)
        if df is None:
            continue
        # Average over the training densities only (10-55 veh/km).
        df = df[df["density"] <= TRAIN_MAX_DENSITY]
        per_seed = df.groupby("seed")[[m[0] for m in metrics]].mean()
        n_seeds = max(n_seeds, len(per_seed))
        rows.append((label, per_seed.mean(), per_seed.std().fillna(0.0)))
    if len(rows) < 2:
        print("no ablation results, skipping ablation plot")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDTH2, 1.9))
    x = np.arange(len(rows))
    # Dark bar = full reward (reference), light hatched bars = one term removed.
    colors = ["#08519c"] + ["#9ecae1"] * (len(rows) - 1)
    hatches = [""] + ["//"] * (len(rows) - 1)
    for ax, (col, title, fmt), tag in zip(axes, metrics, "abcd"):
        means = np.array([r[1][col] for r in rows])
        stds = np.array([r[2][col] for r in rows])
        is_rate = col.endswith("_rate")
        ax.bar(x, means, yerr=clipped_yerr(means, stds, hi=1.0 if is_rate else None),
               capsize=1.5, error_kw={"elinewidth": 0.7}, color=colors, hatch=hatches,
               edgecolor="k", lw=0.4)
        top = (means + stds).max()
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.02 * top, fmt.format(m), ha="center", va="bottom",
                    rotation=90)
        ax.set_ylim(0, 1.8 * top if top > 0 else 0.01)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xticks(x, [r[0] for r in rows], rotation=45, ha="right",
                      rotation_mode="anchor")
        ax.set_title(f"({tag}) {title}", loc="left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    fig.text(0.5, -0.02, f"PPO, mean ± 1 std over {n_seeds} seeds, averaged over the training "
                         f"densities 10–{TRAIN_MAX_DENSITY:.0f} veh/km",
             ha="center", va="top", color="0.3")
    save(fig, fig_dir, "ablation")


# --------------------------------------------------------------------------- #
# Single-episode trace
# --------------------------------------------------------------------------- #
def plot_trace(ckpt: str, fig_dir: Path, density: float, seed: int) -> None:
    """Speed, acceleration and lag-car acceleration of one episode, PPO vs gap."""
    from eval import make_policy, run_episode
    from env.merge_env import MergeEnv
    if not Path(ckpt).exists():
        print(f"{ckpt} missing, skipping trace plot")
        return
    env = MergeEnv()
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.4), sharex=True)
    handles = []
    for name, policy in [("ppo", make_policy("ppo", ckpt, seed)),
                         ("gap", make_policy("gap", None, seed))]:
        _, trace = run_episode(env, policy, seed, density, record=True)
        tr = pd.DataFrame(trace)
        style = {"color": COLORS[name], "ls": LINESTYLES[name]}
        axes[0].plot(tr["time"], tr["speed"], **style)
        axes[1].plot(tr["time"], tr["accel"], **style)
        axes[2].plot(tr["time"], tr["lag_decel"], **style)
        handles.append(Line2D([], [], label=LABELS[name], **style))
        # Vertical line at the merge time, same colour as the policy.
        merged = tr[tr["merged"]]
        if len(merged):
            for ax in axes:
                ax.axvline(merged["time"].iloc[0], color=COLORS[name], ls=":", lw=1.0)
            handles.append(Line2D([], [], color=COLORS[name], ls=":", lw=1.0,
                                  label=f"{LABELS[name].split()[0]} merges"))

    # Action limit of the ego car, and the hard-brake threshold for the lag car.
    for a in (-A_MAX, A_MAX):
        axes[1].axhline(a, color="0.6", ls="--", lw=0.6, zorder=0)
    axes[1].text(0.99, A_MAX, "action limit ±3", transform=axes[1].get_yaxis_transform(),
                 ha="right", va="bottom", color="0.4")
    axes[2].axhline(HARD_BRAKE, color="#D55E00", ls="--", lw=0.8)
    axes[2].text(0.99, HARD_BRAKE, "hard-brake threshold",
                 transform=axes[2].get_yaxis_transform(), ha="right", va="bottom",
                 color="#D55E00")
    axes[1].set_ylim(-A_MAX - 0.6, A_MAX + 1.2)
    axes[2].set_ylim(min(HARD_BRAKE - 0.8, axes[2].get_ylim()[0]), 0.5)

    axes[0].set_ylabel("ego speed [m/s]")
    axes[1].set_ylabel("ego accel [m/s²]")
    axes[2].set_ylabel("lag-car accel\n[m/s²]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.align_ylabels(axes)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.0), title=f"density {density:.0f} veh/km, "
                                                 f"scenario seed {seed}")
    fig.tight_layout(h_pad=0.4)
    save(fig, fig_dir, "trace")


# --------------------------------------------------------------------------- #
# Environment snapshot
# --------------------------------------------------------------------------- #
def draw_scene(ax, env_cfg: dict, state: dict) -> None:
    """Draw one stored scene state on `ax`, in the same style as env/render.py.

    env/render.py makes its own 8-inch-wide figure with fixed font sizes, which
    becomes unreadable when shrunk to one column, so the drawing is repeated
    here on a given axis, reusing the geometry constants of env/render.py.
    """
    from env.render import CAR_W, LANE_W, VIEW
    from env.traffic import CAR_LENGTH
    c = env_cfg
    ax.set_xlim(*VIEW)
    ax.set_ylim(-LANE_W - 0.5, LANE_W + 0.5)
    ax.set_yticks([])
    ax.grid(False)
    for side in ["left", "top", "right"]:
        ax.spines[side].set_visible(False)
    # Highway lane (y in [0, 4]), acceleration lane (y in [-4, 0]), merge zone, ramp end.
    ax.add_patch(Rectangle((VIEW[0], 0.0), VIEW[1] - VIEW[0], LANE_W, color="0.85", lw=0))
    ax.add_patch(Rectangle((VIEW[0], -LANE_W), c["merge_end"] - VIEW[0], LANE_W,
                           color="0.93", lw=0))
    ax.add_patch(Rectangle((c["merge_start"], -LANE_W), c["merge_end"] - c["merge_start"],
                           LANE_W, color="#b9e4b9", lw=0))
    ax.plot([c["merge_end"]] * 2, [-LANE_W, 0.0], color="k", lw=2)
    lane_y = LANE_W / 2 - CAR_W / 2
    for x in state["highway_x"]:
        ax.add_patch(Rectangle((x - CAR_LENGTH, lane_y), CAR_LENGTH, CAR_W, color="0.4"))
    ego_y = lane_y if state["merged"] else -LANE_W / 2 - CAR_W / 2
    ax.add_patch(Rectangle((state["x"] - CAR_LENGTH, ego_y), CAR_LENGTH, CAR_W,
                           color="#d62728"))


def plot_merge_snapshot(ckpt: str, fig_dir: Path, density: float, seed: int) -> None:
    """3-4 frames of one PPO episode: approaching, just before, at and after the merge."""
    from eval import make_policy
    from env.merge_env import MergeEnv
    if not Path(ckpt).exists():
        print(f"{ckpt} missing, skipping merge snapshot")
        return
    env = MergeEnv()
    policy = make_policy("ppo", ckpt, seed)
    obs, _ = env.reset(seed=seed, options={"density": density})

    # Store a copy of the scene after every decision step.
    def snapshot() -> dict:
        return {"t": env.t, "x": env.x, "v": env.v, "merged": env.merged,
                "highway_x": env.highway.x.copy()}

    states, done = [snapshot()], False
    while not done:
        obs, _, term, trunc, _ = env.step(policy(obs, env))
        states.append(snapshot())
        done = term or trunc
    merge_idx = next((i for i, s in enumerate(states) if s["merged"]), None)
    if merge_idx is None:
        print("PPO did not merge in the snapshot episode, skipping merge snapshot")
        return

    t_merge = states[merge_idx]["t"]
    # Frame times relative to the merge; the first frame is the first one where
    # the ego car is inside the drawn window.
    from env.render import VIEW
    first = next(i for i, s in enumerate(states) if s["x"] >= VIEW[0] + 10)
    picks = [first]
    for dt in (-1.5, 0.0, 3.0):
        idx = int(np.argmin([abs(s["t"] - (t_merge + dt)) for s in states]))
        picks.append(merge_idx if dt == 0.0 else idx)
    picks = sorted(set(picks))

    fig, axes = plt.subplots(len(picks), 1, figsize=(WIDTH, 0.62 * len(picks) + 0.35),
                             sharex=True)
    for ax, i in zip(axes, picks):
        s = states[i]
        draw_scene(ax, env.cfg, s)
        rel = s["t"] - t_merge
        when = "merge" if i == merge_idx else f"merge {rel:+.1f} s"
        ax.set_title(f"t = {s['t']:.1f} s ({when}), v = {s['v']:.1f} m/s", loc="left",
                     pad=2)
        ax.tick_params(axis="x", length=2)
    axes[-1].set_xlabel("position [m]")
    # Legend for the drawing elements.
    handles = [Patch(color="#d62728", label="ego (PPO)"), Patch(color="0.4", label="highway car"),
               Patch(color="#b9e4b9", label="merge zone"),
               Line2D([], [], color="k", lw=2, label="ramp end")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8,
               handletextpad=0.4)
    fig.tight_layout(h_pad=0.3)
    save(fig, fig_dir, "merge_snapshot")


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
    plot_merge_snapshot(args.trace_ckpt, fig_dir, args.trace_density, args.trace_seed)
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main()
