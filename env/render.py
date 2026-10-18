"""Top-down matplotlib rendering of the merge scene and GIF export.

Only the region around the merge zone is drawn (x from 100 m to 400 m) so
the cars are large enough to see. Highway lane on top, acceleration lane
below, merge zone shaded green, ego car red, other cars grey.
"""
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from env.traffic import CAR_LENGTH  # noqa: E402

VIEW = (100.0, 400.0)
LANE_W = 4.0
CAR_W = 2.0


def render_frame(env) -> np.ndarray:
    """Draw the current state of a MergeEnv and return an RGB array."""
    c = env.cfg
    fig, ax = plt.subplots(figsize=(8, 1.6), dpi=100)
    ax.set_xlim(*VIEW)
    ax.set_ylim(-LANE_W - 1.0, LANE_W + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("position [m]", fontsize=8)
    ax.tick_params(labelsize=7)

    # Road: highway lane (y in [0, 4]) and acceleration lane (y in [-4, 0]).
    ax.add_patch(Rectangle((VIEW[0], 0.0), VIEW[1] - VIEW[0], LANE_W, color="0.85"))
    ax.add_patch(Rectangle((VIEW[0], -LANE_W), c["merge_end"] - VIEW[0], LANE_W, color="0.92"))
    ax.add_patch(Rectangle((c["merge_start"], -LANE_W), c["merge_end"] - c["merge_start"],
                           LANE_W, color="#b9e4b9"))
    ax.plot([c["merge_end"], c["merge_end"]], [-LANE_W, 0.0], color="k", lw=2)

    lane_y = LANE_W / 2 - CAR_W / 2
    for x in env.highway.x:
        ax.add_patch(Rectangle((x - CAR_LENGTH, lane_y), CAR_LENGTH, CAR_W, color="0.4"))
    ego_y = lane_y if env.merged else -LANE_W / 2 - CAR_W / 2
    ax.add_patch(Rectangle((env.x - CAR_LENGTH, ego_y), CAR_LENGTH, CAR_W, color="#d62728"))

    ax.set_title(f"t = {env.t:4.1f} s   v = {env.v:4.1f} m/s   a = {env.a:+.1f} m/s²   "
                 f"density = {env.density:.0f} veh/km", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_gif(frames: list, path: str, fps: int = 10) -> None:
    """Write a list of RGB frames to a GIF file."""
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
