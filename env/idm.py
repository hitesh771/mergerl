"""Intelligent Driver Model (IDM) for longitudinal car following.

The IDM (Treiber, Hennecke & Helbing, 2000) gives the acceleration of a car
from its own speed v, the bumper-to-bumper gap s to its leader and the
approach rate dv = v - v_leader:

    a = a_max * [ 1 - (v / v0)^delta - (s_star / s)^2 ]

    s_star = s0 + v*T + v*dv / (2 * sqrt(a_max * b))

Intuition:
  * (v / v0)^delta  -> "free road" term, the car accelerates towards its
                       desired speed v0 and stops accelerating near it.
  * (s_star / s)^2  -> "interaction" term. s_star is the gap the driver
                       *wants* (jam distance s0 + time headway T, plus an
                       extra term when closing in on the leader). If the
                       real gap s is smaller than s_star the car brakes.

All functions work on numpy arrays so a whole lane of cars is updated with a
handful of vector operations (no Python loop over cars).
"""
import numpy as np

# Physical limit on braking. IDM itself can ask for unbounded deceleration
# when a gap is very small; real cars cannot brake harder than ~8 m/s^2.
# Because of this cap a car that is cut off too closely can crash into the
# car in front of it, which is exactly what makes bad merges dangerous.
B_MAX = 8.0

# Cars that have no leader get this gap (effectively an empty road).
FREE_GAP = 1e4

# Parameter ranges for the two extreme driver types. Each car gets an
# "aggressiveness" u in [0, 1]; its parameters are interpolated between the
# cautious (u = 0) and aggressive (u = 1) values.
CAUTIOUS = {"v0": 24.0, "T": 1.8, "a_max": 1.0, "b": 1.5, "s0": 2.5}
AGGRESSIVE = {"v0": 33.0, "T": 1.0, "a_max": 2.0, "b": 2.5, "s0": 1.5}
DELTA = 4.0


def sample_driver_params(n: int, rng: np.random.Generator) -> dict:
    """Sample IDM parameters for n drivers.

    Returns a dict of arrays with keys v0, T, a_max, b, s0 (each shape (n,)).
    A small independent noise (+-5 %) is added on top of the interpolation
    so that two drivers with the same aggressiveness still differ a little.
    """
    u = rng.uniform(0.0, 1.0, size=n)
    params = {}
    for key in CAUTIOUS:
        base = CAUTIOUS[key] + u * (AGGRESSIVE[key] - CAUTIOUS[key])
        params[key] = base * rng.uniform(0.95, 1.05, size=n)
    return params


def idm_accel(v, gap, dv, v0, T, a_max, b, s0, delta=DELTA, b_max=B_MAX):
    """Vectorised IDM acceleration.

    Args:
        v:     own speed [m/s]
        gap:   bumper-to-bumper gap to the leader [m] (FREE_GAP if none)
        dv:    approach rate v - v_leader [m/s] (positive = closing in)
        v0, T, a_max, b, s0: IDM driver parameters (scalars or arrays)
        delta: free-road exponent
        b_max: maximal physical deceleration (positive number)

    Returns:
        acceleration [m/s^2], clipped to [-b_max, a_max].
    """
    v = np.asarray(v, dtype=float)
    # Desired dynamic gap. max(., s0) keeps it from going below the jam
    # distance when the leader is much faster (dv very negative).
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a_max * b)))
    # A gap <= 0 means overlap (crash); use a tiny positive value so the
    # formula yields maximal braking instead of dividing by zero.
    s = np.maximum(gap, 0.01)
    accel = a_max * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
    return np.clip(accel, -b_max, a_max)
