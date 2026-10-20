"""Rule-based comparison agents (NOT learning agents).

All agents share the interface `act(obs, env) -> action in [-1, 1]`, where
`env` is the (unwrapped) MergeEnv. The rule-based agents read the simulator
state directly (positions and speeds of all cars), i.e. they have *more*
information than the RL agents, which only see the 13-d observation.
"""
import numpy as np

from env.idm import idm_accel
from env.merge_env import A_MAX
from env.traffic import CAR_LENGTH

# IDM parameters used whenever a baseline simply follows its leader.
EGO_IDM = {"v0": 30.0, "T": 1.5, "a_max": 2.0, "b": 2.5, "s0": 2.0}


def follow_accel(v, gap, v_lead):
    """IDM acceleration of the ego following a leader."""
    return float(idm_accel(v, gap, v - v_lead, **EGO_IDM))


def lead_state(env):
    """Gap and speed of the highway car ahead of the ego (free road if none)."""
    hw = env.highway
    lead, _ = hw.neighbours(env.x)
    if lead is None:
        return 1e4, env.v
    return hw.x[lead] - CAR_LENGTH - env.x, hw.v[lead]


class RandomAgent:
    """Uniformly random acceleration every decision step."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        return np.array([self.rng.uniform(-1.0, 1.0)], dtype=np.float32)


class IDMEgoAgent:
    """Ego drives like an ordinary IDM car and relies on the auto-merge.

    On the ramp it treats the ramp end as a stopped virtual leader, so it
    slows down towards the end; it merges as soon as the environment allows,
    without looking at the highway traffic. After merging it follows the
    highway leader with IDM.
    """

    def act(self, obs, env):
        if env.merged:
            gap, v_lead = lead_state(env)
        else:
            gap, v_lead = env.cfg["merge_end"] - env.x, 0.0
        return np.array([follow_accel(env.v, gap, v_lead) / A_MAX], dtype=np.float32)


class GapAcceptanceAgent:
    """Classic gap-acceptance merging controller with a timing plan.

    The environment merges the ego as soon as it is inside the merge zone
    next to any gap, so the ego has to *arrive* at the zone entry at the
    same time as a good gap. Every decision step the controller:

    1. Lists all gaps between consecutive highway cars whose length (bumper
       to bumper) is at least `min_gap` metres.
    2. For each gap computes the time T at which the gap centre (moving at
       the mean speed v_g of its two cars) reaches the entry point
       x_m = merge_start + entry_offset.
    3. Plans a constant acceleration that brings the ego to x_m in exactly
       T seconds:  D = v T + a T^2 / 2  =>  a = 2 (D - v T) / T^2,
       which gives an arrival speed v_end = v + a T.
    4. Keeps only feasible plans (|a| <= a_plan, 3 <= v_end <= 33 m/s) and
       picks the one whose arrival speed is closest to the gap speed v_g
       (the smaller the speed mismatch, the less the lag car must brake).
    5. If no plan is feasible it waits at low speed before the zone.

    Inside the zone (merged or waiting next to a car) it follows the car
    ahead with IDM. `min_gap` sets the risk level: small = aggressive,
    large = conservative.
    """

    def __init__(self, min_gap: float = 10.0, a_plan: float = 2.5,
                 entry_offset: float = 10.0):
        self.min_gap = min_gap
        self.a_plan = a_plan
        self.entry_offset = entry_offset

    def plan(self, env):
        """Return the planned acceleration, or None if no gap is feasible."""
        hw = env.highway
        if len(hw.x) < 2:
            return None
        front = hw.x[:-1] - CAR_LENGTH         # rear bumper of the front car
        rear = hw.x[1:]                        # front bumper of the rear car
        size = front - rear
        v_g = 0.5 * (hw.v[:-1] + hw.v[1:])
        # Ego front in the middle of the gap: rear + L + (size - L) / 2.
        centre = rear + 0.5 * (size + CAR_LENGTH)
        x_m = env.cfg["merge_start"] + self.entry_offset
        T = (x_m - centre) / np.maximum(v_g, 1.0)
        D = x_m - env.x
        T_safe = np.maximum(T, 1e-3)
        a = 2.0 * (D - env.v * T_safe) / T_safe ** 2
        v_end = env.v + a * T_safe
        ok = ((size >= self.min_gap + CAR_LENGTH) & (T > 0.5)
              & (np.abs(a) <= self.a_plan) & (v_end >= 3.0) & (v_end <= 33.0))
        if not np.any(ok):
            return None
        k = np.flatnonzero(ok)[np.argmin(np.abs(v_end[ok] - v_g[ok]))]
        return float(a[k])

    def act(self, obs, env):
        c = env.cfg
        if env.merged:
            gap, v_lead = lead_state(env)
            accel = follow_accel(env.v, gap, v_lead)
        elif env.x >= c["merge_start"]:
            # In the zone but blocked by a car alongside: match the speed
            # of the highway car just ahead so the gap behind it opens.
            gap, v_lead = lead_state(env)
            accel = follow_accel(env.v, max(gap + CAR_LENGTH + 10.0, 1.0), v_lead)
        else:
            accel = self.plan(env)
            if accel is None:
                # No feasible gap yet: approach the zone at a low speed
                # (IDM towards a virtual stop line at merge_start).
                accel = follow_accel(env.v, c["merge_start"] - env.x, 0.0)
        # Never run into the end of the ramp at speed.
        dist_end = c["merge_end"] - env.x
        if not env.merged and dist_end > 0 and env.v ** 2 / (2 * dist_end) > 2.5:
            accel = min(accel, -env.v ** 2 / (2 * dist_end))
        return np.array([np.clip(accel / A_MAX, -1.0, 1.0)], dtype=np.float32)
