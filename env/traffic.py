"""Single-lane highway traffic driven by the IDM.

Road geometry (all positions in metres, x grows in the driving direction):

    highway lane  0 ------------------------------------------------ road_length
    acc. lane              ramp_start ===== [merge_start ==== merge_end] |
                                                                       hard end

Conventions:
  * x of a car is the position of its FRONT bumper; every car is CAR_LENGTH
    long, so the bumper-to-bumper gap from car i to its leader j is
        gap = x_j - CAR_LENGTH - x_i
  * The highway arrays are kept sorted by position, index 0 = most downstream
    car. Car i's leader is therefore car i - 1.

Steady flow: cars leaving the road at road_length are deleted and new cars
are inserted at x = 0 whenever the most upstream car has moved far enough
away. The spacing between inserted cars is drawn around the target spacing
1000 / density, so the density on the road stays roughly constant.

Interaction with the ego: once the ego has merged it is inserted into the
leader chain, i.e. the first highway car behind the ego follows the ego.
Before the merge highway drivers ignore the ego (it is on another lane).
"""
import numpy as np

from env.idm import idm_accel, sample_driver_params

CAR_LENGTH = 5.0


def equilibrium_speed(gap: np.ndarray, params: dict) -> np.ndarray:
    """Speed at which an IDM driver is happy with a given gap (approximation).

    In steady state (dv = 0, a = 0) and ignoring the free-road term the IDM
    gives gap = s0 + v*T, so v = (gap - s0) / T. Clipped to [0, v0]. Used to
    give freshly spawned cars a speed that matches their spacing so the
    simulation starts without large braking waves.
    """
    v = (gap - params["s0"]) / params["T"]
    return np.clip(v, 0.0, params["v0"])


class Highway:
    """IDM traffic on the main highway lane."""

    def __init__(self, road_length: float, rng: np.random.Generator):
        self.road_length = road_length
        self.rng = rng
        self.density = 0.0
        self.x = np.zeros(0)
        self.v = np.zeros(0)
        self.a = np.zeros(0)
        self.params = {}
        self.next_spawn_spacing = 0.0

    # ------------------------------------------------------------------ setup
    def _sample_spacing(self, n: int = 1) -> np.ndarray:
        """Front-to-front spacing between consecutive cars [m].

        Mean is 1000 / density (density in veh/km); +-40 % uniform noise
        gives a mix of small and large gaps. It never goes below 1.5 car
        lengths so cars do not start overlapping.
        """
        mean = 1000.0 / self.density
        spacing = mean * self.rng.uniform(0.6, 1.4, size=n)
        return np.maximum(spacing, 1.5 * CAR_LENGTH)

    def reset(self, density: float) -> None:
        """Fill the whole road with cars at the given density [veh/km]."""
        self.density = density
        n_max = int(self.road_length * density / 1000.0 * 2) + 2
        spacing = self._sample_spacing(n_max)
        # Cumulative positions from the downstream end backwards.
        offset = self.rng.uniform(0.0, spacing[0])
        x = self.road_length - offset - np.concatenate(([0.0], np.cumsum(spacing[1:])))
        x = x[x >= 0.0]
        n = len(x)
        self.x = x
        self.params = sample_driver_params(n, self.rng)
        gaps = spacing[:n] - CAR_LENGTH
        noise = self.rng.uniform(0.9, 1.05, size=n)
        self.v = equilibrium_speed(gaps, self.params) * noise
        self.a = np.zeros(n)
        self.next_spawn_spacing = float(self._sample_spacing()[0])

    # --------------------------------------------------------------- queries
    def neighbours(self, ego_x: float):
        """Indices of the lead and lag car relative to a position.

        lead = closest car whose front is ahead of (or level with) ego_x,
        lag  = closest car whose front is behind ego_x.
        Returns (lead_idx, lag_idx); either may be None.
        """
        # x is sorted descending; count cars ahead of the ego.
        n_ahead = int(np.sum(self.x >= ego_x))
        lead = n_ahead - 1 if n_ahead > 0 else None
        lag = n_ahead if n_ahead < len(self.x) else None
        return lead, lag

    # ------------------------------------------------------------- dynamics
    def step(self, dt: float, ego_x: float, ego_v: float, ego_merged: bool) -> None:
        """Advance all highway cars by dt seconds."""
        n = len(self.x)
        if n > 0:
            lead_x = np.empty(n)
            lead_v = np.empty(n)
            lead_x[1:] = self.x[:-1]
            lead_v[1:] = self.v[:-1]
            # The most downstream car follows a virtual car at the nominal
            # spacing driving at its own speed; this keeps the platoon from
            # spreading out at the exit (an open boundary would send an
            # acceleration wave upstream into the merge area).
            lead_x[0] = self.x[0] + 1000.0 / self.density
            lead_v[0] = self.v[0]
            if ego_merged:
                _, lag = self.neighbours(ego_x)
                if lag is not None:
                    lead_x[lag] = ego_x
                    lead_v[lag] = ego_v
            gap = lead_x - CAR_LENGTH - self.x
            p = self.params
            self.a = idm_accel(self.v, gap, self.v - lead_v,
                               p["v0"], p["T"], p["a_max"], p["b"], p["s0"])
            # Ballistic update: average of old and new speed, no reversing.
            v_new = np.maximum(self.v + self.a * dt, 0.0)
            self.x = self.x + 0.5 * (self.v + v_new) * dt
            self.v = v_new
            self._resolve_overlaps(ego_x, ego_merged)
        self._recycle()

    def _resolve_overlaps(self, ego_x: float, ego_merged: bool) -> None:
        """Push back highway cars that overlap a highway leader.

        IDM with capped braking can, in rare chain reactions caused by a bad
        merge, rear-end a highway car. Those crashes do not involve the ego,
        so we simply place the follower right behind its leader with the
        leader's speed. Crashes with the ego are detected by the env, so the
        car directly behind a merged ego is excluded here.
        """
        if len(self.x) < 2:
            return
        gap = self.x[:-1] - CAR_LENGTH - self.x[1:]
        bad = np.nonzero(gap < 0.0)[0] + 1
        if ego_merged:
            _, lag = self.neighbours(ego_x)
            bad = bad[bad != lag]
        # Process from downstream to upstream so fixes propagate.
        for i in bad:
            self.x[i] = self.x[i - 1] - CAR_LENGTH
            self.v[i] = min(self.v[i], self.v[i - 1])

    def _recycle(self) -> None:
        """Delete cars past the road end and insert cars at the upstream end."""
        keep = self.x <= self.road_length
        if not np.all(keep):
            self.x, self.v, self.a = self.x[keep], self.v[keep], self.a[keep]
            self.params = {k: val[keep] for k, val in self.params.items()}
        last_x = self.x[-1] if len(self.x) else np.inf
        if last_x >= self.next_spawn_spacing:
            new_p = sample_driver_params(1, self.rng)
            gap = np.array([last_x - CAR_LENGTH])
            v_new = equilibrium_speed(gap, new_p)
            if len(self.v):
                # Do not insert a car much faster than the one it follows.
                v_new = np.minimum(v_new, self.v[-1] + 2.0)
            self.x = np.append(self.x, 0.0)
            self.v = np.append(self.v, v_new)
            self.a = np.append(self.a, 0.0)
            self.params = {k: np.append(val, new_p[k]) for k, val in self.params.items()}
            self.next_spawn_spacing = float(self._sample_spacing()[0])
