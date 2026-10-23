"""Highway on-ramp merging environment (gymnasium API).

The ego car starts on the acceleration lane, upstream of the merge zone.
It controls only its longitudinal acceleration. While it is inside the
merge zone [merge_start, merge_end] it changes lane automatically as soon as
the bumper-to-bumper gaps to the highway cars in front (lead) and behind
(lag) are both at least `merge_min_gap` metres. This minimum is deliberately
small (2 m): the agent is *allowed* to cut in badly, but then the lag car has
to brake hard (IDM braking is capped at 8 m/s^2) and may even crash into the
ego. The interesting decision is therefore *which speed profile* brings the
ego next to a large gap at a matching speed.

Episode end:
  * collision (ego overlaps a highway car)                 -> failure
  * ego reaches the hard end of the ramp without merging   -> failure
  * time limit reached before merging                      -> failure
  * post_merge_time seconds after a merge without crash    -> success

Timing: the simulation runs at dt = 0.1 s; the agent chooses a new action
every `action_repeat` = 2 sub-steps, i.e. at 5 Hz. This halves the episode
length in decisions (easier credit assignment) while keeping an accurate
simulation.

Observation (13 values, roughly in [-1, 1]):

    idx  name              definition                                   scale
    ---  ----------------  -------------------------------------------  ---------
    0    ego_speed         v_ego / V_SCALE                              [0, 1.2]
    1    ego_pos           (x_ego - merge_start) / zone_length          [-1.5, 1.5]
    2    dist_ramp_end     (merge_end - x_ego) / 100, 0 once merged     [-1, 1]
    3    merged            1 if ego is on the highway, else 0           {0, 1}
    4    lead_gap          gap to highway lead / GAP_SCALE (1 if none)  [-1, 1]
    5    lead_rel_speed    (v_lead - v_ego) / DV_SCALE                  [-1, 1]
    6    lag_gap           gap to highway lag / GAP_SCALE (1 if none)   [-1, 1]
    7    lag_rel_speed     (v_lag - v_ego) / DV_SCALE                   [-1, 1]
    8    lag_ttc           TTC of lag car to ego, clip [0,10] s, /5 - 1 [-1, 1]
    9    lag_speed         v_lag / V_SCALE                              [0, 1.2]
    10   density           density / 50 veh/km                          [0, 1.5]
    11   prev_accel        previous ego acceleration / 3 m/s^2          [-1, 1]
    12   lag_headway       lag gap / v_lag, clip [0,4] s, /2 - 1        [-1, 1]

A negative gap means the highway car is (partly) alongside the ego, which
is what blocks a merge. Index 11 lets the policy see the jerk it would
cause; index 12 tells it how hard the lag car will have to brake.

Action: Box([-1], [1]) scaled to acceleration in [-3, +3] m/s^2.

Reward per decision step (weights in config["reward"]):

    r =  w_speed * v / V_MAX                  progress, keep traffic flowing
       - w_jerk  * |a - a_prev| / 6           comfort (6 = max change)
       - w_ttc   * [merged and TTC < 2 s]     unsafe proximity (lead or lag)
       - w_brake * [lag car accel < -4]       forcing the lag car to brake hard
       - w_stop  * [not merged and v < 2]     prevents "stop on the ramp forever"
       - w_time                               small cost per step, merge sooner
    terminal / event bonuses:
       + R_merge     at the moment of merging
       + R_complete  surviving post_merge_time after the merge
       - R_col       collision
       - R_fail      ramp end or time-out without merging
"""
import gymnasium as gym
import numpy as np

from env.traffic import CAR_LENGTH, Highway

V_SCALE = 30.0
V_MAX = 35.0
GAP_SCALE = 50.0
DV_SCALE = 10.0
A_MAX = 3.0
TTC_CLIP = 10.0
INF_TTC = 99.0

DEFAULT_CONFIG = {
    "dt": 0.1,
    "action_repeat": 2,
    "road_length": 700.0,
    "merge_start": 200.0,
    "merge_end": 320.0,
    "ego_x_range": [60.0, 100.0],
    "ego_v_range": [12.0, 20.0],
    "density_range": [10.0, 55.0],
    "warmup_time": 3.0,
    "max_time": 40.0,
    "post_merge_time": 8.0,
    "merge_min_gap": 2.0,
    "ttc_threshold": 2.0,
    "hard_brake": -4.0,
    "reward": {
        "w_speed": 0.05,
        "w_jerk": 0.02,
        "w_ttc": 0.3,
        "w_brake": 2.0,
        "w_stop": 0.1,
        "w_time": 0.02,
        "R_merge": 5.0,
        "R_complete": 5.0,
        "R_col": 20.0,
        "R_fail": 10.0,
    },
}


def merge_config(base: dict, override: dict) -> dict:
    """Recursively overlay `override` on `base` (returns a new dict)."""
    out = dict(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], val)
        else:
            out[key] = val
    return out


class MergeEnv(gym.Env):
    """On-ramp merging with one learning ego car and IDM highway traffic."""

    metadata = {"render_modes": ["rgb_array"]}
    OBS_DIM = 13

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = merge_config(DEFAULT_CONFIG, config)
        self.rw = self.cfg["reward"]
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (self.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.fixed_density = None   # set by eval to override density_range

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        c = self.cfg
        if options and "density" in options:
            self.density = float(options["density"])
        elif self.fixed_density is not None:
            self.density = float(self.fixed_density)
        else:
            self.density = float(rng.uniform(*c["density_range"]))
        self.highway = Highway(c["road_length"], rng)
        self.highway.reset(self.density)
        # Let the traffic settle so the episode does not start with the
        # artificial initial spacing.
        for _ in range(int(c["warmup_time"] / c["dt"])):
            self.highway.step(c["dt"], -1.0, 0.0, False)

        self.x = float(rng.uniform(*c["ego_x_range"]))
        self.v = float(rng.uniform(*c["ego_v_range"]))
        self.a = 0.0
        self.merged = False
        self.t = 0.0
        self.merge_time = None
        return self._obs(), {}

    # ------------------------------------------------------------ geometry
    def _neighbours(self):
        """Return (lead_gap, lead_v, lag_gap, lag_v, lag_accel) on the highway.

        Missing neighbours get a huge gap and the ego's own speed.
        """
        hw = self.highway
        lead, lag = hw.neighbours(self.x)
        if lead is None:
            lead_gap, lead_v = 1e3, self.v
        else:
            lead_gap, lead_v = hw.x[lead] - CAR_LENGTH - self.x, hw.v[lead]
        if lag is None:
            lag_gap, lag_v, lag_a = 1e3, self.v, 0.0
        else:
            lag_gap, lag_v, lag_a = self.x - CAR_LENGTH - hw.x[lag], hw.v[lag], hw.a[lag]
        return float(lead_gap), float(lead_v), float(lag_gap), float(lag_v), float(lag_a)

    @staticmethod
    def _ttc(gap: float, closing_speed: float) -> float:
        """Time to collision; INF_TTC if the gap is not shrinking."""
        if closing_speed <= 1e-6:
            return INF_TTC
        return max(gap, 0.0) / closing_speed

    def _obs(self) -> np.ndarray:
        c = self.cfg
        lead_gap, lead_v, lag_gap, lag_v, _ = self._neighbours()
        zone = c["merge_end"] - c["merge_start"]
        lag_ttc = min(self._ttc(lag_gap, lag_v - self.v), TTC_CLIP)
        # Lag-car headway (gap / own speed): the quantity that decides how
        # hard the lag car must brake after a merge (IDM time gap T is 1-1.8 s).
        lag_headway = min(max(lag_gap, 0.0) / max(lag_v, 1.0), 4.0)
        obs = [
            self.v / V_SCALE,
            np.clip((self.x - c["merge_start"]) / zone, -1.5, 1.5),
            0.0 if self.merged else np.clip((c["merge_end"] - self.x) / 100.0, -1, 1),
            float(self.merged),
            np.clip(lead_gap / GAP_SCALE, -1, 1),
            np.clip((lead_v - self.v) / DV_SCALE, -1, 1),
            np.clip(lag_gap / GAP_SCALE, -1, 1),
            np.clip((lag_v - self.v) / DV_SCALE, -1, 1),
            lag_ttc / 5.0 - 1.0,
            lag_v / V_SCALE,
            self.density / 50.0,
            self.a / A_MAX,
            lag_headway / 2.0 - 1.0,
        ]
        return np.asarray(obs, dtype=np.float32)

    # ----------------------------------------------------------------- step
    def step(self, action):
        c, rw = self.cfg, self.rw
        a_new = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)) * A_MAX
        delta_a = abs(a_new - self.a)
        self.a = a_new

        collision = failed = just_merged = completed = False
        min_ttc, lag_decel = INF_TTC, 0.0
        for _ in range(c["action_repeat"]):
            dt = c["dt"]
            # Highway first (uses the ego state at the start of the sub-step),
            # then the ego. Both use the same ballistic update.
            self.highway.step(dt, self.x, self.v, self.merged)
            v_new = float(np.clip(self.v + self.a * dt, 0.0, V_MAX))
            self.x += 0.5 * (self.v + v_new) * dt
            self.v = v_new
            self.t += dt

            lead_gap, lead_v, lag_gap, lag_v, lag_a = self._neighbours()
            if self.merged:
                if lead_gap < 0.0 or lag_gap < 0.0:
                    collision = True
                    break
                lag_decel = min(lag_decel, lag_a)
                min_ttc = min(min_ttc, self._ttc(lead_gap, self.v - lead_v),
                              self._ttc(lag_gap, lag_v - self.v))
                if self.t - self.merge_time >= c["post_merge_time"]:
                    completed = True
                    break
            else:
                in_zone = c["merge_start"] <= self.x <= c["merge_end"]
                if in_zone and lead_gap >= c["merge_min_gap"] and lag_gap >= c["merge_min_gap"]:
                    self.merged = True
                    self.merge_time = self.t
                    just_merged = True
                elif self.x > c["merge_end"]:
                    failed = True
                    break
        timeout = (not self.merged) and self.t >= c["max_time"] and not failed
        failed = failed or timeout
        hard_brake_lag = lag_decel < c["hard_brake"]
        ttc_violation = self.merged and min_ttc < c["ttc_threshold"]

        reward = (rw["w_speed"] * self.v / V_MAX
                  - rw["w_jerk"] * delta_a / (2 * A_MAX)
                  - rw["w_ttc"] * float(ttc_violation)
                  - rw["w_brake"] * float(hard_brake_lag)
                  - rw["w_stop"] * float((not self.merged) and self.v < 2.0)
                  - rw["w_time"])
        if just_merged:
            reward += rw["R_merge"]
        if completed:
            reward += rw["R_complete"]
        if collision:
            reward -= rw["R_col"]
        if failed:
            reward -= rw["R_fail"]

        terminated = collision or failed or completed
        dt_dec = c["dt"] * c["action_repeat"]
        info = {
            "collision": collision,
            "merged": self.merged,
            "just_merged": just_merged,
            "failed": failed,
            "success": completed,
            "speed": self.v,
            "accel": self.a,
            "jerk": delta_a / dt_dec,
            "ttc": min_ttc,
            "lag_decel": lag_decel,
            "time": self.t,
            "x": self.x,
            "density": self.density,
        }
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        from env.render import render_frame
        return render_frame(self)


class DiscreteActionWrapper(gym.ActionWrapper):
    """Maps a discrete action index to one of n evenly spaced accelerations.

    With n = 7 the levels are -3, -2, -1, 0, +1, +2, +3 m/s^2, i.e. the
    same range as the continuous action space. Used by DQN.
    """

    def __init__(self, env: gym.Env, n_actions: int = 7):
        super().__init__(env)
        self.levels = np.linspace(-1.0, 1.0, n_actions, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)

    def action(self, act):
        return np.array([self.levels[int(act)]], dtype=np.float32)
