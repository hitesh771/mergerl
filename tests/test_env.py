"""Tests for the merge environment."""
import numpy as np

from env.merge_env import DiscreteActionWrapper, MergeEnv
from env.traffic import CAR_LENGTH


def run_episode(env, policy, seed, density=None):
    opts = {"density": density} if density is not None else None
    obs, _ = env.reset(seed=seed, options=opts)
    total, done, info = 0.0, False, {}
    traj = [obs]
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        traj.append(obs)
        done = term or trunc
    return total, info, np.array(traj)


def empty_road(env):
    """Remove all highway cars (used to test the ego alone)."""
    hw = env.highway
    hw.x, hw.v, hw.a = np.zeros(0), np.zeros(0), np.zeros(0)
    hw.params = {k: np.zeros(0) for k in hw.params}
    hw.next_spawn_spacing = np.inf


def test_reset_and_step_shapes():
    env = MergeEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (MergeEnv.OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(np.array([0.5]))
    assert obs.shape == (MergeEnv.OBS_DIM,)
    assert isinstance(r, float)
    for key in ["collision", "merged", "failed", "speed", "accel", "jerk",
                "ttc", "lag_decel", "time"]:
        assert key in info


def test_observations_stay_in_bounds():
    env = MergeEnv()
    rng = np.random.default_rng(0)
    for seed in range(10):
        _, _, traj = run_episode(env, lambda o: rng.uniform(-1, 1, 1), seed)
        assert np.all(np.abs(traj) <= 2.0)


def test_same_seed_gives_same_episode():
    env1, env2 = MergeEnv(), MergeEnv()
    policy = lambda o: np.array([np.sin(10 * o[1])])
    r1, i1, t1 = run_episode(env1, policy, seed=42)
    r2, i2, t2 = run_episode(env2, policy, seed=42)
    assert r1 == r2 and np.array_equal(t1, t2)
    r3, _, t3 = run_episode(env1, policy, seed=43)
    assert not (len(t1) == len(t3) and np.array_equal(t1, t3))


def test_ramp_end_without_merge_is_failure():
    env = MergeEnv()
    env.reset(seed=0)
    hw = env.highway
    # A solid wall of cars alongside the whole ramp: merging is impossible.
    hw.x = np.arange(700.0, -1.0, -(CAR_LENGTH + 1.0))
    n = len(hw.x)
    hw.v = np.full(n, 15.0)
    hw.a = np.zeros(n)
    hw.params = {k: np.full(n, v) for k, v in
                 {"v0": 15.0, "T": 0.1, "a_max": 1.0, "b": 2.0, "s0": 1.0}.items()}
    done, info = False, {}
    while not done:
        _, r, done, _, info = env.step(np.array([1.0]))
    assert info["failed"] and not info["merged"] and not info["collision"]


def test_merges_on_empty_road_and_succeeds():
    env = MergeEnv()
    env.reset(seed=1)
    empty_road(env)
    done, info, merged_reward = False, {}, None
    while not done:
        _, r, done, _, info = env.step(np.array([0.3]))
        if info["just_merged"]:
            merged_reward = r
            assert env.cfg["merge_start"] <= info["x"] <= env.cfg["merge_end"]
    assert info["success"] and info["merged"]
    assert merged_reward > env.rw["R_merge"] - 1.0


def test_no_merge_before_zone():
    env = MergeEnv()
    env.reset(seed=2)
    empty_road(env)
    while env.x < env.cfg["merge_start"] - 2.0:
        _, _, _, _, info = env.step(np.array([0.0]))
        assert not info["merged"]


def test_collision_when_cutting_in_front_of_fast_car():
    env = MergeEnv()
    env.reset(seed=3)
    hw = env.highway
    env.x, env.v = env.cfg["merge_start"] + 1.0, 5.0
    # One fast car 8 m behind the ego: the gap allows a merge (>= 2 m) but
    # the car cannot brake from 30 m/s to 5 m/s within 8 m.
    hw.x = np.array([env.x - CAR_LENGTH - 8.0])
    hw.v = np.array([30.0])
    hw.a = np.zeros(1)
    hw.params = {k: np.array([v]) for k, v in
                 {"v0": 30.0, "T": 1.2, "a_max": 1.5, "b": 2.0, "s0": 2.0}.items()}
    hw.next_spawn_spacing = np.inf
    done, info = False, {}
    while not done:
        _, r, done, _, info = env.step(np.array([-1.0]))
    assert info["merged"] and info["collision"]
    assert r < -env.rw["R_col"] + 1.0


def test_discrete_wrapper_maps_levels():
    env = DiscreteActionWrapper(MergeEnv(), n_actions=7)
    env.reset(seed=0)
    assert env.action_space.n == 7
    _, _, _, _, info = env.step(0)
    assert np.isclose(info["accel"], -3.0)
    _, _, _, _, info = env.step(6)
    assert np.isclose(info["accel"], 3.0)
