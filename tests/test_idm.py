"""Unit tests for the IDM car-following model."""
import numpy as np

from env.idm import B_MAX, FREE_GAP, idm_accel, sample_driver_params

P = dict(v0=30.0, T=1.5, a_max=1.5, b=2.0, s0=2.0)


def simulate_follower(leader_x, v_follow, x_follow, steps, dt=0.1):
    """Simulate one IDM car behind a stopped leader; return final x, v and min gap."""
    min_gap = np.inf
    for _ in range(steps):
        gap = leader_x - x_follow
        a = idm_accel(v_follow, gap, v_follow - 0.0, **P)
        v_new = max(v_follow + a * dt, 0.0)
        x_follow += 0.5 * (v_follow + v_new) * dt
        v_follow = v_new
        min_gap = min(min_gap, leader_x - x_follow)
    return x_follow, v_follow, min_gap


def test_free_road_accel_from_standstill_is_a_max():
    a = idm_accel(0.0, FREE_GAP, 0.0, **P)
    assert np.isclose(a, P["a_max"])


def test_free_road_accel_is_zero_at_desired_speed():
    a = idm_accel(P["v0"], FREE_GAP, 0.0, **P)
    assert abs(a) < 1e-3


def test_brakes_when_gap_is_small():
    a = idm_accel(20.0, 5.0, 0.0, **P)
    assert a < 0.0


def test_deceleration_is_capped():
    a = idm_accel(30.0, 0.5, 30.0, **P)
    assert np.isclose(a, -B_MAX)


def test_vectorised_matches_scalar():
    v = np.array([0.0, 10.0, 25.0])
    gap = np.array([FREE_GAP, 20.0, 8.0])
    dv = np.array([0.0, 2.0, -1.0])
    vec = idm_accel(v, gap, dv, **P)
    for i in range(3):
        assert np.isclose(vec[i], idm_accel(v[i], gap[i], dv[i], **P))


def test_stops_before_stationary_leader():
    # Car at 15 m/s, stopped obstacle 120 m ahead: must stop without contact.
    _, v_end, min_gap = simulate_follower(120.0, 15.0, 0.0, steps=600)
    assert v_end < 0.1
    assert min_gap > 0.5


def test_sampled_params_within_ranges():
    rng = np.random.default_rng(0)
    p = sample_driver_params(1000, rng)
    assert p["v0"].min() > 24.0 * 0.94 and p["v0"].max() < 33.0 * 1.06
    assert p["T"].min() > 1.0 * 0.94 and p["T"].max() < 1.8 * 1.06


def test_mild_braking_for_distant_slower_cut_in():
    # Merging car 35 m ahead, 7 m/s slower: no emergency braking needed.
    a = idm_accel(22.0, 35.0, 7.0, **P)
    assert -3.5 < a < 0.0


def test_hard_braking_for_close_slower_cut_in():
    # Merging car 6 m ahead and 10 m/s slower: must brake hard.
    a = idm_accel(25.0, 6.0, 10.0, **P)
    assert a < -6.0
