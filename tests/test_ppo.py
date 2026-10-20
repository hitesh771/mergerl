"""Tests for the PPO building blocks."""
import numpy as np
import torch

from agents.ppo import PPO, RunningMeanStd, compute_gae


def test_gae_hand_computed():
    # gamma = 0.9, lambda = 0.5, one env, three steps, episode ends after t=1.
    rewards = np.array([[1.0], [0.0], [2.0]])
    values = np.array([[0.5], [1.0], [0.2]])
    dones = np.array([[0.0], [1.0], [0.0]])
    last_value = np.array([1.0])
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.9, lam=0.5)
    # t=2: delta = 2 + 0.9*1.0 - 0.2 = 2.7                 -> A = 2.7
    # t=1: episode ends, delta = 0 - 1.0 = -1.0            -> A = -1.0
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4, A = 1.4 + 0.45*(-1.0) = 0.95
    assert np.allclose(adv[:, 0], [0.95, -1.0, 2.7])
    assert np.allclose(ret[:, 0], [1.45, 0.0, 2.9])


def test_gae_lambda_one_equals_discounted_return_minus_value():
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.zeros((3, 1))
    dones = np.zeros((3, 1))
    adv, _ = compute_gae(rewards, values, dones, np.array([0.0]), gamma=0.5, lam=1.0)
    assert np.allclose(adv[:, 0], [1.75, 1.5, 1.0])


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(1000, 4))
    rms = RunningMeanStd(4)
    for chunk in np.split(data, 10):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0), atol=1e-3)
    assert np.allclose(rms.var, data.var(0), atol=1e-2)


def test_save_load_roundtrip(tmp_path):
    cfg = {"hidden": 16, "lr": 1e-3}
    agent = PPO(5, 1, cfg)
    agent.obs_rms.update(np.random.default_rng(0).normal(size=(50, 5)))
    path = tmp_path / "ppo.pt"
    agent.save(str(path))
    other = PPO.load(str(path), 5)
    obs = np.ones((3, 5), dtype=np.float32)
    assert np.allclose(agent.act(obs, deterministic=True), other.act(obs, deterministic=True))
    assert torch.equal(agent.actor.log_std, other.actor.log_std)
