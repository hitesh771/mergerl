"""Tests for the DQN agent."""
import numpy as np

from agents.dqn import DQN, ReplayBuffer

CFG = {"hidden": 16, "lr": 1e-2, "buffer_size": 100, "batch_size": 32,
       "gamma": 0.9, "eps_start": 1.0, "eps_end": 0.1, "eps_fraction": 0.5,
       "max_grad_norm": 10.0}


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(2, 5, np.random.default_rng(0))
    for i in range(7):
        buf.add(np.full(2, i), i % 3, float(i), np.full(2, i + 1), 0.0)
    assert len(buf) == 5
    assert set(buf.rew.tolist()) == {2.0, 3.0, 4.0, 5.0, 6.0}


def test_epsilon_schedule():
    agent = DQN(3, 4, CFG)
    assert agent.epsilon(0.0) == 1.0
    assert np.isclose(agent.epsilon(0.25), 0.55)
    assert np.isclose(agent.epsilon(0.9), 0.1)


def test_learns_one_step_bandit():
    # Single-state problem, terminal after one step: Q(s, a) -> r(a).
    agent = DQN(1, 3, CFG)
    rewards = [0.0, 1.0, -1.0]
    obs = np.zeros(1, dtype=np.float32)
    for i in range(100):
        a = i % 3
        agent.buffer.add(obs, a, rewards[a], obs, 1.0)
    for _ in range(300):
        agent.train_step()
    assert agent.act(obs) == 1
