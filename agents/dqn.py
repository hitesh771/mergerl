"""Deep Q-Network (Mnih et al. 2015) with the Double-DQN target, from scratch.

The agent learns Q(s, a) for the 7 discrete accelerations of
DiscreteActionWrapper. Training loop pieces:

  * Replay buffer: stores transitions (s, a, r, s', done); minibatches are
    drawn uniformly at random, which breaks the correlation between
    consecutive steps.
  * Target network Q_target: a copy of Q that is only updated every
    `target_update` steps, so the regression target does not move with
    every gradient step.
  * Double-DQN target (van Hasselt et al. 2016): the online net *chooses*
    the next action, the target net *evaluates* it:
        y = r + gamma * (1 - done) * Q_target(s', argmax_a' Q(s', a'))
    This reduces the over-estimation of plain max_a' Q_target(s', a').
  * Huber loss between Q(s, a) and y (quadratic for small errors, linear for
    large ones -> robust to the big terminal rewards).
  * epsilon-greedy exploration with epsilon decaying linearly from
    eps_start to eps_end over the first eps_fraction of training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.networks import QNet


class ReplayBuffer:
    """Fixed-size circular buffer of transitions stored in numpy arrays."""

    def __init__(self, obs_dim: int, size: int, rng: np.random.Generator):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros(size, dtype=np.int64)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.size, self.ptr, self.full = size, 0, False
        self.rng = rng

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, len(self), size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


class DQN:
    """Double DQN agent for a discrete action space."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.q = QNet(obs_dim, n_actions, cfg["hidden"])
        self.q_target = QNet(obs_dim, n_actions, cfg["hidden"])
        self.update_target()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(obs_dim, cfg["buffer_size"], self.rng)

    def epsilon(self, progress: float) -> float:
        """Linearly decayed exploration rate at training progress in [0, 1]."""
        c = self.cfg
        frac = min(progress / c["eps_fraction"], 1.0)
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        """epsilon-greedy action for a single observation."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    def update_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())

    def train_step(self) -> float:
        """One gradient step on a random minibatch; returns the loss."""
        c = self.cfg
        b = self.buffer.sample(c["batch_size"])
        q_sa = self.q(b["obs"]).gather(1, b["act"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.q(b["next_obs"]).argmax(1, keepdim=True)
            next_q = self.q_target(b["next_obs"]).gather(1, next_a).squeeze(1)
            target = b["rew"] + c["gamma"] * (1.0 - b["done"]) * next_q
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c["max_grad_norm"])
        self.opt.step()
        return loss.item()

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"algo": "dqn", "cfg": self.cfg, "n_actions": self.n_actions,
                    "q": self.q.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, obs_dim: int) -> "DQN":
        ck = torch.load(path, weights_only=False)
        cfg = dict(ck["cfg"], buffer_size=1)   # no replay memory needed for eval
        agent = cls(obs_dim, ck["n_actions"], cfg)
        agent.q.load_state_dict(ck["q"])
        agent.update_target()
        return agent
