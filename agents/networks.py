"""Neural networks used by PPO and DQN (plain PyTorch MLPs)."""
import numpy as np
import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden: int = 64, layers: int = 2) -> nn.Sequential:
    """Fully connected network: `layers` hidden layers of size `hidden` with tanh."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def init_layer(layer: nn.Linear, gain: float) -> None:
    """Orthogonal weight init (standard for policy-gradient methods)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


class GaussianActor(nn.Module):
    """Policy pi(a|s) = Normal(mu(s), sigma) with a learned, state-independent sigma.

    The network outputs the mean mu(s); log(sigma) is a free parameter
    vector. Sampled actions are later clipped to [-1, 1] by the env; the
    log-probability is computed for the unclipped sample, which is the
    usual simple choice for PPO.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, init_log_std: float = -0.5):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden)
        for layer in self.mu:
            if isinstance(layer, nn.Linear):
                init_layer(layer, np.sqrt(2))
        # Small last layer -> initial mean close to 0 (no strong bias).
        init_layer(self.mu[-1], 0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mu(obs), self.log_std.exp())


class Critic(nn.Module):
    """State-value function V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden)
        for layer in self.v:
            if isinstance(layer, nn.Linear):
                init_layer(layer, np.sqrt(2))
        init_layer(self.v[-1], 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class QNet(nn.Module):
    """Action-value function Q(s, .) for a discrete action set (one output per action)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(obs)
