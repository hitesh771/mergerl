"""Proximal Policy Optimisation (PPO, Schulman et al. 2017), written from scratch.

One PPO iteration:
  1. Roll out the current policy for `n_steps` steps in each of `n_envs`
     environments (a tiny vector loop, see VecEnv).
  2. Compute advantages with Generalised Advantage Estimation:
         delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
         A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
     and returns R_t = A_t + V(s_t) as regression targets for the critic.
  3. For `n_epochs` passes over the data in random minibatches minimise
         L = -E[min(rho A, clip(rho, 1-eps, 1+eps) A)]     (clipped surrogate)
             + c_v * E[(V(s) - R)^2]                         (value loss)
             - c_e * E[entropy]                              (exploration)
     with rho = pi_new(a|s) / pi_old(a|s). The clip stops one update from
     moving the policy too far from the policy that collected the data.

Extras: advantage normalisation per minibatch, running observation
normalisation, gradient-norm clipping and a linearly decaying learning rate.
"""
import numpy as np
import torch
import torch.nn as nn

from agents.networks import Critic, GaussianActor


class RunningMeanStd:
    """Running mean / variance of observations (parallel Welford update)."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), d["count"]


class VecEnv:
    """Runs n environments in lock-step and resets each one when it finishes.

    `step` returns the observation of the *new* episode for finished envs,
    and stores the info dict of the finished episode in `infos`.
    """

    def __init__(self, make_env, n: int, seed: int):
        self.envs = [make_env() for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_return = np.zeros(n)

    def step(self, actions: np.ndarray):
        n = len(self.envs)
        rewards, dones, finished = np.zeros(n), np.zeros(n), []
        for i, env in enumerate(self.envs):
            obs, r, term, trunc, info = env.step(actions[i])
            rewards[i] = r
            self.ep_return[i] += r
            if term or trunc:
                dones[i] = 1.0
                finished.append({**info, "return": self.ep_return[i]})
                self.ep_return[i] = 0.0
                obs, _ = env.reset()
            self.obs[i] = obs
        return self.obs.copy(), rewards, dones, finished


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalised Advantage Estimation.

    Args (arrays of shape (T, N) except last_value (N,)):
        rewards[t], values[t] = V(s_t), dones[t] = 1 if the episode ended
        after step t (then s_{t+1} belongs to a new episode and must not be
        bootstrapped), last_value = V(s_T).
    Returns:
        advantages, returns (both (T, N)).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    next_adv = np.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_adv = delta + gamma * lam * not_done * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


class PPO:
    """PPO agent with a Gaussian actor and a separate critic."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict):
        self.cfg = cfg
        self.actor = GaussianActor(obs_dim, act_dim, cfg["hidden"])
        self.critic = Critic(obs_dim, cfg["hidden"])
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg["lr"], eps=1e-5)
        self.params = params
        self.obs_rms = RunningMeanStd(obs_dim)

    # ----------------------------------------------------------- acting
    def _tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action(s) for raw observation(s); the mean action if deterministic."""
        dist = self.actor.dist(self._tensor(obs))
        a = dist.mean if deterministic else dist.sample()
        return a.numpy()

    # ---------------------------------------------------------- training
    def collect(self, venv: VecEnv, n_steps: int):
        """Roll out n_steps in every env. Returns the batch and finished-episode infos."""
        n = len(venv.envs)
        obs_buf = np.zeros((n_steps, n, venv.obs.shape[1]), dtype=np.float32)
        act_buf = np.zeros((n_steps, n, 1), dtype=np.float32)
        logp_buf, rew_buf, done_buf, val_buf = (np.zeros((n_steps, n)) for _ in range(4))
        episodes = []
        obs = venv.obs.copy()
        for t in range(n_steps):
            self.obs_rms.update(obs)
            obs_buf[t] = obs
            with torch.no_grad():
                x = self._tensor(obs)
                dist = self.actor.dist(x)
                a = dist.sample()
                logp_buf[t] = dist.log_prob(a).sum(-1).numpy()
                val_buf[t] = self.critic(x).numpy()
            act_buf[t] = a.numpy()
            obs, rew_buf[t], done_buf[t], finished = venv.step(act_buf[t])
            episodes += finished
        with torch.no_grad():
            last_val = self.critic(self._tensor(obs)).numpy()
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_val,
                               self.cfg["gamma"], self.cfg["lam"])
        batch = {
            # Observations are normalised with the stats at update time.
            "obs": self.obs_rms.normalize(obs_buf.reshape(-1, obs_buf.shape[-1])),
            "act": act_buf.reshape(-1, 1),
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "ret": ret.reshape(-1),
            "val": val_buf.reshape(-1),
        }
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}, episodes

    def update(self, batch: dict, lr: float) -> dict:
        """Several epochs of minibatch PPO updates on one batch."""
        c = self.cfg
        for g in self.opt.param_groups:
            g["lr"] = lr
        size = batch["obs"].shape[0]
        mb = size // c["n_minibatches"]
        stats = []
        for _ in range(c["n_epochs"]):
            perm = torch.randperm(size)
            for start in range(0, size, mb):
                idx = perm[start:start + mb]
                obs, act = batch["obs"][idx], batch["act"][idx]
                adv = batch["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist = self.actor.dist(obs)
                logp = dist.log_prob(act).sum(-1)
                ratio = torch.exp(logp - batch["logp"][idx])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss: the new value prediction may not move
                # further than `clip` away from the old one without penalty.
                v = self.critic(obs)
                v_old, ret = batch["val"][idx], batch["ret"][idx]
                v_clip = v_old + torch.clamp(v - v_old, -c["clip_v"], c["clip_v"])
                value_loss = 0.5 * torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()

                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + c["vf_coef"] * value_loss - c["ent_coef"] * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, c["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                stats.append([policy_loss.item(), value_loss.item(), entropy.item(), approx_kl])
        return dict(zip(["policy_loss", "value_loss", "entropy", "approx_kl"],
                        np.mean(stats, axis=0)))

    # ------------------------------------------------------------ saving
    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save({"algo": "ppo", "cfg": self.cfg,
                    "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "obs_rms": self.obs_rms.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, obs_dim: int, act_dim: int = 1) -> "PPO":
        ck = torch.load(path, weights_only=False)
        agent = cls(obs_dim, act_dim, ck["cfg"])
        agent.actor.load_state_dict(ck["actor"])
        agent.critic.load_state_dict(ck["critic"])
        agent.obs_rms.load_state_dict(ck["obs_rms"])
        return agent
