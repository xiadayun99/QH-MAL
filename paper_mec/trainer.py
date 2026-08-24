from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

from paper_mec.config import MECConfig, MethodSpec, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.evaluation import aggregate_episode_outcomes
from paper_mec.models import (
    CriticNet,
    DeterministicActor,
    RoleConditionedActor,
    SharedCategoricalActor,
    TwinCritic,
)
from paper_mec.utils import create_progress, format_duration, resolve_torch_device


@dataclass
class PolicyBundle:
    user_actor: SharedCategoricalActor | None
    price_actor: DeterministicActor | None
    alloc_actor: DeterministicActor | None
    role_conditioned_actor: RoleConditionedActor | None
    spec: MethodSpec
    device: torch.device


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        num_users: int,
        user_obs_dim: int,
        num_servers: int,
        price_obs_dim: int,
        alloc_obs_dim: int,
    ) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)

        self.user_obs = np.zeros((capacity, num_users, user_obs_dim), dtype=np.float32)
        self.next_user_obs = np.zeros((capacity, num_users, user_obs_dim), dtype=np.float32)

        self.price_obs = np.zeros((capacity, num_servers, price_obs_dim), dtype=np.float32)
        self.next_price_obs = np.zeros((capacity, num_servers, price_obs_dim), dtype=np.float32)

        self.alloc_obs = np.zeros((capacity, num_servers, alloc_obs_dim), dtype=np.float32)
        self.next_alloc_obs = np.zeros((capacity, num_servers, alloc_obs_dim), dtype=np.float32)

        self.user_actions = np.zeros((capacity, num_users), dtype=np.int64)
        self.executed_user_actions = np.zeros((capacity, num_users), dtype=np.int64)
        self.price_frac = np.zeros((capacity, num_servers), dtype=np.float32)
        # Two fixed allocation controls per server: aggregate CPU utilization
        # and the temperature that shapes the per-user split.
        self.alloc_frac = np.zeros((capacity, num_servers, 2), dtype=np.float32)

        self.user_reward = np.zeros(capacity, dtype=np.float32)
        self.constraint_cost = np.zeros(capacity, dtype=np.float32)
        self.server_reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, item: dict[str, Any]) -> None:
        idx = self.ptr
        self.state[idx] = item["state"]
        self.next_state[idx] = item["next_state"]
        self.user_obs[idx] = item["user_obs"]
        self.next_user_obs[idx] = item["next_user_obs"]
        self.price_obs[idx] = item["price_obs"]
        self.next_price_obs[idx] = item["next_price_obs"]
        self.alloc_obs[idx] = item["alloc_obs"]
        self.next_alloc_obs[idx] = item["next_alloc_obs"]
        self.user_actions[idx] = item["user_actions"]
        self.executed_user_actions[idx] = item["executed_user_actions"]
        self.price_frac[idx] = item["price_frac"]
        self.alloc_frac[idx] = item["alloc_frac"]
        self.user_reward[idx] = item["user_reward"]
        self.constraint_cost[idx] = item["constraint_cost"]
        self.server_reward[idx] = item["server_reward"]
        self.done[idx] = item["done"]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": self.state[idx],
            "next_state": self.next_state[idx],
            "user_obs": self.user_obs[idx],
            "next_user_obs": self.next_user_obs[idx],
            "price_obs": self.price_obs[idx],
            "next_price_obs": self.next_price_obs[idx],
            "alloc_obs": self.alloc_obs[idx],
            "next_alloc_obs": self.next_alloc_obs[idx],
            "user_actions": self.user_actions[idx],
            "executed_user_actions": self.executed_user_actions[idx],
            "price_frac": self.price_frac[idx],
            "alloc_frac": self.alloc_frac[idx],
            "user_reward": self.user_reward[idx],
            "constraint_cost": self.constraint_cost[idx],
            "server_reward": self.server_reward[idx],
            "done": self.done[idx],
        }

    def __len__(self) -> int:
        return self.size


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


class HeteroOffPolicyTrainer:
    def __init__(self, env: MECEnvironment, train_cfg: TrainConfig, spec: MethodSpec) -> None:
        self.env = env
        self.cfg: MECConfig = env.cfg
        self.train_cfg = train_cfg
        self.spec = spec
        if spec.learner not in {
            "qhmarl",
            "maddpg",
            "matd3",
        }:
            raise ValueError(f"Unknown learner '{spec.learner}'.")
        self.uses_twin_critics = spec.learner != "maddpg"
        self.device, self.device_label = resolve_torch_device(train_cfg.device)
        self.num_users = self.cfg.num_users
        self.num_servers = self.cfg.num_servers
        self.action_dim = self.num_servers + 1
        self.total_steps = 0
        env.reset()
        init_price_obs = env.get_price_observations(spec)
        init_prices = self._price_from_fraction(self._reference_price_fraction(torch.tensor(init_price_obs, dtype=torch.float32)))
        init_user_obs = env.get_user_observations(init_prices, spec)
        init_actions = self._heuristic_user_actions(env, init_prices)
        init_alloc_obs = env.get_allocation_observations(init_actions, init_prices, spec)
        init_state = env.get_global_state(init_prices, spec)

        state_dim = init_state.shape[0]
        user_obs_dim = init_user_obs.shape[1]
        price_obs_dim = init_price_obs.shape[1]
        alloc_obs_dim = init_alloc_obs.shape[1]
        # The centralized critics always receive a fixed-width joint action:
        # one S+1 categorical block per user and three continuous controls per
        # server (price, utilization, and allocation shape).  The number of
        # users associated with a server never changes this neural interface.
        self.user_joint_action_dim = self.num_users * self.action_dim
        self.server_joint_action_dim = 3 * self.num_servers
        self.joint_action_dim = self.user_joint_action_dim + self.server_joint_action_dim

        if spec.role_specific_actors:
            user_actor = SharedCategoricalActor(
                user_obs_dim,
                self.action_dim,
                hidden_dims=list(train_cfg.user_actor_hidden_dims),
            ).to(self.device)
            price_actor = DeterministicActor(
                price_obs_dim,
                1,
                hidden_dims=list(train_cfg.server_actor_hidden_dims),
            ).to(self.device)
            alloc_actor = DeterministicActor(
                alloc_obs_dim,
                2,
                hidden_dims=list(train_cfg.server_actor_hidden_dims),
            ).to(self.device)
            role_conditioned_actor = None
        else:
            user_actor = None
            price_actor = None
            alloc_actor = None
            role_conditioned_actor = RoleConditionedActor(
                max(user_obs_dim, price_obs_dim, alloc_obs_dim),
                self.action_dim,
                hidden_dims=list(train_cfg.role_agnostic_actor_hidden_dims),
            ).to(self.device)

        self.bundle = PolicyBundle(
            user_actor=user_actor,
            price_actor=price_actor,
            alloc_actor=alloc_actor,
            role_conditioned_actor=role_conditioned_actor,
            spec=spec,
            device=self.device,
        )
        self.user_target_actor = (
            copy.deepcopy(self.bundle.user_actor).to(self.device)
            if self.bundle.user_actor is not None
            else None
        )
        self.price_target_actor = (
            copy.deepcopy(self.bundle.price_actor).to(self.device)
            if self.bundle.price_actor is not None
            else None
        )
        self.alloc_target_actor = (
            copy.deepcopy(self.bundle.alloc_actor).to(self.device)
            if self.bundle.alloc_actor is not None
            else None
        )
        self.role_conditioned_target_actor = (
            copy.deepcopy(self.bundle.role_conditioned_actor).to(self.device)
            if self.bundle.role_conditioned_actor is not None
            else None
        )

        critic_cls = TwinCritic if self.uses_twin_critics else CriticNet
        self.user_critic = critic_cls(state_dim, self.joint_action_dim, hidden_dims=list(train_cfg.critic_hidden_dims)).to(self.device)
        self.server_critic = critic_cls(state_dim, self.joint_action_dim, hidden_dims=list(train_cfg.critic_hidden_dims)).to(self.device)
        self.user_target_critic = copy.deepcopy(self.user_critic).to(self.device)
        self.server_target_critic = copy.deepcopy(self.server_critic).to(self.device)

        self.server_actor_lr = (
            train_cfg.user_actor_lr
            if spec.equal_actor_learning_rates
            else train_cfg.server_actor_lr
        )
        if spec.role_specific_actors:
            assert self.bundle.user_actor is not None
            assert self.bundle.price_actor is not None
            assert self.bundle.alloc_actor is not None
            self.user_actor_opt = Adam(self.bundle.user_actor.parameters(), lr=train_cfg.user_actor_lr)
            self.price_actor_opt = Adam(self.bundle.price_actor.parameters(), lr=self.server_actor_lr)
            self.alloc_actor_opt = Adam(self.bundle.alloc_actor.parameters(), lr=self.server_actor_lr)
            self.role_conditioned_actor_opt = None
        else:
            assert self.bundle.role_conditioned_actor is not None
            self.user_actor_opt = None
            self.price_actor_opt = None
            self.alloc_actor_opt = None
            self.role_conditioned_actor_opt = Adam(
                self.bundle.role_conditioned_actor.parameters(),
                lr=train_cfg.user_actor_lr,
            )
        self.user_critic_opt = Adam(self.user_critic.parameters(), lr=train_cfg.critic_lr, weight_decay=train_cfg.critic_l2)
        self.server_critic_opt = Adam(self.server_critic.parameters(), lr=train_cfg.critic_lr, weight_decay=train_cfg.critic_l2)

        self.replay = ReplayBuffer(
            train_cfg.replay_buffer_size,
            state_dim,
            self.num_users,
            user_obs_dim,
            self.num_servers,
            price_obs_dim,
            alloc_obs_dim,
        )
        self.best_model_state: dict[str, Any] | None = None
        self.best_validation_score = -float("inf")
        self.best_validation_metrics: dict[str, float] | None = None
        self.best_episode = 0
        self.training_time_s = 0.0

    def _episode_arrival_scale(self, episode_idx: int) -> float:
        scales = self.train_cfg.train_arrival_scales or (self.cfg.arrival_scale,)
        scales = tuple(float(scale) for scale in scales)
        return scales[episode_idx % len(scales)]

    def _selection_scales(self) -> tuple[float, ...]:
        scales = self.train_cfg.selection_arrival_scales or (self.cfg.arrival_scale,)
        return tuple(float(scale) for scale in scales)

    def _snapshot_model_state(self) -> dict[str, Any]:
        modules: dict[str, nn.Module] = {
            "user_critic": self.user_critic,
            "server_critic": self.server_critic,
            "user_target_critic": self.user_target_critic,
            "server_target_critic": self.server_target_critic,
        }
        if self.spec.role_specific_actors:
            assert self.bundle.user_actor is not None
            assert self.bundle.price_actor is not None
            assert self.bundle.alloc_actor is not None
            assert self.user_target_actor is not None
            assert self.price_target_actor is not None
            assert self.alloc_target_actor is not None
            modules.update(
                {
                    "user_actor": self.bundle.user_actor,
                    "price_actor": self.bundle.price_actor,
                    "alloc_actor": self.bundle.alloc_actor,
                    "user_target_actor": self.user_target_actor,
                    "price_target_actor": self.price_target_actor,
                    "alloc_target_actor": self.alloc_target_actor,
                }
            )
        else:
            assert self.bundle.role_conditioned_actor is not None
            assert self.role_conditioned_target_actor is not None
            modules.update(
                {
                    "role_conditioned_actor": self.bundle.role_conditioned_actor,
                    "role_conditioned_target_actor": self.role_conditioned_target_actor,
                }
            )
        return {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}

    def _restore_model_state(self, snapshot: dict[str, Any]) -> None:
        modules: dict[str, nn.Module] = {
            "user_critic": self.user_critic,
            "server_critic": self.server_critic,
            "user_target_critic": self.user_target_critic,
            "server_target_critic": self.server_target_critic,
        }
        if self.spec.role_specific_actors:
            assert self.bundle.user_actor is not None
            assert self.bundle.price_actor is not None
            assert self.bundle.alloc_actor is not None
            assert self.user_target_actor is not None
            assert self.price_target_actor is not None
            assert self.alloc_target_actor is not None
            modules.update(
                {
                    "user_actor": self.bundle.user_actor,
                    "price_actor": self.bundle.price_actor,
                    "alloc_actor": self.bundle.alloc_actor,
                    "user_target_actor": self.user_target_actor,
                    "price_target_actor": self.price_target_actor,
                    "alloc_target_actor": self.alloc_target_actor,
                }
            )
        else:
            assert self.bundle.role_conditioned_actor is not None
            assert self.role_conditioned_target_actor is not None
            modules.update(
                {
                    "role_conditioned_actor": self.bundle.role_conditioned_actor,
                    "role_conditioned_target_actor": self.role_conditioned_target_actor,
                }
            )
        for name, module in modules.items():
            module.load_state_dict(snapshot[name])

    def _selection_score(self, metrics: dict[str, float]) -> float:
        latency_norm = max(self.cfg.latency_max_s, 1.0e-6)
        profit_norm = max(
            self.cfg.num_users * self.cfg.price_max * (self.cfg.cycles_max / 1.0e9) / max(self.cfg.num_servers, 1),
            1.0,
        )
        backlog_term = np.log1p(metrics["avg_queue_backlog"] / max(self.cfg.queue_capacity_cycles, 1.0))
        wait_term = metrics["avg_waiting_delay"] / latency_norm
        delay_term = metrics["avg_delay"] / latency_norm
        p95_term = metrics["p95_delay"] / latency_norm
        return float(
            0.90 * (metrics["avg_profit"] / profit_norm)
            + 0.10 * metrics["fairness"]
            - 1.00 * delay_term
            - 1.35 * p95_term
            - 1.55 * metrics["violation_ratio"]
            - 1.15 * backlog_term
            - 0.90 * wait_term
        )

    def _run_robust_validation(self) -> dict[str, float]:
        rows: list[dict[str, float]] = []
        scales = self._selection_scales()
        validation_bar = create_progress(
            total=len(scales) * self.train_cfg.selection_eval_episodes,
            desc=f"{self.spec.name} validation",
            disable=not self.train_cfg.show_progress,
            leave=False,
            refresh_interval_s=self.train_cfg.progress_refresh_seconds,
        )
        try:
            for scale in scales:
                validation_bar.set_postfix({"arrival": f"{scale:.1f}"}, refresh=False)
                env = MECEnvironment(replace(self.cfg, arrival_scale=scale))
                metrics = evaluate_policy(self, env, self.train_cfg.selection_eval_episodes, progress_bar=validation_bar)
                rows.append({**metrics, "arrival_scale": scale})
        finally:
            validation_bar.close()
        aggregate = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0].keys()
            if key != "arrival_scale"
        }
        aggregate["selection_p95_max"] = float(max(row["p95_delay"] for row in rows))
        aggregate["selection_violation_max"] = float(max(row["violation_ratio"] for row in rows))
        aggregate["selection_backlog_max"] = float(max(row["avg_queue_backlog"] for row in rows))
        latency_norm = max(self.cfg.latency_max_s, 1.0e-6)
        worst_case_penalty = (
            0.35 * (aggregate["selection_p95_max"] / latency_norm)
            + 0.35 * aggregate["selection_violation_max"]
            + 0.25 * np.log1p(aggregate["selection_backlog_max"] / max(self.cfg.queue_capacity_cycles, 1.0))
        )
        aggregate["selection_score"] = float(np.mean([self._selection_score(row) for row in rows]) - worst_case_penalty)
        return aggregate

    def _price_from_fraction(self, frac: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self.cfg.price_min + frac * (self.cfg.price_max - self.cfg.price_min)

    def _controls_from_fraction(
        self,
        frac: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        """Decode normalized actor outputs as utilization/shape controls."""
        utilization = self.cfg.cpu_utilization_min + frac[..., 0] * (
            1.0 - self.cfg.cpu_utilization_min
        )
        temperature = frac[..., 1] * self.cfg.alloc_temp_max
        if isinstance(frac, torch.Tensor):
            return torch.stack([utilization, temperature], dim=-1)
        return np.stack([utilization, temperature], axis=-1)

    def _reference_price_fraction(self, price_obs: torch.Tensor) -> torch.Tensor:
        if self.spec.queue_visible:
            return torch.clamp(0.62 + 0.05 * price_obs[..., 0] + 0.015 * price_obs[..., 1], 0.56, 0.72)
        return torch.clamp(0.42 + 0.04 * price_obs[..., 1], 0.35, 0.52)

    def _reference_alloc_fraction(self, alloc_obs: torch.Tensor) -> torch.Tensor:
        if self.spec.queue_visible:
            utilization = torch.clamp(
                0.75 + 0.15 * alloc_obs[..., 0] + 0.08 * alloc_obs[..., 1],
                0.65,
                0.99,
            )
            temperature = torch.clamp(
                0.014 + 0.005 * alloc_obs[..., 5] + 0.002 * alloc_obs[..., 6],
                0.010,
                0.028,
            )
        else:
            utilization = torch.clamp(
                0.72 + 0.08 * alloc_obs[..., 1],
                0.65,
                0.90,
            )
            temperature = torch.clamp(
                0.020 + 0.006 * alloc_obs[..., 5],
                0.016,
                0.035,
            )
        return torch.stack([utilization, temperature], dim=-1)

    def _allocation_residual_scales(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            [
                self.train_cfg.server_util_residual_scale,
                self.train_cfg.server_alloc_residual_scale,
            ],
            dtype=reference.dtype,
            device=reference.device,
        )

    def _parse_user_obs(self, user_obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s = self.num_servers
        data_mb = user_obs[..., 0] * self.cfg.data_mb_max
        cycles = user_obs[..., 1] * self.cfg.cycles_max
        latency_tol = user_obs[..., 2] * self.cfg.latency_max_s
        local_cpu = user_obs[..., 3] * self.cfg.local_cpu_max_hz
        uplink = user_obs[..., 4 : 4 + s] * self.cfg.uplink_max_mbps
        downlink = user_obs[..., 4 + s : 4 + 2 * s] * self.cfg.downlink_max_mbps
        price_ratio = user_obs[..., 4 + 2 * s : 4 + 3 * s]
        queue_ratio = user_obs[..., 4 + 3 * s : 4 + 4 * s]
        candidates = user_obs[..., 4 + 4 * s : 4 + 5 * s]
        return data_mb, cycles, latency_tol, local_cpu, uplink, downlink, price_ratio, queue_ratio, candidates

    def _prior_user_logits(self, user_obs: torch.Tensor) -> torch.Tensor:
        if user_obs.dim() == 2:
            user_obs = user_obs.unsqueeze(0)
        data_mb, cycles, latency_tol, local_cpu, uplink, downlink, price_ratio, queue_ratio, candidates = self._parse_user_obs(user_obs)
        eps = 1.0e-6
        local_delay = cycles / torch.clamp(local_cpu, min=1.0)
        local_energy = self.cfg.energy_coeff * torch.square(local_cpu) * cycles
        local_risk = torch.square(torch.clamp(local_delay / torch.clamp(latency_tol, min=eps) - self.cfg.risk_trigger_ratio, min=0.0))
        local_cost = self.cfg.alpha_delay * local_delay + self.cfg.beta_energy * local_energy + self.cfg.violation_penalty * local_risk

        tx_u = data_mb.unsqueeze(-1) * 8.0 / torch.clamp(uplink, min=eps)
        tx_d = self.cfg.output_ratio * data_mb.unsqueeze(-1) * 8.0 / torch.clamp(downlink, min=eps)
        wait = queue_ratio * (self.cfg.queue_capacity_cycles / self.cfg.server_cpu_hz) if self.spec.queue_visible else 0.0
        exec_delay = cycles.unsqueeze(-1) / (0.55 * self.cfg.server_cpu_hz)
        perceived_delay = tx_u + wait + exec_delay + tx_d
        price = self.cfg.price_min + price_ratio * (self.cfg.price_max - self.cfg.price_min)
        payment = price * (cycles.unsqueeze(-1) / 1.0e9)
        off_energy = self.cfg.tx_power_w * (tx_u + tx_d)
        off_risk = torch.square(torch.clamp(perceived_delay / torch.clamp(latency_tol.unsqueeze(-1), min=eps) - self.cfg.risk_trigger_ratio, min=0.0))
        off_cost = self.cfg.alpha_delay * perceived_delay + self.cfg.beta_energy * off_energy + self.cfg.eta_payment * payment + self.cfg.violation_penalty * off_risk

        local_cost = local_cost.unsqueeze(-1)
        costs = torch.cat([local_cost, off_cost], dim=-1)
        logits = -costs / 0.25

        mask = torch.cat([torch.ones_like(candidates[..., :1]), candidates], dim=-1)
        logits = logits.masked_fill(mask < 0.5, -1.0e9)
        return logits

    def _heuristic_user_actions(self, env: MECEnvironment, prices: np.ndarray) -> np.ndarray:
        slot = env.current_slot
        actions = np.zeros(self.num_users, dtype=np.int64)
        counts = np.zeros(self.num_servers, dtype=np.float64)
        order = np.arange(self.num_users)
        env.rng.shuffle(order)
        for user in order:
            local_delay = slot.cycles[user] / env.local_cpu_hz[user]
            local_energy = self.cfg.energy_coeff * (env.local_cpu_hz[user] ** 2) * slot.cycles[user]
            latency_tol = slot.latency_tol_s[user]
            local_risk = max(local_delay / max(latency_tol, 1.0e-6) - self.cfg.risk_trigger_ratio, 0.0) ** 2
            best_cost = self.cfg.alpha_delay * local_delay + self.cfg.beta_energy * local_energy + self.cfg.violation_penalty * local_risk
            best_action = 0
            for server in np.where(env.candidate_mask[user] > 0.5)[0]:
                uplink = slot.data_mb[user] * 8.0 / slot.uplink_mbps[user, server]
                downlink = self.cfg.output_ratio * slot.data_mb[user] * 8.0 / slot.downlink_mbps[user, server]
                # Warm-up behavior obeys the same information boundary as the
                # actor. Queue-induced delay remains in the environment reward
                # when ``queue_in_delay`` is true, but a queue-hidden ablation
                # must not leak the backlog through its heuristic behavior.
                wait = env.queues[server] / self.cfg.server_cpu_hz if self.spec.queue_visible else 0.0
                exec_delay = slot.cycles[user] / (self.cfg.server_cpu_hz / max(counts[server] + 1.0, 1.0))
                perceived_delay = uplink + wait + exec_delay + downlink
                off_energy = self.cfg.tx_power_w * (uplink + downlink)
                payment = prices[server] * (slot.cycles[user] / 1.0e9)
                off_risk = max(perceived_delay / max(latency_tol, 1.0e-6) - self.cfg.risk_trigger_ratio, 0.0) ** 2
                off_cost = self.cfg.alpha_delay * perceived_delay + self.cfg.beta_energy * off_energy + self.cfg.eta_payment * payment + self.cfg.violation_penalty * off_risk
                if off_cost < best_cost:
                    best_cost = off_cost
                    best_action = server + 1
            actions[user] = best_action
            if best_action > 0:
                counts[best_action - 1] += 1.0
        return actions

    def _epsilon(self) -> float:
        progress = min(self.total_steps / max(self.train_cfg.epsilon_decay_steps, 1), 1.0)
        return self.train_cfg.epsilon_start + progress * (self.train_cfg.epsilon_end - self.train_cfg.epsilon_start)

    def _user_actor_output(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        if self.spec.role_specific_actors:
            actor = self.user_target_actor if target else self.bundle.user_actor
            assert actor is not None
            return actor(obs)
        actor = self.role_conditioned_target_actor if target else self.bundle.role_conditioned_actor
        assert actor is not None
        return actor.user_logits(obs)

    def _price_actor_output(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        if self.spec.role_specific_actors:
            actor = self.price_target_actor if target else self.bundle.price_actor
            assert actor is not None
            return actor(obs).squeeze(-1)
        actor = self.role_conditioned_target_actor if target else self.bundle.role_conditioned_actor
        assert actor is not None
        return actor.price_residual(obs)

    def _allocation_actor_output(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        if self.spec.role_specific_actors:
            actor = self.alloc_target_actor if target else self.bundle.alloc_actor
            assert actor is not None
            return actor(obs)
        actor = self.role_conditioned_target_actor if target else self.bundle.role_conditioned_actor
        assert actor is not None
        return actor.allocation_residual(obs)

    def actor_parameter_count(self) -> int:
        """Return the parameters used by the active actor design."""
        if self.spec.role_specific_actors:
            actors = (self.bundle.user_actor, self.bundle.price_actor, self.bundle.alloc_actor)
            assert all(actor is not None for actor in actors)
            return int(sum(parameter.numel() for actor in actors for parameter in actor.parameters()))
        assert self.bundle.role_conditioned_actor is not None
        return int(sum(parameter.numel() for parameter in self.bundle.role_conditioned_actor.parameters()))

    def _select_price_fraction(self, price_obs_np: np.ndarray, explore: bool, target: bool = False) -> np.ndarray:
        obs_t = torch.tensor(price_obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            ref = self._reference_price_fraction(obs_t)
            residual = self._price_actor_output(obs_t, target=target) * self.train_cfg.server_price_residual_scale
            frac = torch.clamp(ref + residual, 0.01, 0.99)
            if explore:
                frac = torch.clamp(frac + torch.randn_like(frac) * self.train_cfg.gaussian_noise_std, 0.01, 0.99)
        return frac.cpu().numpy()

    def _select_user_actions(
        self,
        user_obs_np: np.ndarray,
        explore: bool,
        env: MECEnvironment | None = None,
        prices: np.ndarray | None = None,
        use_heuristic_decode: bool = False,
    ) -> np.ndarray:
        if use_heuristic_decode:
            if env is None or prices is None:
                raise ValueError("Heuristic user decoding requires both env and prices.")
            return self._heuristic_user_actions(env, prices)
        obs_t = torch.tensor(user_obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self._prior_user_logits(obs_t).squeeze(0) + self.train_cfg.user_residual_scale * self._user_actor_output(obs_t)
        actions = []
        epsilon = self._epsilon() if explore else 0.0
        candidates = user_obs_np[:, -self.num_servers :]
        for user in range(self.num_users):
            valid = [0] + [server + 1 for server in range(self.num_servers) if candidates[user, server] > 0.5]
            if explore and np.random.rand() < epsilon:
                actions.append(int(np.random.choice(valid)))
                continue
            dist = Categorical(logits=logits[user])
            action = int(dist.sample().item()) if explore else int(torch.argmax(logits[user]).item())
            if action not in valid:
                action = 0
            actions.append(action)
        return np.asarray(actions, dtype=np.int64)

    def _select_alloc_fraction(self, alloc_obs_np: np.ndarray, explore: bool, target: bool = False) -> np.ndarray:
        obs_t = torch.tensor(alloc_obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            ref = self._reference_alloc_fraction(obs_t)
            residual = self._allocation_actor_output(
                obs_t,
                target=target,
            ) * self._allocation_residual_scales(ref)
            frac = torch.clamp(ref + residual, 0.01, 0.99)
            if explore:
                frac = torch.clamp(frac + torch.randn_like(frac) * self.train_cfg.gaussian_noise_std, 0.01, 0.99)
        return frac.cpu().numpy()

    def _compose_joint_action(self, user_actions: torch.Tensor, price_frac: torch.Tensor, alloc_frac: torch.Tensor) -> torch.Tensor:
        if user_actions.dim() == 2:
            user_onehot = F.one_hot(user_actions.long(), num_classes=self.action_dim).float()
        else:
            user_onehot = user_actions.float()
        return torch.cat(
            [
                user_onehot.reshape(user_onehot.shape[0], -1),
                price_frac.reshape(price_frac.shape[0], -1),
                alloc_frac.reshape(alloc_frac.shape[0], -1),
            ],
            dim=-1,
        )

    def _straight_through_user_actions(self, logits: torch.Tensor) -> torch.Tensor:
        """Return hard categorical actions with a soft pathwise gradient.

        Replay critics are fitted on realized one-hot tentative requests.  The
        straight-through Gumbel--Softmax estimator therefore keeps the forward
        input on that same one-hot support while using the soft relaxation only
        in the backward pass to update the shared categorical actor.
        """
        return F.gumbel_softmax(
            logits,
            tau=self.train_cfg.user_gumbel_tau,
            hard=True,
            dim=-1,
        )

    def _preview_next_policy_obs(self, env: MECEnvironment, target: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        next_price_obs = env.get_price_observations(self.spec)
        next_price_frac = self._select_price_fraction(next_price_obs, explore=False, target=target)
        next_prices = self._price_from_fraction(next_price_frac)
        next_user_obs = env.get_user_observations(next_prices, self.spec)
        next_user_actions = self._select_user_actions(next_user_obs, explore=False)
        next_alloc_obs = env.get_allocation_observations(next_user_actions, next_prices, self.spec)
        return next_price_obs, next_user_obs, next_alloc_obs, next_user_actions

    def _sample_behavior_action(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        price_obs = self.env.get_price_observations(self.spec)
        if self.total_steps < self.train_cfg.warmup_steps:
            price_frac = self._reference_price_fraction(torch.tensor(price_obs, dtype=torch.float32)).cpu().numpy()
        else:
            price_frac = self._select_price_fraction(price_obs, explore=True)
        prices = self._price_from_fraction(price_frac)

        user_obs = self.env.get_user_observations(prices, self.spec)
        if self.total_steps < self.train_cfg.warmup_steps:
            user_actions = self._heuristic_user_actions(self.env, prices)
        else:
            user_actions = self._select_user_actions(user_obs, explore=True, env=self.env, prices=prices)

        alloc_obs = self.env.get_allocation_observations(user_actions, prices, self.spec)
        if self.total_steps < self.train_cfg.warmup_steps:
            alloc_frac = self._reference_alloc_fraction(torch.tensor(alloc_obs, dtype=torch.float32)).cpu().numpy()
        else:
            alloc_frac = self._select_alloc_fraction(alloc_obs, explore=True)
        return price_obs, user_obs, alloc_obs, user_actions, price_frac, alloc_frac

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)

    def _target_critic_value(self, critic: nn.Module, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        values = critic(state, joint_action)
        if self.uses_twin_critics:
            q1, q2 = values
            return torch.min(q1, q2)
        return values

    def _critic_loss(self, critic: nn.Module, state: torch.Tensor, joint_action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values = critic(state, joint_action)
        if self.uses_twin_critics:
            q1, q2 = values
            return F.mse_loss(q1, target) + F.mse_loss(q2, target)
        return F.mse_loss(values, target)

    def _critic_policy_value(self, critic: nn.Module, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        if self.uses_twin_critics:
            return critic.q1_forward(state, joint_action)
        return critic(state, joint_action)

    def _actor_update_intervals(self) -> tuple[int, int]:
        if self.spec.learner == "maddpg":
            return 1, 1
        if self.spec.learner == "matd3":
            delay = max(int(self.train_cfg.matd3_policy_delay), 1)
            return delay, delay
        if not self.spec.two_timescale:
            return 1, 1
        return (
            max(int(self.train_cfg.user_actor_update_interval), 1),
            max(int(self.train_cfg.server_actor_update_interval), 1),
        )

    def _critic_target_joint(self, next_user_obs: torch.Tensor, next_price_obs: torch.Tensor, next_alloc_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        next_price_ref = self._reference_price_fraction(next_price_obs)
        next_price_frac = torch.clamp(
            next_price_ref
            + self._price_actor_output(next_price_obs, target=True)
            * self.train_cfg.server_price_residual_scale,
            0.01,
            0.99,
        )

        next_prior_logits = self._prior_user_logits(next_user_obs)
        next_logits = (
            next_prior_logits
            + self.train_cfg.user_residual_scale
            * self._user_actor_output(next_user_obs, target=True)
        )
        next_user_actions = self._straight_through_user_actions(next_logits)

        next_alloc_ref = self._reference_alloc_fraction(next_alloc_obs)
        next_alloc_frac = torch.clamp(
            next_alloc_ref
            + self._allocation_actor_output(next_alloc_obs, target=True)
            * self._allocation_residual_scales(next_alloc_ref),
            0.01,
            0.99,
        )
        if self.spec.learner == "matd3":
            noise_std = max(float(self.train_cfg.matd3_target_noise_std), 0.0)
            noise_clip = max(float(self.train_cfg.matd3_target_noise_clip), 0.0)
            price_noise = torch.clamp(torch.randn_like(next_price_frac) * noise_std, -noise_clip, noise_clip)
            alloc_noise = torch.clamp(torch.randn_like(next_alloc_frac) * noise_std, -noise_clip, noise_clip)
            next_price_frac = torch.clamp(next_price_frac + price_noise, 0.01, 0.99)
            next_alloc_frac = torch.clamp(next_alloc_frac + alloc_noise, 0.01, 0.99)
        return next_user_actions, next_price_frac, next_alloc_frac

    def update_step(self) -> None:
        if len(self.replay) < self.train_cfg.batch_size:
            return
        batch = self.replay.sample(self.train_cfg.batch_size)
        state = self._tensor(batch["state"])
        next_state = self._tensor(batch["next_state"])
        user_obs = self._tensor(batch["user_obs"])
        next_user_obs = self._tensor(batch["next_user_obs"])
        price_obs = self._tensor(batch["price_obs"])
        next_price_obs = self._tensor(batch["next_price_obs"])
        alloc_obs = self._tensor(batch["alloc_obs"])
        next_alloc_obs = self._tensor(batch["next_alloc_obs"])
        user_actions = torch.tensor(batch["user_actions"], dtype=torch.long, device=self.device)
        price_frac = self._tensor(batch["price_frac"])
        alloc_frac = self._tensor(batch["alloc_frac"])
        user_reward = self._tensor(batch["user_reward"])
        server_reward = self._tensor(batch["server_reward"])
        done = self._tensor(batch["done"])

        current_joint = self._compose_joint_action(user_actions, price_frac, alloc_frac)

        with torch.no_grad():
            next_user_actions, next_price_frac, next_alloc_frac = self._critic_target_joint(next_user_obs, next_price_obs, next_alloc_obs)
            next_joint = self._compose_joint_action(next_user_actions, next_price_frac, next_alloc_frac)
            target_user_q = user_reward + self.train_cfg.gamma * (1.0 - done) * self._target_critic_value(self.user_target_critic, next_state, next_joint)
            target_server_q = server_reward + self.train_cfg.gamma * (1.0 - done) * self._target_critic_value(self.server_target_critic, next_state, next_joint)

        if self.total_steps % self.train_cfg.critic_update_interval == 0:
            user_critic_loss = self._critic_loss(self.user_critic, state, current_joint, target_user_q)
            server_critic_loss = self._critic_loss(self.server_critic, state, current_joint, target_server_q)

            self.user_critic_opt.zero_grad()
            user_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.user_critic.parameters(), self.train_cfg.grad_clip_norm)
            self.user_critic_opt.step()

            self.server_critic_opt.zero_grad()
            server_critic_loss.backward()
            nn.utils.clip_grad_norm_(self.server_critic.parameters(), self.train_cfg.grad_clip_norm)
            self.server_critic_opt.step()

        user_actor_interval, server_actor_interval = self._actor_update_intervals()
        user_actor_updated = self.total_steps % user_actor_interval == 0
        server_actor_updated = self.total_steps % server_actor_interval == 0

        if user_actor_updated:
            user_residual = self._user_actor_output(user_obs)
            user_logits = self._prior_user_logits(user_obs) + self.train_cfg.user_residual_scale * user_residual
            user_st = self._straight_through_user_actions(user_logits)
            # Coordinate update: only the categorical block is differentiable;
            # replayed server controls are held fixed for the user-role step.
            current_user_joint = self._compose_joint_action(user_st, price_frac.detach(), alloc_frac.detach())
            user_actor_loss = -self._critic_policy_value(self.user_critic, state, current_user_joint).mean()
            user_actor_loss += self.train_cfg.actor_l2 * torch.square(user_residual).mean()
            if self.spec.role_specific_actors:
                assert self.user_actor_opt is not None
                assert self.bundle.user_actor is not None
                self.user_actor_opt.zero_grad()
                user_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.bundle.user_actor.parameters(), self.train_cfg.grad_clip_norm)
                self.user_actor_opt.step()
            else:
                assert self.role_conditioned_actor_opt is not None
                assert self.bundle.role_conditioned_actor is not None
                for group in self.role_conditioned_actor_opt.param_groups:
                    group["lr"] = self.train_cfg.user_actor_lr
                self.role_conditioned_actor_opt.zero_grad()
                user_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.bundle.role_conditioned_actor.parameters(),
                    self.train_cfg.grad_clip_norm,
                )
                self.role_conditioned_actor_opt.step()

        if server_actor_updated:
            # The allocation observation in this replay row was constructed
            # from ``user_actions``.  Reusing the same hard one-hot requests
            # keeps that observation/action pair coherent and means the server
            # gradient never differentiates through categorical user choices.
            replay_user_onehot = F.one_hot(
                user_actions,
                num_classes=self.action_dim,
            ).to(dtype=state.dtype)

            current_price_ref = self._reference_price_fraction(price_obs)
            current_price_resid = self._price_actor_output(price_obs)
            current_price_frac = torch.clamp(current_price_ref + current_price_resid * self.train_cfg.server_price_residual_scale, 0.01, 0.99)

            current_alloc_ref = self._reference_alloc_fraction(alloc_obs)
            current_alloc_resid = self._allocation_actor_output(alloc_obs)
            current_alloc_frac = torch.clamp(
                current_alloc_ref
                + current_alloc_resid
                * self._allocation_residual_scales(current_alloc_ref),
                0.01,
                0.99,
            )

            current_server_joint = self._compose_joint_action(
                replay_user_onehot.detach(),
                current_price_frac,
                current_alloc_frac,
            )
            server_actor_loss = -self._critic_policy_value(self.server_critic, state, current_server_joint).mean()
            server_actor_loss += self.train_cfg.actor_l2 * (torch.square(current_price_resid).mean() + torch.square(current_alloc_resid).mean())

            if self.spec.role_specific_actors:
                assert self.price_actor_opt is not None
                assert self.alloc_actor_opt is not None
                assert self.bundle.price_actor is not None
                assert self.bundle.alloc_actor is not None
                self.price_actor_opt.zero_grad()
                self.alloc_actor_opt.zero_grad()
                server_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.bundle.price_actor.parameters(), self.train_cfg.grad_clip_norm)
                nn.utils.clip_grad_norm_(self.bundle.alloc_actor.parameters(), self.train_cfg.grad_clip_norm)
                self.price_actor_opt.step()
                self.alloc_actor_opt.step()
            else:
                assert self.role_conditioned_actor_opt is not None
                assert self.bundle.role_conditioned_actor is not None
                for group in self.role_conditioned_actor_opt.param_groups:
                    group["lr"] = self.server_actor_lr
                self.role_conditioned_actor_opt.zero_grad()
                server_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.bundle.role_conditioned_actor.parameters(),
                    self.train_cfg.grad_clip_norm,
                )
                self.role_conditioned_actor_opt.step()

        # MATD3 delays both actor and target updates; MADDPG and QH-MAL retain
        # their usual per-critic-step target updates.
        update_targets = self.spec.learner != "matd3" or (user_actor_updated and server_actor_updated)
        if update_targets:
            if self.spec.role_specific_actors:
                assert self.user_target_actor is not None
                assert self.price_target_actor is not None
                assert self.alloc_target_actor is not None
                assert self.bundle.user_actor is not None
                assert self.bundle.price_actor is not None
                assert self.bundle.alloc_actor is not None
                soft_update(self.user_target_actor, self.bundle.user_actor, self.train_cfg.target_tau)
                soft_update(self.price_target_actor, self.bundle.price_actor, self.train_cfg.target_tau)
                soft_update(self.alloc_target_actor, self.bundle.alloc_actor, self.train_cfg.target_tau)
            else:
                assert self.role_conditioned_target_actor is not None
                assert self.bundle.role_conditioned_actor is not None
                soft_update(
                    self.role_conditioned_target_actor,
                    self.bundle.role_conditioned_actor,
                    self.train_cfg.target_tau,
                )
            soft_update(self.user_target_critic, self.user_critic, self.train_cfg.target_tau)
            soft_update(self.server_target_critic, self.server_critic, self.train_cfg.target_tau)

    def train(self) -> tuple[PolicyBundle, list[dict[str, float]]]:
        history: list[dict[str, float]] = []
        selection_interval = max(int(self.train_cfg.selection_interval), 1)
        train_start_time = time.perf_counter()
        progress_bar = create_progress(
            total=self.train_cfg.episodes,
            desc=self.spec.name,
            disable=not self.train_cfg.show_progress,
            leave=True,
            refresh_interval_s=self.train_cfg.progress_refresh_seconds,
        )
        try:
            for episode_idx in range(self.train_cfg.episodes):
                episode_scale = self._episode_arrival_scale(episode_idx)
                self.env = MECEnvironment(replace(self.cfg, arrival_scale=episode_scale))
                self.env.reset()
                episode_outcomes: list[dict[str, Any]] = []
                for step in range(self.train_cfg.episode_length):
                    self.total_steps += 1
                    state = self.env.get_global_state(self._price_from_fraction(np.full(self.num_servers, 0.5)), self.spec)
                    price_obs, user_obs, alloc_obs, user_actions, price_frac, alloc_frac = self._sample_behavior_action()
                    prices = self._price_from_fraction(price_frac)
                    allocation_controls = self._controls_from_fraction(alloc_frac)
                    state = self.env.get_global_state(prices, self.spec)
                    outcome = self.env.step(
                        user_actions,
                        prices,
                        allocation_controls,
                        self.spec,
                    )
                    done = float(step == self.train_cfg.episode_length - 1)
                    next_price_obs, next_user_obs, next_alloc_obs, _ = self._preview_next_policy_obs(self.env)
                    next_state = self.env.get_global_state(self._price_from_fraction(self._select_price_fraction(next_price_obs, explore=False)), self.spec)

                    self.replay.add(
                        {
                            "state": state,
                            "next_state": next_state,
                            "user_obs": user_obs,
                            "next_user_obs": next_user_obs,
                            "price_obs": price_obs,
                            "next_price_obs": next_price_obs,
                            "alloc_obs": alloc_obs,
                            "next_alloc_obs": next_alloc_obs,
                            "user_actions": user_actions,
                            "executed_user_actions": outcome["executed_actions"],
                            "price_frac": price_frac,
                            "alloc_frac": alloc_frac,
                            "user_reward": outcome["user_reward_mean"],
                            "constraint_cost": outcome["constraint_cost_mean"],
                            "server_reward": outcome["server_reward_mean"],
                            "done": done,
                        }
                    )
                    episode_outcomes.append(outcome)

                    if self.total_steps >= self.train_cfg.warmup_steps:
                        for _ in range(self.train_cfg.updates_per_step):
                            self.update_step()

                summary = aggregate_episode_outcomes(episode_outcomes)
                summary["train_arrival_scale"] = float(episode_scale)
                if (episode_idx + 1) % self.train_cfg.curve_eval_interval == 0:
                    eval_env = MECEnvironment(replace(self.cfg, arrival_scale=self.train_cfg.curve_eval_arrival_scale))
                    eval_metrics = evaluate_policy(self, eval_env, self.train_cfg.curve_eval_episodes)
                    summary.update({f"eval_{key}": value for key, value in eval_metrics.items()})
                else:
                    summary.update({f"eval_{key}": float("nan") for key in summary.keys()})

                should_select = ((episode_idx + 1) % selection_interval == 0) or (episode_idx + 1 == self.train_cfg.episodes)
                summary["selected_checkpoint"] = 0.0
                if should_select:
                    progress_bar.write(f"{self.spec.name}: robust validation at episode {episode_idx + 1}/{self.train_cfg.episodes}")
                    robust_metrics = self._run_robust_validation()
                    summary.update({f"select_{key}": value for key, value in robust_metrics.items()})
                    if robust_metrics["selection_score"] > self.best_validation_score:
                        self.best_validation_score = robust_metrics["selection_score"]
                        self.best_validation_metrics = robust_metrics
                        self.best_model_state = self._snapshot_model_state()
                        self.best_episode = episode_idx + 1
                        summary["selected_checkpoint"] = 1.0
                else:
                    summary["select_selection_score"] = float("nan")

                summary["episode"] = float(episode_idx + 1)
                history.append(summary)
                completed_episodes = episode_idx + 1
                elapsed_seconds = max(time.perf_counter() - train_start_time, 0.0)
                eta_seconds = elapsed_seconds * max(self.train_cfg.episodes - completed_episodes, 0) / max(completed_episodes, 1)
                progress_bar.set_postfix(
                    {
                        "eta": format_duration(eta_seconds),
                        "load": f"{episode_scale:.1f}",
                        "eps": self._epsilon(),
                        "best_ep": self.best_episode if self.best_episode else 0,
                        "best_score": self.best_validation_score if np.isfinite(self.best_validation_score) else float("nan"),
                        "sel": summary.get("select_selection_score", float("nan")),
                        "eval_p95": summary.get("eval_p95_delay", float("nan")),
                        "eval_vio": summary.get("eval_violation_ratio", float("nan")),
                    },
                    refresh=False,
                )
                progress_bar.update(1)
        finally:
            progress_bar.close()

        self.training_time_s = max(time.perf_counter() - train_start_time, 0.0)

        if self.best_model_state is not None:
            self._restore_model_state(self.best_model_state)
        else:
            self.best_model_state = self._snapshot_model_state()
            self.best_episode = self.train_cfg.episodes
        return self.bundle, history

    @torch.no_grad()
    def act_deterministically(self, env: MECEnvironment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        price_obs = env.get_price_observations(self.spec)
        price_frac = self._select_price_fraction(price_obs, explore=False)
        prices = self._price_from_fraction(price_frac)
        user_obs = env.get_user_observations(prices, self.spec)
        user_actions = self._select_user_actions(user_obs, explore=False)
        alloc_obs = env.get_allocation_observations(user_actions, prices, self.spec)
        alloc_frac = self._select_alloc_fraction(alloc_obs, explore=False)
        allocation_controls = self._controls_from_fraction(alloc_frac)
        return user_actions, prices, allocation_controls


def evaluate_policy(
    trainer: HeteroOffPolicyTrainer,
    env: MECEnvironment,
    episodes: int,
    progress_bar: Any | None = None,
) -> dict[str, float]:
    summaries: list[dict[str, float]] = []
    for episode in range(episodes):
        env_eval = env.clone_with_seed(env.cfg.seed + 1000 + episode)
        env_eval.reset()
        episode_outcomes: list[dict[str, Any]] = []
        for _ in range(trainer.train_cfg.episode_length):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            decision_start = time.perf_counter()
            user_actions, prices, allocation_controls = trainer.act_deterministically(env_eval)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            decision_time_ms = 1.0e3 * (time.perf_counter() - decision_start)
            outcome = env_eval.step(
                user_actions,
                prices,
                allocation_controls,
                trainer.spec,
            )
            timed_metrics = dict(outcome["metrics"])
            timed_metrics["decision_time_ms"] = float(decision_time_ms)
            outcome = dict(outcome)
            outcome["metrics"] = timed_metrics
            episode_outcomes.append(outcome)
        summaries.append(aggregate_episode_outcomes(episode_outcomes))
        if progress_bar is not None:
            progress_bar.update(1)
    return {key: float(np.mean([item[key] for item in summaries])) for key in summaries[0].keys()}


PPOTrainer = HeteroOffPolicyTrainer
