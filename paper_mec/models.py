from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, final_tanh: bool = False) -> None:
        super().__init__()
        layers = []
        last = in_dim
        for width in hidden_dims:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width
        layers.append(nn.Linear(last, out_dim))
        if final_tanh:
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedCategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.model = MLP(obs_dim, hidden_dims or [128, 128], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.model = MLP(obs_dim, hidden_dims or [96, 96], action_dim, final_tanh=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class RoleConditionedActor(nn.Module):
    """One parameter-shared actor used by the role-agnostic ablation.

    User, pricing, and allocation observations are zero padded to a common
    width and augmented with a three-way role indicator.  Users consume the
    complete output as categorical logits, whereas each continuous server
    role consumes a fixed prefix after a tanh projection: one output for price
    and two outputs for aggregate CPU utilization and allocation shape.  This
    keeps the action domains executable while removing role-specific actor
    parameters.
    """

    USER_ROLE = 0
    PRICE_ROLE = 1
    ALLOCATION_ROLE = 2
    NUM_ROLES = 3

    def __init__(
        self,
        max_obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.max_obs_dim = int(max_obs_dim)
        self.action_dim = int(action_dim)
        self.model = MLP(
            self.max_obs_dim + self.NUM_ROLES,
            hidden_dims or [288, 288, 224],
            self.action_dim,
        )

    def forward(self, obs: torch.Tensor, role: int) -> torch.Tensor:
        if role not in {self.USER_ROLE, self.PRICE_ROLE, self.ALLOCATION_ROLE}:
            raise ValueError(f"Unknown actor role index {role}.")
        if obs.shape[-1] > self.max_obs_dim:
            raise ValueError(
                f"Observation width {obs.shape[-1]} exceeds configured maximum "
                f"{self.max_obs_dim}."
            )
        padded = F.pad(obs, (0, self.max_obs_dim - obs.shape[-1]))
        role_code = torch.zeros(
            (*obs.shape[:-1], self.NUM_ROLES),
            dtype=obs.dtype,
            device=obs.device,
        )
        role_code[..., role] = 1.0
        return self.model(torch.cat([padded, role_code], dim=-1))

    def user_logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs, self.USER_ROLE)

    def price_residual(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.forward(obs, self.PRICE_ROLE)[..., 0])

    def allocation_residual(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.forward(obs, self.ALLOCATION_ROLE)[..., :2])


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.model = MLP(state_dim + joint_action_dim, hidden_dims or [256, 256], 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([state, joint_action], dim=-1)).squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.q1 = CriticNet(state_dim, joint_action_dim, hidden_dims=hidden_dims)
        self.q2 = CriticNet(state_dim, joint_action_dim, hidden_dims=hidden_dims)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, joint_action), self.q2(state, joint_action)

    def q1_forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, joint_action)
