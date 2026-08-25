from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from paper_mec.config import MECConfig, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.trainer import HeteroOffPolicyTrainer
from run import build_learning_specs


def _small_train_config(**overrides: object) -> TrainConfig:
    base = TrainConfig(
        episodes=1,
        episode_length=3,
        eval_episodes=1,
        curve_eval_episodes=1,
        curve_eval_interval=10,
        train_arrival_scales=(1.0,),
        selection_arrival_scales=(1.0,),
        selection_eval_episodes=1,
        selection_interval=1,
        user_actor_hidden_dims=(16,),
        server_actor_hidden_dims=(16,),
        critic_hidden_dims=(24,),
        batch_size=2,
        replay_buffer_size=16,
        warmup_steps=0,
        show_progress=False,
        device="cpu",
    )
    return replace(base, **overrides)


def _proposed_spec(cfg: MECConfig):
    return next(
        spec for spec in build_learning_specs(cfg) if spec.name == "Proposed Method"
    )


def test_straight_through_user_actions_are_hard_but_differentiable() -> None:
    torch.manual_seed(2050)
    cfg = MECConfig(num_users=4, num_servers=2, seed=2050)
    trainer = HeteroOffPolicyTrainer(
        MECEnvironment(cfg),
        _small_train_config(),
        _proposed_spec(cfg),
    )
    env = trainer.env
    price_obs = env.get_price_observations(trainer.spec)
    price_frac = trainer._select_price_fraction(price_obs, explore=False)
    prices = trainer._price_from_fraction(price_frac)
    user_obs = torch.tensor(
        env.get_user_observations(prices, trainer.spec),
        dtype=torch.float32,
    )

    residual = trainer._user_actor_output(user_obs)
    logits = trainer._prior_user_logits(user_obs) + trainer.train_cfg.user_residual_scale * residual
    action_st = trainer._straight_through_user_actions(logits)

    action_sums = action_st.sum(dim=-1)
    torch.testing.assert_close(
        action_sums,
        torch.ones_like(action_sums),
    )
    assert action_st.shape == logits.shape
    assert torch.all((action_st == 0.0) | (action_st == 1.0))

    weights = torch.arange(
        action_st.numel(),
        dtype=action_st.dtype,
    ).reshape_as(action_st)
    loss = torch.sum(action_st * weights)
    assert trainer.bundle.user_actor is not None
    trainer.bundle.user_actor.zero_grad()
    loss.backward()
    grad_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in trainer.bundle.user_actor.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_server_interface_is_fixed_width_for_changing_associations() -> None:
    cfg = MECConfig(num_users=7, num_servers=3, seed=2051)
    trainer = HeteroOffPolicyTrainer(
        MECEnvironment(cfg),
        _small_train_config(),
        _proposed_spec(cfg),
    )
    env = trainer.env
    prices = np.full(cfg.num_servers, 0.5 * (cfg.price_min + cfg.price_max))
    association_patterns = (
        np.zeros(cfg.num_users, dtype=np.int64),
        np.asarray([1, 1, 1, 2, 2, 3, 0], dtype=np.int64),
    )

    assert trainer.server_joint_action_dim == 3 * cfg.num_servers
    assert trainer.joint_action_dim == cfg.num_users * (cfg.num_servers + 1) + 3 * cfg.num_servers
    for user_actions in association_patterns:
        alloc_obs = env.get_allocation_observations(
            user_actions,
            prices,
            trainer.spec,
        )
        assert alloc_obs.shape == (cfg.num_servers, 7)
        alloc_output = trainer._allocation_actor_output(
            torch.tensor(alloc_obs, dtype=torch.float32)
        )
        assert alloc_output.shape == (cfg.num_servers, 2)


def test_server_only_coordinate_updates_leave_user_actor_unchanged() -> None:
    torch.manual_seed(2053)
    np.random.seed(2053)
    cfg = MECConfig(num_users=3, num_servers=2, seed=2053)
    train_cfg = _small_train_config(
        user_actor_update_interval=1000,
        server_actor_update_interval=1,
    )
    trainer = HeteroOffPolicyTrainer(
        MECEnvironment(cfg),
        train_cfg,
        _proposed_spec(cfg),
    )
    assert trainer.bundle.user_actor is not None
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.bundle.user_actor.named_parameters()
    }

    trainer.train()

    for name, parameter in trainer.bundle.user_actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)
    assert trainer.user_actor_opt is not None
    assert trainer.price_actor_opt is not None
    assert trainer.alloc_actor_opt is not None
    assert len(trainer.user_actor_opt.state) == 0
    assert len(trainer.price_actor_opt.state) > 0
    assert len(trainer.alloc_actor_opt.state) > 0
