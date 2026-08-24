from __future__ import annotations

from dataclasses import replace

import numpy as np

from paper_mec.config import MECConfig, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.trainer import HeteroOffPolicyTrainer
from run import build_ablation_specs


def _small_train_config() -> TrainConfig:
    return replace(
        TrainConfig(),
        episodes=1,
        episode_length=3,
        eval_episodes=1,
        curve_eval_episodes=1,
        curve_eval_interval=10,
        train_arrival_scales=(1.0,),
        selection_arrival_scales=(1.0,),
        selection_eval_episodes=1,
        selection_interval=1,
        critic_hidden_dims=(24,),
        batch_size=2,
        replay_buffer_size=16,
        warmup_steps=0,
        show_progress=False,
        device="cpu",
    )


def test_ablation_specs_separate_the_requested_factors() -> None:
    cfg = MECConfig()
    specs = {spec.name: spec for spec in build_ablation_specs(cfg)}
    full = specs["Full Model"]

    no_obs = specs["w/o Queue Observations"]
    assert no_obs.queue_visible is False
    assert no_obs.queue_in_delay is full.queue_in_delay is True
    assert no_obs.queue_penalty == full.queue_penalty

    no_queue_penalty = specs["w/o Queue Penalty"]
    assert no_queue_penalty.queue_visible is True
    assert no_queue_penalty.queue_in_delay is True
    assert no_queue_penalty.queue_penalty == 0.0

    no_slow_server = specs["w/o Slower Server Updates"]
    assert no_slow_server.two_timescale is False
    assert no_slow_server.equal_actor_learning_rates is True

    role_agnostic = specs["w/o Role-Specific Actors"]
    assert role_agnostic.role_specific_actors is False
    assert role_agnostic.queue_visible == full.queue_visible
    assert role_agnostic.queue_penalty == full.queue_penalty
    assert role_agnostic.two_timescale == full.two_timescale


def test_queue_observation_ablation_keeps_congestion_rewards() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=41)
    specs = {spec.name: spec for spec in build_ablation_specs(cfg)}
    full = specs["Full Model"]
    no_obs = specs["w/o Queue Observations"]
    no_penalty = specs["w/o Queue Penalty"]

    env = MECEnvironment(cfg)
    env.queues = np.array([0.25, 0.75]) * cfg.queue_capacity_cycles
    full_price_obs = env.get_price_observations(full)
    no_obs_price_obs = env.get_price_observations(no_obs)

    np.testing.assert_allclose(full_price_obs[:, 0], [0.25, 0.75])
    np.testing.assert_allclose(no_obs_price_obs[:, 0], 0.0)
    np.testing.assert_allclose(
        env.get_price_observations(no_penalty),
        full_price_obs,
    )
    assert no_obs.queue_penalty == cfg.queue_penalty
    assert no_penalty.queue_penalty == 0.0

    trainer = HeteroOffPolicyTrainer(
        MECEnvironment(cfg),
        _small_train_config(),
        no_obs,
    )
    left = MECEnvironment(cfg, np.random.default_rng(409))
    right = MECEnvironment(cfg, np.random.default_rng(409))
    left.queues[:] = 0.0
    right.queues[:] = cfg.queue_capacity_cycles
    prices = np.full(cfg.num_servers, 0.5 * (cfg.price_min + cfg.price_max))
    np.testing.assert_array_equal(
        trainer._heuristic_user_actions(left, prices),
        trainer._heuristic_user_actions(right, prices),
    )


def test_slow_server_ablation_equalizes_frequency_and_learning_rate() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=43)
    train_cfg = _small_train_config()
    spec = next(
        item
        for item in build_ablation_specs(cfg)
        if item.name == "w/o Slower Server Updates"
    )
    trainer = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, spec)

    assert trainer._actor_update_intervals() == (1, 1)
    assert trainer.server_actor_lr == train_cfg.user_actor_lr
    assert trainer.price_actor_opt is not None
    assert trainer.alloc_actor_opt is not None
    assert trainer.price_actor_opt.param_groups[0]["lr"] == train_cfg.user_actor_lr
    assert trainer.alloc_actor_opt.param_groups[0]["lr"] == train_cfg.user_actor_lr


def test_role_agnostic_ablation_is_parameter_matched_and_trainable() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=47)
    train_cfg = _small_train_config()
    specs = {spec.name: spec for spec in build_ablation_specs(cfg)}
    full = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, specs["Full Model"])
    ablated = HeteroOffPolicyTrainer(
        MECEnvironment(cfg),
        train_cfg,
        specs["w/o Role-Specific Actors"],
    )

    assert full.bundle.role_conditioned_actor is None
    assert ablated.bundle.user_actor is None
    assert ablated.bundle.price_actor is None
    assert ablated.bundle.alloc_actor is None
    assert ablated.bundle.role_conditioned_actor is not None
    relative_parameter_gap = abs(
        ablated.actor_parameter_count() - full.actor_parameter_count()
    ) / full.actor_parameter_count()
    assert relative_parameter_gap < 0.10

    _, history = ablated.train()
    assert len(history) == 1
    assert ablated.best_model_state is not None
