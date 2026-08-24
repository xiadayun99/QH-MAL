from __future__ import annotations

import numpy as np

from paper_mec.config import MECConfig, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.models import CriticNet, TwinCritic
from paper_mec.trainer import HeteroOffPolicyTrainer
from run import build_learning_specs


def _small_train_config() -> TrainConfig:
    return TrainConfig(
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


def test_queue_aware_marl_baselines_share_information_and_rewards() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=17)
    specs = {spec.name: spec for spec in build_learning_specs(cfg)}
    proposed = specs["Proposed Method"]

    for name in (
        "Queue-Aware MADDPG",
        "Queue-Aware MATD3",
    ):
        spec = specs[name]
        assert spec.queue_visible == proposed.queue_visible is True
        assert spec.queue_in_delay == proposed.queue_in_delay is True
        assert spec.queue_penalty == proposed.queue_penalty
        assert spec.fairness_penalty == proposed.fairness_penalty

        env = MECEnvironment(cfg)
        prices = np.full(cfg.num_servers, 0.5 * (cfg.price_min + cfg.price_max))
        np.testing.assert_allclose(
            env.get_user_observations(prices, spec),
            env.get_user_observations(prices, proposed),
        )
        np.testing.assert_allclose(
            env.get_price_observations(spec),
            env.get_price_observations(proposed),
        )


def test_maddpg_and_matd3_use_the_documented_critic_and_update_rules() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=19)
    train_cfg = _small_train_config()
    specs = {spec.name: spec for spec in build_learning_specs(cfg)}

    maddpg = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, specs["Queue-Aware MADDPG"])
    assert isinstance(maddpg.user_critic, CriticNet)
    assert isinstance(maddpg.server_critic, CriticNet)
    assert maddpg._actor_update_intervals() == (1, 1)

    matd3 = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, specs["Queue-Aware MATD3"])
    assert isinstance(matd3.user_critic, TwinCritic)
    assert isinstance(matd3.server_critic, TwinCritic)
    assert matd3._actor_update_intervals() == (
        train_cfg.matd3_policy_delay,
        train_cfg.matd3_policy_delay,
    )

    proposed = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, specs["Proposed Method"])
    assert isinstance(proposed.user_critic, TwinCritic)
    assert proposed._actor_update_intervals() == (
        train_cfg.user_actor_update_interval,
        train_cfg.server_actor_update_interval,
    )


def test_all_same_information_learners_complete_a_gradient_smoke_run() -> None:
    cfg = MECConfig(num_users=3, num_servers=2, seed=23)
    train_cfg = _small_train_config()
    for spec in build_learning_specs(cfg):
        if spec.name == "Queue-Unaware QH-MAL":
            continue
        trainer = HeteroOffPolicyTrainer(MECEnvironment(cfg), train_cfg, spec)
        _, history = trainer.train()
        assert len(history) == 1
        assert trainer.best_model_state is not None
