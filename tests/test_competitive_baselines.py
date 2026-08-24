from __future__ import annotations

from dataclasses import replace

import numpy as np

from paper_mec.baselines import (
    _predicted_profile,
    dpp_joint_pricing_offloading_policy,
    stackelberg_joint_pricing_offloading_policy,
)
from paper_mec.config import MECConfig, MethodSpec
from paper_mec.env import MECEnvironment


def _assert_policy_output_is_feasible(
    env: MECEnvironment,
    actions: np.ndarray,
    prices: np.ndarray,
    controls: np.ndarray,
) -> None:
    cfg = env.cfg
    assert actions.shape == (cfg.num_users,)
    assert prices.shape == (cfg.num_servers,)
    assert controls.shape == (cfg.num_servers, 2)
    assert np.all(np.isfinite(prices))
    assert np.all(np.isfinite(controls))
    assert np.all(prices >= cfg.price_min)
    assert np.all(prices <= cfg.price_max)
    assert np.all(controls[:, 0] >= cfg.cpu_utilization_min)
    assert np.all(controls[:, 0] <= 1.0)
    assert np.all(controls[:, 1] >= 0.0)
    assert np.all(controls[:, 1] <= cfg.alloc_temp_max)
    for user, action in enumerate(actions):
        assert 0 <= action <= cfg.num_servers
        if action > 0:
            assert env.candidate_mask[user, action - 1] > 0.5


def test_model_based_baselines_return_valid_controls() -> None:
    cfg = replace(
        MECConfig(),
        num_users=12,
        baseline_price_grid_size=5,
        baseline_best_response_rounds=1,
    )
    for policy in (
        dpp_joint_pricing_offloading_policy,
        stackelberg_joint_pricing_offloading_policy,
    ):
        env = MECEnvironment(cfg, np.random.default_rng(2030))
        actions, prices, controls = policy(env)
        _assert_policy_output_is_feasible(env, actions, prices, controls)


def test_model_based_baselines_share_finite_buffer_projection() -> None:
    cfg = replace(
        MECConfig(),
        num_users=24,
        arrival_scale=1.8,
        queue_capacity_cycles=6.0e9,
        baseline_price_grid_size=5,
        baseline_best_response_rounds=1,
    )
    for policy in (
        dpp_joint_pricing_offloading_policy,
        stackelberg_joint_pricing_offloading_policy,
    ):
        env = MECEnvironment(cfg, np.random.default_rng(2031))
        spec = MethodSpec(
            name=policy.__name__,
            queue_visible=True,
            queue_in_delay=True,
            queue_penalty=cfg.queue_penalty,
            fairness_penalty=cfg.fairness_penalty,
        )
        for _ in range(30):
            actions, prices, controls = policy(env)
            outcome = env.step(actions, prices, controls, spec)
            assert np.all(env.queues <= cfg.queue_capacity_cycles + 1.0e-6)
            assert outcome["metrics"]["max_queue_occupancy"] <= 1.0 + 1.0e-9
            assert outcome["metrics"]["overflow_count"] == 0.0


def test_model_based_baselines_are_deterministic_for_identical_state() -> None:
    cfg = replace(
        MECConfig(),
        num_users=10,
        baseline_price_grid_size=5,
        baseline_best_response_rounds=1,
    )
    for policy in (
        dpp_joint_pricing_offloading_policy,
        stackelberg_joint_pricing_offloading_policy,
    ):
        left = MECEnvironment(cfg, np.random.default_rng(2032))
        right = MECEnvironment(cfg, np.random.default_rng(2032))
        left_output = policy(left)
        right_output = policy(right)
        for left_item, right_item in zip(left_output, right_output):
            np.testing.assert_allclose(left_item, right_item)


def test_model_based_one_step_queue_matches_environment_transition() -> None:
    """The model-based comparator uses the exact queue recurrence, not a fit."""
    cfg = replace(
        MECConfig(),
        num_users=12,
        arrival_scale=1.2,
        baseline_price_grid_size=5,
        baseline_best_response_rounds=1,
    )
    env = MECEnvironment(cfg, np.random.default_rng(2033))
    env.queues = np.linspace(0.1, 0.7, cfg.num_servers) * cfg.queue_capacity_cycles
    actions, prices, controls = dpp_joint_pricing_offloading_policy(env)
    _, _, predicted_queues, _ = _predicted_profile(env, actions, prices)
    spec = MethodSpec(
        name="exact-queue-transition-check",
        queue_visible=True,
        queue_in_delay=True,
        queue_penalty=cfg.queue_penalty,
        fairness_penalty=cfg.fairness_penalty,
    )
    env.step(actions, prices, controls, spec)
    np.testing.assert_allclose(env.queues, predicted_queues, rtol=0.0, atol=1.0e-6)
