from __future__ import annotations

from dataclasses import replace

import numpy as np

from paper_mec.config import MECConfig, MethodSpec
from paper_mec.env import MECEnvironment


def test_admission_projection_preserves_buffer_bound_under_overload() -> None:
    cfg = replace(
        MECConfig(),
        num_users=40,
        arrival_scale=2.0,
        queue_capacity_cycles=6.0e9,
    )
    env = MECEnvironment(cfg, np.random.default_rng(123))
    spec = MethodSpec(name="finite-buffer-test")

    observed_fallback = False
    for _ in range(500):
        tentative = np.ones(cfg.num_users, dtype=np.int64)
        prices = np.full(cfg.num_servers, 0.55, dtype=np.float64)
        allocation_controls = np.column_stack(
            [
                np.ones(cfg.num_servers, dtype=np.float64),
                np.zeros(cfg.num_servers, dtype=np.float64),
            ]
        )
        outcome = env.step(tentative, prices, allocation_controls, spec)

        assert np.all(env.queues >= 0.0)
        assert np.all(env.queues <= cfg.queue_capacity_cycles + 1.0e-6)
        assert outcome["metrics"]["max_queue_occupancy"] <= 1.0 + 1.0e-9
        assert outcome["metrics"]["overflow_count"] == 0.0
        observed_fallback |= outcome["metrics"]["fallback_ratio"] > 0.0

    assert observed_fallback


def test_unadmitted_requests_execute_locally_without_payment() -> None:
    cfg = replace(
        MECConfig(),
        num_users=20,
        arrival_scale=2.5,
        queue_capacity_cycles=1.0e9,
    )
    env = MECEnvironment(cfg, np.random.default_rng(321))
    tentative = np.ones(cfg.num_users, dtype=np.int64)
    outcome = env.step(
        tentative,
        np.full(cfg.num_servers, 0.55, dtype=np.float64),
        np.column_stack(
            [
                np.ones(cfg.num_servers, dtype=np.float64),
                np.zeros(cfg.num_servers, dtype=np.float64),
            ]
        ),
        MethodSpec(name="redirect-test"),
    )

    redirected = (tentative > 0) & (outcome["executed_actions"] == 0)
    assert redirected.any()
    assert np.all(outcome["user_payment"][redirected] == 0.0)
    assert outcome["metrics"]["fallback_ratio"] > 0.0
    assert outcome["metrics"]["redirected_workload"] > 0.0


def test_admission_budget_does_not_precredit_current_slot_service() -> None:
    cfg = replace(
        MECConfig(),
        num_users=2,
        num_servers=1,
        queue_capacity_cycles=5.0e9,
    )
    env = MECEnvironment(cfg, np.random.default_rng(322))
    env.queues[:] = 4.0e9
    env.current_slot.cycles = np.asarray([0.6e9, 0.6e9], dtype=np.float64)

    executed, admission, budget, _, _ = env._apply_admission_projection(
        np.ones(cfg.num_users, dtype=np.int64)
    )

    np.testing.assert_allclose(budget, np.asarray([1.0e9]))
    admitted_work = env.current_slot.cycles[admission.astype(bool)].sum()
    assert admitted_work <= budget[0] + 1.0e-9
    assert env.queues[0] + admitted_work <= cfg.queue_capacity_cycles + 1.0e-9
    assert np.count_nonzero(executed) == 1
