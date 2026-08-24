from __future__ import annotations

from dataclasses import replace

import numpy as np

from paper_mec.config import MECConfig, MethodSpec
from paper_mec.env import MECEnvironment


def test_cpu_allocation_mapping_is_feasible_for_all_controls() -> None:
    cfg = replace(MECConfig(), num_users=40)
    env = MECEnvironment(cfg, np.random.default_rng(2026))
    users = np.arange(cfg.num_users, dtype=np.int64)

    for server in range(cfg.num_servers):
        for utilization in (cfg.cpu_utilization_min, 0.5, 1.0):
            for temperature in (0.0, 0.5, 2.0, cfg.alloc_temp_max):
                shares = env._allocation_from_controls(
                    users,
                    server,
                    utilization,
                    temperature,
                )

                assert shares.shape == (cfg.num_users,)
                assert np.all(np.isfinite(shares))
                assert np.all(shares >= 0.0)
                np.testing.assert_allclose(
                    shares.sum(),
                    utilization * cfg.server_cpu_hz,
                    rtol=0.0,
                    atol=1.0e-3,
                )


def test_single_admitted_user_never_exceeds_server_capacity() -> None:
    cfg = MECConfig()
    env = MECEnvironment(cfg, np.random.default_rng(2027))

    for server in range(cfg.num_servers):
        shares = env._allocation_from_controls(
            np.asarray([0], dtype=np.int64),
            server,
            utilization=1.0,
            temperature=cfg.alloc_temp_max,
        )
        assert shares[0] >= 0.0
        assert shares[0] <= cfg.server_cpu_hz + 1.0e-3


def test_queue_transition_uses_sum_of_projected_cpu_allocations() -> None:
    cfg = replace(MECConfig(), num_servers=2, slot_duration_s=1.0)
    env = MECEnvironment(cfg, np.random.default_rng(2036))
    backlog = np.asarray([8.0e9, 8.0e9], dtype=np.float64)
    arrivals = np.asarray([4.0e9, 4.0e9], dtype=np.float64)
    allocated_rates = np.asarray(
        [cfg.server_cpu_hz, 0.5 * cfg.server_cpu_hz],
        dtype=np.float64,
    )

    served, next_queues = env._queue_transition(
        backlog,
        arrivals,
        allocated_rates,
    )

    np.testing.assert_allclose(served, np.asarray([12.0e9, 6.0e9]))
    np.testing.assert_allclose(next_queues, np.asarray([0.0, 6.0e9]))


def test_environment_reports_allocation_coupled_service_rate() -> None:
    cfg = replace(MECConfig(), num_users=8)
    env = MECEnvironment(cfg, np.random.default_rng(2037))
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
        MethodSpec(name="allocation-coupling-test"),
    )

    for server in range(cfg.num_servers):
        users = np.where(outcome["executed_actions"] == (server + 1))[0]
        if users.size == 0:
            continue
        np.testing.assert_allclose(
            outcome["allocated_cpu_rates"][server],
            outcome["user_cpu_allocations"][users].sum(),
            rtol=0.0,
            atol=1.0e-3,
        )


def test_same_slot_admitted_workload_contributes_to_waiting_delay() -> None:
    cfg = replace(
        MECConfig(),
        num_users=4,
        num_servers=1,
        queue_capacity_cycles=5.0e10,
    )
    env = MECEnvironment(cfg, np.random.default_rng(2040))
    env.queues[:] = 1.0e9
    env.current_slot.cycles = np.asarray(
        [1.0e9, 2.0e9, 3.0e9, 4.0e9],
        dtype=np.float64,
    )

    outcome = env.step(
        np.ones(cfg.num_users, dtype=np.int64),
        np.asarray([0.55], dtype=np.float64),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        MethodSpec(name="same-slot-wait-test"),
    )

    expected_ahead = np.asarray([0.0, 1.0e9, 3.0e9, 6.0e9])
    expected_wait = (1.0e9 + expected_ahead) / cfg.server_cpu_hz
    np.testing.assert_allclose(outcome["same_slot_workload_ahead"], expected_ahead)
    np.testing.assert_allclose(outcome["waiting_by_user"], expected_wait)
    assert np.all(np.diff(outcome["waiting_by_user"]) > 0.0)


def test_environment_projects_price_and_allocation_controls_every_slot() -> None:
    cfg = MECConfig()
    env = MECEnvironment(cfg, np.random.default_rng(2028))
    tentative = np.zeros(cfg.num_users, dtype=np.int64)
    proposed_prices = np.asarray([-1.0, 0.25, 0.75, 2.0], dtype=np.float64)
    proposed_controls = np.asarray(
        [
            [-3.0, -3.0],
            [0.5, 0.5],
            [1.0, cfg.alloc_temp_max],
            [100.0, 100.0],
        ],
        dtype=np.float64,
    )

    outcome = env.step(
        tentative,
        proposed_prices,
        proposed_controls,
        MethodSpec(name="interval-projection-test"),
    )

    executed_prices = outcome["executed_prices"]
    executed_controls = outcome["executed_allocation_controls"]
    assert np.all(executed_prices >= cfg.price_min)
    assert np.all(executed_prices <= cfg.price_max)
    assert np.all(executed_controls[:, 0] >= cfg.cpu_utilization_min)
    assert np.all(executed_controls[:, 0] <= 1.0)
    assert np.all(executed_controls[:, 1] >= 0.0)
    assert np.all(executed_controls[:, 1] <= cfg.alloc_temp_max)


def test_environment_maps_out_of_domain_requests_to_local_execution() -> None:
    cfg = MECConfig()
    env = MECEnvironment(cfg, np.random.default_rng(2029))
    tentative = np.zeros(cfg.num_users, dtype=np.int64)
    tentative[0] = -1
    tentative[1] = cfg.num_servers + 3

    outcome = env.step(
        tentative,
        np.full(cfg.num_servers, 0.55, dtype=np.float64),
        np.column_stack(
            [
                np.ones(cfg.num_servers, dtype=np.float64),
                np.zeros(cfg.num_servers, dtype=np.float64),
            ]
        ),
        MethodSpec(name="request-domain-test"),
    )

    assert outcome["executed_actions"][0] == 0
    assert outcome["executed_actions"][1] == 0
    assert np.all(outcome["executed_actions"] >= 0)
    assert np.all(outcome["executed_actions"] <= cfg.num_servers)


def test_learned_utilization_changes_queue_service_and_backlog() -> None:
    cfg = replace(MECConfig(), num_users=4, num_servers=1, slot_duration_s=1.0)
    env = MECEnvironment(cfg, np.random.default_rng(2041))
    users = np.arange(cfg.num_users, dtype=np.int64)

    low_shares = env._allocation_from_controls(
        users,
        server=0,
        utilization=0.25,
        temperature=0.0,
    )
    high_shares = env._allocation_from_controls(
        users,
        server=0,
        utilization=0.75,
        temperature=0.0,
    )
    backlog = np.asarray([cfg.server_cpu_hz], dtype=np.float64)
    arrivals = np.asarray([cfg.server_cpu_hz], dtype=np.float64)

    low_served, low_queue = env._queue_transition(
        backlog,
        arrivals,
        np.asarray([low_shares.sum()]),
    )
    high_served, high_queue = env._queue_transition(
        backlog,
        arrivals,
        np.asarray([high_shares.sum()]),
    )

    assert high_served[0] > low_served[0]
    assert high_queue[0] < low_queue[0]


def test_allocation_shape_changes_jain_fairness_without_changing_total_rate() -> None:
    cfg = replace(MECConfig(), num_users=8, num_servers=1)
    env = MECEnvironment(cfg, np.random.default_rng(2042))
    users = np.arange(cfg.num_users, dtype=np.int64)

    equal_shares = env._allocation_from_controls(
        users,
        server=0,
        utilization=0.75,
        temperature=0.0,
    )
    shaped_shares = env._allocation_from_controls(
        users,
        server=0,
        utilization=0.75,
        temperature=cfg.alloc_temp_max,
    )

    np.testing.assert_allclose(equal_shares.sum(), shaped_shares.sum(), atol=1.0e-3)
    equal_jain = equal_shares.sum() ** 2 / (
        users.size * np.square(equal_shares).sum()
    )
    shaped_jain = shaped_shares.sum() ** 2 / (
        users.size * np.square(shaped_shares).sum()
    )
    np.testing.assert_allclose(equal_jain, 1.0, atol=1.0e-12)
    assert shaped_jain < equal_jain


def test_allocation_projection_is_permutation_equivariant() -> None:
    cfg = replace(MECConfig(), num_users=7, num_servers=1, seed=2052)
    env = MECEnvironment(cfg, np.random.default_rng(2052))
    users = np.asarray([0, 2, 4, 6], dtype=np.int64)
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    shares = env._allocation_from_controls(
        users,
        server=0,
        utilization=0.72,
        temperature=3.5,
    )
    permuted_shares = env._allocation_from_controls(
        users[permutation],
        server=0,
        utilization=0.72,
        temperature=3.5,
    )
    np.testing.assert_allclose(permuted_shares, shares[permutation])


def test_allocation_observation_width_is_independent_of_association_count() -> None:
    cfg = replace(MECConfig(), num_users=7, num_servers=3, seed=2054)
    env = MECEnvironment(cfg, np.random.default_rng(2054))
    prices = np.full(cfg.num_servers, 0.5 * (cfg.price_min + cfg.price_max))
    patterns = (
        np.zeros(cfg.num_users, dtype=np.int64),
        np.asarray([1, 1, 1, 2, 2, 3, 0], dtype=np.int64),
    )

    for actions in patterns:
        observation = env.get_allocation_observations(
            actions,
            prices,
            MethodSpec(name="fixed-allocation-observation-test"),
        )
        assert observation.shape == (cfg.num_servers, 7)


def test_user_observation_exposes_candidate_queue_advertisements_only() -> None:
    cfg = MECConfig()
    env = MECEnvironment(cfg, np.random.default_rng(2033))
    env.queues = np.linspace(
        0.1 * cfg.queue_capacity_cycles,
        0.4 * cfg.queue_capacity_cycles,
        cfg.num_servers,
    )
    prices = np.linspace(cfg.price_min, cfg.price_max, cfg.num_servers)
    spec = MethodSpec(name="advertisement-interface-test", queue_visible=True)

    observations = env.get_user_observations(prices, spec)
    server_count = cfg.num_servers
    price_start = 4 + 2 * server_count
    queue_start = 4 + 3 * server_count

    expected_price_ratio = (prices - cfg.price_min) / (cfg.price_max - cfg.price_min)
    expected_queue_ratio = env.queues / cfg.queue_capacity_cycles
    for user in range(cfg.num_users):
        candidates = env.candidate_mask[user]
        np.testing.assert_allclose(
            observations[user, price_start : price_start + server_count],
            expected_price_ratio * candidates,
        )
        np.testing.assert_allclose(
            observations[user, queue_start : queue_start + server_count],
            expected_queue_ratio * candidates,
        )


def test_server_price_observation_uses_no_current_user_task_attributes() -> None:
    cfg = MECConfig()
    left = MECEnvironment(cfg, np.random.default_rng(2034))
    right = MECEnvironment(cfg, np.random.default_rng(2035))

    right.queues = left.queues.copy()
    right.prices = left.prices.copy()
    right.prev_load = left.prev_load.copy()
    right.prev_counts = left.prev_counts.copy()
    right.prev_jain = left.prev_jain.copy()

    # The environments have different current task/channel samples.  Pricing
    # observations must nevertheless match because price is chosen before
    # tentative user requests reveal current task attributes.
    assert not np.array_equal(left.current_slot.cycles, right.current_slot.cycles)
    np.testing.assert_allclose(
        left.get_price_observations(MethodSpec(name="left")),
        right.get_price_observations(MethodSpec(name="right")),
    )
