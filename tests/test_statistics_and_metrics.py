from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from paper_mec.config import MECConfig, MethodSpec
from paper_mec.env import MECEnvironment
from paper_mec.evaluation import aggregate_episode_outcomes
from paper_mec.statistics import (
    build_paired_inference_table,
    exact_paired_sign_flip_pvalue,
    holm_adjust,
)
from scripts.run_multiseed_evaluation import (
    build_stratified_paired_inference_table,
)


def _local_outcome(env: MECEnvironment) -> dict[str, object]:
    cfg = env.cfg
    return env.step(
        np.zeros(cfg.num_users, dtype=np.int64),
        np.full(cfg.num_servers, 0.5, dtype=np.float64),
        np.column_stack(
            [
                np.ones(cfg.num_servers, dtype=np.float64),
                np.zeros(cfg.num_servers, dtype=np.float64),
            ]
        ),
        MethodSpec(name="metric-test"),
    )


def test_delay_decomposition_and_participation_metrics_are_auditable() -> None:
    env = MECEnvironment(MECConfig(num_users=8), np.random.default_rng(6101))
    outcomes = [_local_outcome(env) for _ in range(5)]
    summary = aggregate_episode_outcomes(outcomes)

    assert summary["delay_decomposition_error"] < 1.0e-10
    assert summary["requested_offload_ratio"] == 0.0
    assert summary["edge_participation_ratio"] == 0.0
    assert summary["local_execution_ratio"] == 1.0
    assert summary["user_service_fairness"] == 0.0
    assert summary["hard_constraint_violation_rate"] == 0.0


def test_unseen_distributions_are_deterministic_but_differ_from_training() -> None:
    base = MECConfig(num_users=10, num_servers=3, seed=6102)
    left = MECEnvironment(
        replace(
            base,
            arrival_distribution="bursty_lognormal",
            channel_distribution="beta_low",
        ),
        np.random.default_rng(6102),
    )
    right = MECEnvironment(
        replace(
            base,
            arrival_distribution="bursty_lognormal",
            channel_distribution="beta_low",
        ),
        np.random.default_rng(6102),
    )
    in_distribution = MECEnvironment(base, np.random.default_rng(6102))

    np.testing.assert_allclose(left.current_slot.cycles, right.current_slot.cycles)
    np.testing.assert_allclose(left.current_slot.uplink_mbps, right.current_slot.uplink_mbps)
    assert not np.allclose(left.current_slot.cycles, in_distribution.current_slot.cycles)
    assert not np.allclose(left.current_slot.uplink_mbps, in_distribution.current_slot.uplink_mbps)


def test_five_seed_exact_test_exposes_its_resolution() -> None:
    advantages = np.ones(5, dtype=np.float64)
    assert exact_paired_sign_flip_pvalue(advantages) == 0.0625
    assert holm_adjust([0.01, 0.04, 0.20]) == [0.03, 0.08, 0.20]


def test_paired_inference_uses_shared_seed_pairs_and_holm_correction() -> None:
    rows = []
    for seed, proposed, baseline in zip(
        [7, 11, 19, 23, 31],
        [0.90, 0.91, 0.92, 0.93, 0.94],
        [1.00, 1.01, 1.02, 1.03, 1.04],
    ):
        rows.extend(
            [
                {"Seed": seed, "Method": "Proposed Method", "Avg Delay": proposed},
                {"Seed": seed, "Method": "Baseline", "Avg Delay": baseline},
            ]
        )
    table = build_paired_inference_table(
        pd.DataFrame(rows),
        ["Avg Delay"],
        {"Avg Delay"},
    )
    assert table.loc[0, "Seed Count"] == 5
    assert table.loc[0, "Mean Paired Advantage"] > 0.0
    assert table.loc[0, "Exact Two-Sided Sign-Flip p"] == 0.0625
    assert not bool(table.loc[0, "Holm Significant at 0.05"])


def test_shift_inference_keeps_holm_families_scenario_specific() -> None:
    rows = []
    for scenario in ["unseen-arrival", "unseen-channel"]:
        for seed in [7, 11, 19, 23, 31]:
            rows.extend(
                [
                    {
                        "Seed": seed,
                        "scenario": scenario,
                        "method": "Proposed Method",
                        "avg_delay": 0.9,
                    },
                    {
                        "Seed": seed,
                        "scenario": scenario,
                        "method": "Baseline",
                        "avg_delay": 1.0,
                    },
                ]
            )

    table = build_stratified_paired_inference_table(
        pd.DataFrame(rows),
        ["scenario"],
        ["avg_delay"],
        {"avg_delay"},
    )
    assert table["scenario"].nunique() == 2
    assert len(table) == 2
    assert np.allclose(table["Exact Two-Sided Sign-Flip p"], 0.0625)
    assert np.allclose(table["Holm-Adjusted p"], 0.0625)
