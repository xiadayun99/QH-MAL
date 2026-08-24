from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def jain_index(values: np.ndarray, *, zero_value: float = 0.0) -> float:
    """Return Jain's index for a nonnegative vector.

    A zero service vector is assigned ``zero_value`` because equal denial of
    service should not be reported as perfect access fairness.  The companion
    edge-participation metric makes this convention explicit in the tables.
    """

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return float(zero_value)
    if np.any(~np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("Jain-index inputs must be finite and nonnegative.")
    squared_sum = float(np.square(vector).sum())
    if squared_sum <= 1.0e-18:
        return float(zero_value)
    return float(vector.sum() ** 2 / (vector.size * squared_sum))


def aggregate_episode_outcomes(
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate scalar and individual-user metrics over one episode.

    Slot-wise server allocation fairness and long-horizon user fairness answer
    different questions.  The former remains in ``fairness``.  The latter is
    computed here from each named user's edge-service rate across the complete
    episode, together with a delay-satisfaction Jain index and worst-user
    delay.  This avoids averaging a per-slot binary Jain statistic that would
    confound participation with fairness.
    """

    if not outcomes:
        raise ValueError("At least one environment outcome is required.")

    metric_keys = tuple(outcomes[0]["metrics"].keys())
    summary = {
        key: float(np.mean([float(outcome["metrics"][key]) for outcome in outcomes]))
        for key in metric_keys
    }

    service = np.stack(
        [np.asarray(outcome["offloaded_indicators"], dtype=np.float64) for outcome in outcomes],
        axis=0,
    )
    delays = np.stack(
        [np.asarray(outcome["actual_delay_by_user"], dtype=np.float64) for outcome in outcomes],
        axis=0,
    )
    payments = np.stack(
        [np.asarray(outcome["user_payment"], dtype=np.float64) for outcome in outcomes],
        axis=0,
    )

    per_user_service_rate = service.mean(axis=0)
    per_user_avg_delay = delays.mean(axis=0)
    per_user_avg_payment = payments.mean(axis=0)
    delay_satisfaction = 1.0 / np.maximum(per_user_avg_delay, 1.0e-9)

    summary.update(
        {
            "user_service_fairness": jain_index(per_user_service_rate),
            "user_delay_fairness": jain_index(delay_satisfaction),
            "p05_user_service_rate": float(np.quantile(per_user_service_rate, 0.05)),
            "worst_user_avg_delay": float(np.max(per_user_avg_delay)),
            "p95_user_avg_payment": float(np.quantile(per_user_avg_payment, 0.95)),
            "max_min_user_service_gap": float(
                np.max(per_user_service_rate) - np.min(per_user_service_rate)
            ),
        }
    )

    component_sum = (
        summary["avg_uplink_delay"]
        + summary["avg_waiting_delay_all_users"]
        + summary["avg_execution_delay"]
        + summary["avg_downlink_delay"]
    )
    summary["delay_decomposition_error"] = float(
        abs(summary["avg_delay"] - component_sum)
    )
    return summary
