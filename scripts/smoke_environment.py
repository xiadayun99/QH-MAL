from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_mec.config import MECConfig, MethodSpec
from paper_mec.env import MECEnvironment


def main() -> None:
    cfg = replace(
        MECConfig(),
        arrival_scale=4.0,
        queue_capacity_cycles=0.5 * MECConfig().server_cpu_hz,
        seed=20260822,
    )
    env = MECEnvironment(cfg, np.random.default_rng(cfg.seed))
    spec = MethodSpec(name="projection-smoke")

    for _ in range(500):
        requests = np.asarray(
            [int(np.flatnonzero(env.candidate_mask[user])[0]) + 1 for user in range(cfg.num_users)],
            dtype=np.int64,
        )
        prices = np.linspace(cfg.price_min - 1.0, cfg.price_max + 1.0, cfg.num_servers)
        controls = np.column_stack(
            [
                np.full(cfg.num_servers, 2.0),
                np.full(cfg.num_servers, -1.0),
            ]
        )
        outcome = env.step(requests, prices, controls, spec)
        metrics = outcome["metrics"]
        assert np.all(env.queues >= -1.0e-6)
        assert np.all(env.queues <= cfg.queue_capacity_cycles + 1.0e-6)
        assert metrics["overflow_count"] == 0.0
        assert metrics["hard_constraint_violation_rate"] == 0.0

    print("Projection smoke test passed: 500 overloaded slots, zero overflow, all queues bounded.")


if __name__ == "__main__":
    main()

