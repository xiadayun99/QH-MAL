from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from run import apply_train_overrides, run_learning_method
from paper_mec.config import MECConfig, MethodSpec, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.evaluation import aggregate_episode_outcomes
from paper_mec.utils import ensure_dir, seed_everything


def proposed_spec(cfg: MECConfig) -> MethodSpec:
    return MethodSpec(
        name="Proposed Method",
        queue_visible=True,
        queue_in_delay=True,
        queue_penalty=cfg.queue_penalty,
        fairness_penalty=cfg.fairness_penalty,
        two_timescale=True,
    )


def aggregate_long_horizon(trainer, env: MECEnvironment, horizon: int) -> dict[str, float]:
    outcomes: list[dict[str, object]] = []
    occupancy: list[float] = []
    capacity_hits: list[float] = []
    env.reset()
    for _ in range(horizon):
        tentative, prices, allocation_controls = trainer.act_deterministically(env)
        outcome = env.step(tentative, prices, allocation_controls, trainer.spec)
        outcomes.append(outcome)
        occupancy.append(float(np.mean(env.queues / env.cfg.queue_capacity_cycles)))
        capacity_hits.append(float(outcome["metrics"]["max_queue_occupancy"] >= 0.999))
    aggregate = aggregate_episode_outcomes(outcomes)
    occupancy_array = np.asarray(occupancy, dtype=np.float64)
    split = max(horizon // 2, 1)
    aggregate.update({
        "horizon_slots": float(horizon),
        "queue_occupancy_first_half": float(occupancy_array[:split].mean()),
        "queue_occupancy_second_half": float(occupancy_array[split:].mean())
        if split < horizon
        else float(occupancy_array[:split].mean()),
        "queue_occupancy_slope_per_1000_slots": float(
            np.polyfit(np.arange(horizon), occupancy_array, 1)[0] * 1000.0
        )
        if horizon > 1
        else 0.0,
        "capacity_hit_ratio": float(np.mean(capacity_hits)),
    })
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain across finite-buffer capacities and run a long-horizon heavy-load stress test."
    )
    parser.add_argument("--train-config", type=Path, default=Path("configs/paper_results.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--stress-arrival-scale", type=float, default=1.6)
    parser.add_argument("--buffer-windows", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--output-dir", type=Path, default=Path("buffer_stress_results"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    base_cfg = replace(MECConfig(), seed=args.seed)
    train_cfg = apply_train_overrides(TrainConfig(episodes=args.steps), args.train_config)
    train_cfg = replace(train_cfg, device=args.device)
    if args.quick:
        train_cfg = replace(
            train_cfg,
            episodes=min(train_cfg.episodes, 8),
            episode_length=min(train_cfg.episode_length, 10),
            eval_episodes=min(train_cfg.eval_episodes, 2),
            warmup_steps=min(train_cfg.warmup_steps, 20),
            batch_size=min(train_cfg.batch_size, 16),
            selection_eval_episodes=min(train_cfg.selection_eval_episodes, 1),
            selection_interval=min(train_cfg.selection_interval, 4),
        )

    sensitivity_rows: list[dict[str, float]] = []
    default_trainer = None
    for window_s in args.buffer_windows:
        capacity = base_cfg.server_cpu_hz * float(window_s)
        cfg = replace(base_cfg, queue_capacity_cycles=capacity)
        trainer, _, metrics = run_learning_method(proposed_spec(cfg), cfg, train_cfg)
        sensitivity_rows.append(
            {
                "buffer_window_s": float(window_s),
                "queue_capacity_cycles": float(capacity),
                **metrics,
            }
        )
        if np.isclose(window_s, 1.0):
            default_trainer = trainer

    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "buffer_capacity_sensitivity.csv",
        index=False,
    )

    if default_trainer is None:
        cfg = replace(base_cfg, queue_capacity_cycles=base_cfg.server_cpu_hz)
        default_trainer, _, _ = run_learning_method(proposed_spec(cfg), cfg, train_cfg)
    stress_cfg = replace(default_trainer.cfg, arrival_scale=args.stress_arrival_scale)
    stress_env = MECEnvironment(stress_cfg, np.random.default_rng(args.seed + 50_000))
    stress = aggregate_long_horizon(default_trainer, stress_env, args.horizon)
    pd.DataFrame([{**stress, "arrival_scale": args.stress_arrival_scale}]).to_csv(
        args.output_dir / "long_horizon_stress_test.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
