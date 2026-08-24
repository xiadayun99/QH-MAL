from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from paper_mec.baselines import (
    DPP_JPO,
    MODEL_BASED_BASELINES,
    STACKELBERG_JPO,
    dpp_joint_pricing_offloading_policy,
    greedy_policy,
    random_policy,
    stackelberg_joint_pricing_offloading_policy,
)
from paper_mec.config import MECConfig, MethodSpec, PathConfig, TrainConfig
from paper_mec.env import MECEnvironment
from paper_mec.evaluation import aggregate_episode_outcomes
from paper_mec.reporting import plot_arrival_dashboard, plot_metric_vs_x, plot_penalty_dashboard, plot_training_curve, save_dataframe
from paper_mec.trainer import HeteroOffPolicyTrainer, evaluate_policy
from paper_mec.utils import create_progress, ensure_dir, resolve_torch_device, save_json, seed_everything

METHOD_ORDER = [
    "Random Offloading",
    "Greedy Offloading",
    "Fixed-Pricing Queue-Aware Greedy",
    DPP_JPO,
    STACKELBERG_JPO,
    "Queue-Unaware QH-MAL",
    "Queue-Aware MADDPG",
    "Queue-Aware MATD3",
    "Proposed Method",
]

HEURISTIC_BASELINES = [
    "Random Offloading",
    "Greedy Offloading",
    "Fixed-Pricing Queue-Aware Greedy",
]
BASELINE_NAMES = HEURISTIC_BASELINES + MODEL_BASED_BASELINES
LEARNING_BASELINE_NAMES = [
    "Queue-Unaware QH-MAL",
    "Queue-Aware MADDPG",
    "Queue-Aware MATD3",
]


def build_learning_specs(mec_cfg: MECConfig) -> list[MethodSpec]:
    """Return the controlled learning comparison used throughout the paper.

    QA-MADDPG, QA-MATD3, and QH-MAL receive the same queue observations,
    rewards, action parameterization, and feasibility projections.  The
    queue-unaware variant changes only the queue inputs/reward and is therefore
    treated as an ablation rather than an external MARL baseline.
    """
    return [
        MethodSpec(
            name="Queue-Unaware QH-MAL",
            queue_visible=False,
            queue_in_delay=False,
            queue_penalty=0.0,
            fairness_penalty=mec_cfg.fairness_penalty,
            two_timescale=True,
            learner="qhmarl",
        ),
        MethodSpec(
            name="Queue-Aware MADDPG",
            queue_visible=True,
            queue_in_delay=True,
            queue_penalty=mec_cfg.queue_penalty,
            fairness_penalty=mec_cfg.fairness_penalty,
            two_timescale=False,
            learner="maddpg",
        ),
        MethodSpec(
            name="Queue-Aware MATD3",
            queue_visible=True,
            queue_in_delay=True,
            queue_penalty=mec_cfg.queue_penalty,
            fairness_penalty=mec_cfg.fairness_penalty,
            two_timescale=False,
            learner="matd3",
        ),
        MethodSpec(
            name="Proposed Method",
            queue_visible=True,
            queue_in_delay=True,
            queue_penalty=mec_cfg.queue_penalty,
            fairness_penalty=mec_cfg.fairness_penalty,
            two_timescale=True,
            learner="qhmarl",
        ),
    ]


def build_ablation_specs(mec_cfg: MECConfig) -> list[MethodSpec]:
    """Return single-factor ablations of the full QH-MAL configuration.

    Queue observations and the queue reward are deliberately removed in
    separate variants.  The slow-server variant removes both sources of slower
    adaptation: its actor interval becomes one and its learning rate is matched
    to the user actor.  The role-agnostic variant replaces all three actor
    networks with one padded, role-conditioned network while preserving the
    role rewards, critics, update schedule, and physical projections.
    """
    common = {
        "queue_visible": True,
        "queue_in_delay": True,
        "queue_penalty": mec_cfg.queue_penalty,
        "fairness_penalty": mec_cfg.fairness_penalty,
        "two_timescale": True,
        "equal_actor_learning_rates": False,
        "role_specific_actors": True,
        "learner": "qhmarl",
    }
    return [
        MethodSpec("w/o Queue Observations", **{**common, "queue_visible": False}),
        MethodSpec("w/o Queue Penalty", **{**common, "queue_penalty": 0.0}),
        MethodSpec("w/o Fairness Penalty", **{**common, "fairness_penalty": 0.0}),
        MethodSpec(
            "w/o Slower Server Updates",
            **{
                **common,
                "two_timescale": False,
                "equal_actor_learning_rates": True,
            },
        ),
        MethodSpec(
            "w/o Role-Specific Actors",
            **{**common, "role_specific_actors": False},
        ),
        MethodSpec("Full Model", **common),
    ]


def build_smoothed_training_plot_df(history_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = history_df.copy()
    for metric in ["avg_queue_backlog", "avg_waiting_delay"]:
        smoothed_col = f"plot_{metric}"
        plot_df[smoothed_col] = plot_df.groupby("method")[metric].transform(
            lambda s: np.expm1(np.log1p(s.clip(lower=0.0)).ewm(span=6, adjust=False).mean())
        )
    return plot_df


def apply_train_overrides(train_cfg: TrainConfig, config_path: Path | None) -> TrainConfig:
    if config_path is None:
        return train_cfg
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in train_cfg.__dataclass_fields__.values()}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return replace(train_cfg, **filtered)


def run_learning_method(
    method: MethodSpec,
    mec_cfg: MECConfig,
    train_cfg: TrainConfig,
) -> tuple[HeteroOffPolicyTrainer, pd.DataFrame, dict[str, float]]:
    # Reset all stochastic sources so controlled learning methods use the
    # same initialization seed, exploration stream, and environment seed.
    seed_everything(mec_cfg.seed)
    env = MECEnvironment(mec_cfg)
    trainer = HeteroOffPolicyTrainer(env, train_cfg, method)
    print(f"[Stage] Training {method.name} on {trainer.device_label}...", flush=True)
    _, history = trainer.train()
    history_df = pd.DataFrame(history)
    history_df["method"] = method.name
    print(f"[Stage] Final evaluation for {method.name}...", flush=True)
    final_eval_bar = create_progress(
        total=train_cfg.eval_episodes,
        desc=f"{method.name} eval",
        disable=not train_cfg.show_progress,
        leave=False,
        refresh_interval_s=train_cfg.progress_refresh_seconds,
    )
    try:
        metrics = evaluate_policy(trainer, env, train_cfg.eval_episodes, progress_bar=final_eval_bar)
    finally:
        final_eval_bar.close()
    metrics["offline_training_time_s"] = float(trainer.training_time_s)
    return trainer, history_df, metrics


def evaluate_baseline(
    name: str,
    mec_cfg: MECConfig,
    train_cfg: TrainConfig,
    show_progress: bool = True,
    eval_episodes: int | None = None,
) -> dict[str, float]:
    # Match the deterministic evaluation realizations used by evaluate_policy.
    seed_base = mec_cfg.seed + 1000
    summaries = []
    episodes = train_cfg.eval_episodes if eval_episodes is None else int(eval_episodes)
    progress_bar = create_progress(
        total=episodes,
        desc=f"{name} eval",
        disable=not (train_cfg.show_progress and show_progress),
        leave=False,
        refresh_interval_s=train_cfg.progress_refresh_seconds,
    )
    try:
        for episode in range(episodes):
            env = MECEnvironment(mec_cfg).clone_with_seed(seed_base + episode)
            env.reset()
            episode_outcomes = []
            for _ in range(train_cfg.episode_length):
                decision_start = time.perf_counter()
                if name == "Random Offloading":
                    user_actions, prices, allocation_controls = random_policy(env, env.rng)
                elif name == "Greedy Offloading":
                    user_actions, prices, allocation_controls = greedy_policy(env, queue_aware=False)
                elif name == "Fixed-Pricing Queue-Aware Greedy":
                    user_actions, prices, allocation_controls = greedy_policy(env, queue_aware=True)
                elif name == DPP_JPO:
                    user_actions, prices, allocation_controls = dpp_joint_pricing_offloading_policy(env)
                elif name == STACKELBERG_JPO:
                    user_actions, prices, allocation_controls = stackelberg_joint_pricing_offloading_policy(env)
                else:
                    raise ValueError(name)
                decision_time_ms = 1.0e3 * (time.perf_counter() - decision_start)
                outcome = env.step(
                    user_actions,
                    prices,
                    allocation_controls,
                    MethodSpec(
                        name=name,
                        queue_visible=True,
                        queue_in_delay=True,
                        queue_penalty=mec_cfg.queue_penalty,
                        fairness_penalty=mec_cfg.fairness_penalty,
                    ),
                )
                timed_metrics = dict(outcome["metrics"])
                timed_metrics["decision_time_ms"] = float(decision_time_ms)
                outcome = dict(outcome)
                outcome["metrics"] = timed_metrics
                episode_outcomes.append(outcome)
            summaries.append(aggregate_episode_outcomes(episode_outcomes))
            progress_bar.update(1)
    finally:
        progress_bar.close()
    aggregate = {key: float(pd.Series([m[key] for m in summaries]).mean()) for key in summaries[0]}
    aggregate["offline_training_time_s"] = 0.0
    return aggregate


def build_overall_table(
    proposed_metrics: dict[str, float],
    learning_baselines: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    ordered = [(name, baselines[name]) for name in BASELINE_NAMES]
    ordered.extend((name, learning_baselines[name]) for name in LEARNING_BASELINE_NAMES)
    ordered.append(("Proposed Method", proposed_metrics))
    for name, metrics in ordered:
        rows.append(
            {
                "Method": name,
                "Avg Delay": metrics["avg_delay"],
                "P95 Delay": metrics["p95_delay"],
                "Uplink Delay": metrics["avg_uplink_delay"],
                "Population Wait Delay": metrics["avg_waiting_delay_all_users"],
                "Execution Delay": metrics["avg_execution_delay"],
                "Downlink Delay": metrics["avg_downlink_delay"],
                "Energy": metrics["avg_energy"],
                "Payment": metrics["avg_payment"],
                "Conditional Payment": metrics["avg_payment_if_offloaded"],
                "Average Price": metrics["avg_server_price"],
                "Server Return": metrics["avg_profit"],
                "Fairness": metrics["fairness"],
                "User Service Fairness": metrics["user_service_fairness"],
                "User Delay Fairness": metrics["user_delay_fairness"],
                "P05 User Service Rate": metrics["p05_user_service_rate"],
                "Worst-User Avg Delay": metrics["worst_user_avg_delay"],
                "Queue Backlog": metrics["avg_queue_backlog"],
                "Wait Delay": metrics["avg_waiting_delay"],
                "Violation Ratio": metrics["violation_ratio"],
                "Requested Offload Ratio": metrics["requested_offload_ratio"],
                "Edge Participation Ratio": metrics["edge_participation_ratio"],
                "Admission Acceptance Ratio": metrics["admission_acceptance_ratio"],
                "Max Buffer Occupancy": metrics["max_queue_occupancy"],
                "Local Fallback Ratio": metrics["fallback_ratio"],
                "Overflow Count": metrics["overflow_count"],
                "Hard Constraint Violation Rate": metrics["hard_constraint_violation_rate"],
                "Projection Intervention Ratio": metrics["projection_intervention_ratio"],
                "Delay Decomposition Error": metrics["delay_decomposition_error"],
                "CPU Utilization": metrics["avg_cpu_utilization"],
                "Decision Time (ms)": metrics["decision_time_ms"],
                "Offline Training Time (s)": metrics.get("offline_training_time_s", 0.0),
            }
        )
    return pd.DataFrame(rows)


def sweep_arrival_rate(
    learning_trainers: dict[str, HeteroOffPolicyTrainer],
    train_cfg: TrainConfig,
    base_cfg: MECConfig,
) -> pd.DataFrame:
    rows = []
    scales = [0.8, 1.0, 1.2, 1.4]
    eval_episodes = max(10, train_cfg.eval_episodes // 2)
    progress_bar = create_progress(
        total=len(scales) * len(learning_trainers) * eval_episodes,
        desc="Arrival sweep",
        disable=not train_cfg.show_progress,
        leave=False,
        refresh_interval_s=train_cfg.progress_refresh_seconds,
    )
    try:
        for scale in scales:
            cfg = replace(base_cfg, arrival_scale=scale)
            for label in [name for name in METHOD_ORDER if name in learning_trainers]:
                method_trainer = learning_trainers[label]
                progress_bar.set_postfix({"arrival": f"{scale:.1f}", "method": label}, refresh=False)
                env = MECEnvironment(cfg)
                metrics = evaluate_policy(method_trainer, env, eval_episodes, progress_bar=progress_bar)
                rows.append({"arrival_scale": scale, "method": label, **metrics})
    finally:
        progress_bar.close()
    return pd.DataFrame(rows)


def sweep_penalty(
    base_cfg: MECConfig,
    base_train_cfg: TrainConfig,
    which: str,
) -> pd.DataFrame:
    rows = []
    sweep_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    sweep_train_cfg = replace(
        base_train_cfg,
        episodes=max(24, base_train_cfg.episodes // 4),
        eval_episodes=max(6, min(10, base_train_cfg.eval_episodes // 2)),
    )
    for value in sweep_values:
        cfg = replace(base_cfg, queue_penalty=value) if which == "queue" else replace(base_cfg, fairness_penalty=value)
        spec = MethodSpec(
            name="Proposed Method",
            queue_visible=True,
            queue_in_delay=True,
            queue_penalty=cfg.queue_penalty,
            fairness_penalty=cfg.fairness_penalty,
            two_timescale=True,
        )
        _, _, metrics = run_learning_method(spec, cfg, sweep_train_cfg)
        rows.append({("lambda_q" if which == "queue" else "lambda_f"): value, **metrics})
    return pd.DataFrame(rows)


def build_ablation_table(
    base_cfg: MECConfig,
    train_cfg: TrainConfig,
    full_metrics: dict[str, float] | None = None,
    full_actor_parameters: int | None = None,
) -> pd.DataFrame:
    rows = []
    for spec in build_ablation_specs(base_cfg):
        if spec.name == "Full Model" and full_metrics is not None:
            rows.append(
                {
                    "Variant": spec.name,
                    "Actor Parameters": full_actor_parameters,
                    **full_metrics,
                }
            )
            continue
        trainer, _, metrics = run_learning_method(spec, base_cfg, train_cfg)
        rows.append(
            {
                "Variant": spec.name,
                "Actor Parameters": trainer.actor_parameter_count(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_full_arrival_comparison(
    base_cfg: MECConfig,
    train_cfg: TrainConfig,
    baseline_names: list[str],
    learning_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    scales = [0.8, 1.0, 1.2, 1.4]
    progress_bar = create_progress(
        total=len(baseline_names) * len(scales),
        desc="Baseline arrival comparison",
        disable=not train_cfg.show_progress,
        leave=False,
        refresh_interval_s=train_cfg.progress_refresh_seconds,
    )
    try:
        for method in baseline_names:
            for scale in scales:
                progress_bar.set_postfix({"arrival": f"{scale:.1f}", "method": method}, refresh=False)
                cfg = replace(base_cfg, arrival_scale=scale)
                metrics_scale = evaluate_baseline(method, cfg, train_cfg, show_progress=False)
                rows.append({"arrival_scale": scale, "method": method, **metrics_scale})
                progress_bar.update(1)
    finally:
        progress_bar.close()
    rows.extend(learning_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_distribution_shift_comparison(
    learning_trainers: dict[str, HeteroOffPolicyTrainer],
    train_cfg: TrainConfig,
    base_cfg: MECConfig,
) -> pd.DataFrame:
    """Evaluate fixed trained policies on distributions absent from training."""

    scenarios = [
        ("ID uniform", 1.2, "uniform", "uniform"),
        ("Unseen high load", 1.8, "uniform", "uniform"),
        ("Unseen bursty arrivals", 1.2, "bursty_lognormal", "uniform"),
        ("Unseen low-goodput channels", 1.2, "uniform", "beta_low"),
        ("Joint unseen shift", 1.8, "bursty_lognormal", "beta_low"),
    ]
    eval_episodes = max(10, train_cfg.eval_episodes // 2)
    rows: list[dict[str, object]] = []
    for scenario, scale, arrival_distribution, channel_distribution in scenarios:
        cfg = replace(
            base_cfg,
            arrival_scale=scale,
            arrival_distribution=arrival_distribution,
            channel_distribution=channel_distribution,
        )
        for method in METHOD_ORDER:
            if method in learning_trainers:
                metrics = evaluate_policy(
                    learning_trainers[method],
                    MECEnvironment(cfg),
                    eval_episodes,
                )
            elif method in BASELINE_NAMES:
                metrics = evaluate_baseline(
                    method,
                    cfg,
                    train_cfg,
                    show_progress=False,
                    eval_episodes=eval_episodes,
                )
            else:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "arrival_scale": scale,
                    "arrival_distribution": arrival_distribution,
                    "channel_distribution": channel_distribution,
                    "method": method,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _baseline_action(
    name: str,
    env: MECEnvironment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "Random Offloading":
        return random_policy(env, env.rng)
    if name == "Greedy Offloading":
        return greedy_policy(env, queue_aware=False)
    if name == "Fixed-Pricing Queue-Aware Greedy":
        return greedy_policy(env, queue_aware=True)
    if name == DPP_JPO:
        return dpp_joint_pricing_offloading_policy(env)
    if name == STACKELBERG_JPO:
        return stackelberg_joint_pricing_offloading_policy(env)
    raise ValueError(name)


def evaluate_overload_stability(
    learning_trainers: dict[str, HeteroOffPolicyTrainer],
    train_cfg: TrainConfig,
    base_cfg: MECConfig,
    horizon: int,
    arrival_scale: float = 1.6,
) -> pd.DataFrame:
    """Run uninterrupted overload trajectories and diagnose saturation.

    Finite-buffer projection makes divergence impossible by construction, so
    this protocol reports the operational price of overload (capacity hits,
    fallback, participation, and queue drift) rather than mislabeling bounded
    queues as a learned Lyapunov-stability theorem.
    """

    cfg = replace(
        base_cfg,
        arrival_scale=arrival_scale,
        arrival_distribution="uniform",
        channel_distribution="uniform",
    )
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        trainer = learning_trainers.get(method)
        if trainer is None and method not in BASELINE_NAMES:
            continue
        env = MECEnvironment(cfg).clone_with_seed(cfg.seed + 50_000)
        env.reset()
        outcomes: list[dict[str, object]] = []
        queue_occupancy: list[float] = []
        capacity_hits: list[float] = []
        for _ in range(horizon):
            if trainer is not None and trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            started = time.perf_counter()
            if trainer is not None:
                user_actions, prices, allocation_controls = trainer.act_deterministically(env)
                spec = trainer.spec
            else:
                user_actions, prices, allocation_controls = _baseline_action(method, env)
                spec = MethodSpec(
                    name=method,
                    queue_visible=True,
                    queue_in_delay=True,
                    queue_penalty=cfg.queue_penalty,
                    fairness_penalty=cfg.fairness_penalty,
                )
            if trainer is not None and trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            decision_time_ms = 1.0e3 * (time.perf_counter() - started)
            outcome = env.step(user_actions, prices, allocation_controls, spec)
            timed_metrics = dict(outcome["metrics"])
            timed_metrics["decision_time_ms"] = float(decision_time_ms)
            recorded = dict(outcome)
            recorded["metrics"] = timed_metrics
            outcomes.append(recorded)
            normalized = float(np.mean(env.queues / cfg.queue_capacity_cycles))
            queue_occupancy.append(normalized)
            capacity_hits.append(float(timed_metrics["max_queue_occupancy"] >= 0.999))

        aggregate = aggregate_episode_outcomes(outcomes)
        queue_array = np.asarray(queue_occupancy, dtype=np.float64)
        split = max(horizon // 2, 1)
        x = np.arange(horizon, dtype=np.float64)
        slope = float(np.polyfit(x, queue_array, 1)[0] * 1000.0) if horizon > 1 else 0.0
        rows.append(
            {
                "method": method,
                "horizon_slots": int(horizon),
                "arrival_scale": float(arrival_scale),
                "queue_occupancy_first_half": float(queue_array[:split].mean()),
                "queue_occupancy_second_half": float(queue_array[split:].mean())
                if split < horizon
                else float(queue_array[:split].mean()),
                "queue_occupancy_slope_per_1000_slots": slope,
                "capacity_hit_ratio": float(np.mean(capacity_hits)),
                **aggregate,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue-aware MEC paper reproduction.")
    parser.add_argument("--steps", type=int, default=220, help="Training episodes for the main methods.")
    parser.add_argument("--quick", action="store_true", help="Run a lightweight smoke test.")
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip sensitivity sweeps to save time.")
    parser.add_argument("--skip-arrival-sweep", action="store_true", help="Skip the arrival-intensity experiment (useful only for a smoke run).")
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run the component ablation even when --skip-sweeps is set.",
    )
    parser.add_argument("--train-config", type=Path, default=None, help="Optional JSON file with TrainConfig overrides.")
    parser.add_argument("--device", type=str, default=None, help="Override compute device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed for environment generation and training.")
    parser.add_argument(
        "--stress-horizon",
        type=int,
        default=10_000,
        help="Uninterrupted slots used by the overload-stability protocol.",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip the long-horizon overload protocol.",
    )
    parser.add_argument(
        "--run-ood-evaluation",
        action="store_true",
        help="Run the optional distribution-shift extension; it is not part of the final paper protocol.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    paths = PathConfig(root=root)
    ensure_dir(paths.results_dir)
    ensure_dir(paths.figures_dir)
    ensure_dir(paths.checkpoints_dir)

    seed_everything(args.seed)
    mec_cfg = replace(MECConfig(), seed=args.seed)
    train_cfg = TrainConfig(episodes=args.steps)
    train_cfg = apply_train_overrides(train_cfg, args.train_config)
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    if args.quick:
        train_cfg = replace(train_cfg, episodes=min(train_cfg.episodes, 30), eval_episodes=min(train_cfg.eval_episodes, 6), episode_length=min(train_cfg.episode_length, 20), warmup_steps=min(train_cfg.warmup_steps, 400))
        args.stress_horizon = min(args.stress_horizon, 200)
    resolved_device, device_label = resolve_torch_device(train_cfg.device)
    train_cfg = replace(train_cfg, device=str(resolved_device))
    print(f"[Config] Seed: {mec_cfg.seed}", flush=True)
    print(f"[Config] Compute device: {device_label}", flush=True)

    learning_runs: dict[str, tuple[HeteroOffPolicyTrainer, pd.DataFrame, dict[str, float]]] = {}
    for spec in build_learning_specs(mec_cfg):
        learning_runs[spec.name] = run_learning_method(spec, mec_cfg, train_cfg)
    proposed_trainer, _, proposed_metrics = learning_runs["Proposed Method"]
    learning_baseline_metrics = {
        name: learning_runs[name][2]
        for name in LEARNING_BASELINE_NAMES
    }

    baseline_names = BASELINE_NAMES
    print("[Stage] Evaluating heuristic and model-based baselines...", flush=True)
    baseline_metrics = {name: evaluate_baseline(name, mec_cfg, train_cfg) for name in baseline_names}

    overall_df = build_overall_table(proposed_metrics, learning_baseline_metrics, baseline_metrics)
    save_dataframe(overall_df, paths.results_dir / "table_iv.csv")
    save_dataframe(overall_df, paths.results_dir / "overall_performance_comparison.csv")

    history_df = pd.concat([learning_runs[name][1] for name in METHOD_ORDER if name in learning_runs], ignore_index=True)
    save_dataframe(history_df, paths.results_dir / "training_history.csv")
    plot_history_df = build_smoothed_training_plot_df(history_df)
    plot_training_curve(
        plot_history_df,
        "plot_avg_queue_backlog",
        paths.figures_dir / "average_queue_backlog_vs_episode.pdf",
        "Average Queue Backlog vs Episode",
        "Average queue backlog",
        baseline_lines={name: metrics["avg_queue_backlog"] for name, metrics in baseline_metrics.items()},
        series_order=[name for name in METHOD_ORDER if name in learning_runs],
    )
    plot_training_curve(
        plot_history_df,
        "plot_avg_waiting_delay",
        paths.figures_dir / "average_waiting_delay_vs_episode.pdf",
        "Average Waiting Delay vs Episode",
        "Average waiting delay",
        baseline_lines={name: metrics["avg_waiting_delay"] for name, metrics in baseline_metrics.items()},
        series_order=[name for name in METHOD_ORDER if name in learning_runs],
    )

    learning_trainers = {name: learning_runs[name][0] for name in learning_runs}
    if not args.skip_arrival_sweep:
        print("[Stage] Running arrival-rate comparison sweep...", flush=True)
        arrival_df = sweep_arrival_rate(learning_trainers, train_cfg, mec_cfg)
        save_dataframe(arrival_df, paths.results_dir / "arrival_sensitivity.csv")
        comparison_df = build_full_arrival_comparison(mec_cfg, train_cfg, baseline_names, arrival_df)
        save_dataframe(comparison_df, paths.results_dir / "metrics_vs_arrival_rate.csv")
        plot_metric_vs_x(
            comparison_df,
            "arrival_scale",
            "p95_delay",
            "method",
            paths.figures_dir / "p95_delay_vs_arrival_rate.pdf",
            "P95 Delay under Different Arrival Intensity",
            series_order=METHOD_ORDER,
        )
        plot_arrival_dashboard(
            comparison_df,
            paths.figures_dir / "metrics_vs_arrival_rate.pdf",
            "Overall Performance under Different Arrival Intensity",
            series_order=METHOD_ORDER,
        )

    if args.run_ood_evaluation:
        print("[Stage] Evaluating unseen arrival/channel distributions...", flush=True)
        distribution_shift_df = build_distribution_shift_comparison(
            learning_trainers,
            train_cfg,
            mec_cfg,
        )
        save_dataframe(
            distribution_shift_df,
            paths.results_dir / "distribution_shift_generalization.csv",
        )

    if not args.skip_stress:
        print("[Stage] Running uninterrupted overload-stability protocol...", flush=True)
        overload_df = evaluate_overload_stability(
            learning_trainers,
            train_cfg,
            mec_cfg,
            horizon=max(int(args.stress_horizon), 1),
        )
        save_dataframe(overload_df, paths.results_dir / "overload_stability.csv")

    if not args.skip_sweeps:
        print("[Stage] Running penalty sweeps...", flush=True)
        lambda_q_df = sweep_penalty(mec_cfg, train_cfg, "queue")
        save_dataframe(lambda_q_df, paths.results_dir / "lambda_q_sensitivity.csv")
        plot_penalty_dashboard(
            lambda_q_df,
            "lambda_q",
            paths.figures_dir / "metrics_vs_lambda_q.pdf",
            "Impact of Queue Penalty Coefficient",
            "p95_delay",
            "P95 Delay",
            "violation_ratio",
            "Violation Ratio",
        )

        lambda_f_df = sweep_penalty(mec_cfg, train_cfg, "fairness")
        save_dataframe(lambda_f_df, paths.results_dir / "lambda_f_sensitivity.csv")
        plot_penalty_dashboard(
            lambda_f_df,
            "lambda_f",
            paths.figures_dir / "metrics_vs_lambda_f.pdf",
            "Impact of Fairness Penalty Coefficient",
            "fairness",
            "Fairness",
            "avg_profit",
            "Server Return",
        )

    if not args.skip_sweeps or args.run_ablation:
        print("[Stage] Running component-wise ablations...", flush=True)
        ablation_df = build_ablation_table(
            mec_cfg,
            train_cfg,
            full_metrics=proposed_metrics,
            full_actor_parameters=proposed_trainer.actor_parameter_count(),
        )
        save_dataframe(ablation_df, paths.results_dir / "ablation.csv")

    torch.save(
        {
            "user_actor": proposed_trainer.bundle.user_actor.state_dict(),
            "price_actor": proposed_trainer.bundle.price_actor.state_dict(),
            "alloc_actor": proposed_trainer.bundle.alloc_actor.state_dict(),
            "user_critic": proposed_trainer.user_critic.state_dict(),
            "server_critic": proposed_trainer.server_critic.state_dict(),
            "best_episode": getattr(proposed_trainer, "best_episode", train_cfg.episodes),
            "best_selection_score": getattr(proposed_trainer, "best_validation_score", float("nan")),
        },
        paths.checkpoints_dir / "proposed_method.pt",
    )

    save_json(
        paths.results_dir / "summary.json",
        {
            "seed": mec_cfg.seed,
            "device": train_cfg.device,
            "train_config_path": str(args.train_config.resolve()) if args.train_config is not None else None,
            "proposed": proposed_metrics,
            "learning_baselines": learning_baseline_metrics,
            "baselines": baseline_metrics,
            "proposed_best_episode": getattr(proposed_trainer, "best_episode", train_cfg.episodes),
            "proposed_best_selection_score": getattr(proposed_trainer, "best_validation_score", float("nan")),
            "learning_baseline_best_episodes": {
                name: getattr(learning_runs[name][0], "best_episode", train_cfg.episodes)
                for name in LEARNING_BASELINE_NAMES
            },
            "learning_baseline_best_selection_scores": {
                name: getattr(learning_runs[name][0], "best_validation_score", float("nan"))
                for name in LEARNING_BASELINE_NAMES
            },
            "table_iv_path": str(paths.results_dir / "table_iv.csv"),
            "overall_performance_comparison_path": str(paths.results_dir / "overall_performance_comparison.csv"),
            "distribution_shift_generalization_path": (
                str(paths.results_dir / "distribution_shift_generalization.csv")
                if args.run_ood_evaluation
                else None
            ),
            "overload_stability_path": (
                str(paths.results_dir / "overload_stability.csv")
                if not args.skip_stress
                else None
            ),
        },
    )
    print("Saved results to", paths.results_dir)
    print("Saved figures to", paths.figures_dir)
    print(overall_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
