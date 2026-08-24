from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_mec.statistics import (
    build_mean_ci_table,
    build_paired_inference_table,
    mean_t_interval,
)

RESULT_FILES = [
    Path('results/summary.json'),
    Path('results/overall_performance_comparison.csv'),
    Path('results/table_iv.csv'),
    Path('results/metrics_vs_arrival_rate.csv'),
    Path('results/training_history.csv'),
    Path('results/distribution_shift_generalization.csv'),
    Path('results/overload_stability.csv'),
]
OPTIONAL_RESULT_FILES = [
    Path('results/arrival_sensitivity.csv'),
    Path('results/lambda_q_sensitivity.csv'),
    Path('results/lambda_f_sensitivity.csv'),
    Path('results/ablation.csv'),
]
FIGURE_FILES = [
    Path('figures/average_queue_backlog_vs_episode.pdf'),
    Path('figures/average_waiting_delay_vs_episode.pdf'),
    Path('figures/p95_delay_vs_arrival_rate.pdf'),
    Path('figures/metrics_vs_arrival_rate.pdf'),
    Path('figures/metrics_vs_lambda_q.pdf'),
    Path('figures/metrics_vs_lambda_f.pdf'),
]
CHECKPOINT_FILES = [
    Path('artifacts/checkpoints/proposed_method.pt'),
]
LOWER_IS_BETTER = {
    'Avg Delay',
    'P95 Delay',
    'Uplink Delay',
    'Population Wait Delay',
    'Execution Delay',
    'Downlink Delay',
    'Energy',
    'Payment',
    'Conditional Payment',
    'Average Price',
    'Queue Backlog',
    'Wait Delay',
    'Violation Ratio',
    'Max Buffer Occupancy',
    'Local Fallback Ratio',
    'Overflow Count',
    'Hard Constraint Violation Rate',
    'Worst-User Avg Delay',
    'Projection Intervention Ratio',
}
HIGHER_IS_BETTER = {
    'Server Return',
    'Fairness',
    'User Service Fairness',
    'User Delay Fairness',
    'Requested Offload Ratio',
    'Edge Participation Ratio',
    'Admission Acceptance Ratio',
}
METRICS = [
    'Avg Delay',
    'P95 Delay',
    'Energy',
    'Payment',
    'Server Return',
    'Fairness',
    'Queue Backlog',
    'Wait Delay',
    'Violation Ratio',
    'Max Buffer Occupancy',
    'Local Fallback Ratio',
    'Overflow Count',
    'User Service Fairness',
    'User Delay Fairness',
    'Worst-User Avg Delay',
    'Hard Constraint Violation Rate',
]
TRADEOFF_METRICS = [
    'Uplink Delay',
    'Population Wait Delay',
    'Execution Delay',
    'Downlink Delay',
    'Conditional Payment',
    'Average Price',
    'Requested Offload Ratio',
    'Edge Participation Ratio',
    'Admission Acceptance Ratio',
    'Projection Intervention Ratio',
]
COMPUTE_METRICS = [
    'CPU Utilization',
    'Decision Time (ms)',
    'Offline Training Time (s)',
]
AGGREGATE_METRICS = METRICS + TRADEOFF_METRICS + COMPUTE_METRICS
MAIN_INFERENCE_METRICS = METRICS + TRADEOFF_METRICS
GENERALIZATION_INFERENCE_METRICS = [
    'avg_delay',
    'p95_delay',
    'avg_profit',
    'avg_queue_backlog',
    'violation_ratio',
    'edge_participation_ratio',
    'fallback_ratio',
    'user_service_fairness',
    'user_delay_fairness',
    'worst_user_avg_delay',
    'hard_constraint_violation_rate',
]
GENERALIZATION_LOWER_IS_BETTER = {
    'avg_delay',
    'p95_delay',
    'avg_queue_backlog',
    'violation_ratio',
    'fallback_ratio',
    'worst_user_avg_delay',
    'hard_constraint_violation_rate',
}
OVERLOAD_INFERENCE_METRICS = GENERALIZATION_INFERENCE_METRICS + [
    'queue_occupancy_second_half',
    'queue_occupancy_slope_per_1000_slots',
    'capacity_hit_ratio',
]
OVERLOAD_LOWER_IS_BETTER = GENERALIZATION_LOWER_IS_BETTER | {
    'queue_occupancy_second_half',
    'queue_occupancy_slope_per_1000_slots',
    'capacity_hit_ratio',
}
DEFAULT_SEEDS = [7, 11, 19, 23, 31]
DEFAULT_ABLATION_SEEDS = [7, 11, 19, 23, 31]
ABLATION_METRICS = [
    ('avg_delay', 'Avg Delay', False),
    ('p95_delay', 'P95 Delay', False),
    ('avg_profit', 'Server Return', True),
    ('fairness', 'Fairness', True),
    ('avg_queue_backlog', 'Queue Backlog', False),
    ('avg_waiting_delay', 'Wait Delay', False),
    ('violation_ratio', 'Violation Ratio', False),
]


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def clear_previous_outputs() -> None:
    for rel_path in RESULT_FILES + OPTIONAL_RESULT_FILES + FIGURE_FILES + CHECKPOINT_FILES:
        if rel_path.exists():
            rel_path.unlink()


def archive_run(run_dir: Path, config_path: Path, seed: int, elapsed_s: float, command: list[str], returncode: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / config_path.name)
    for rel_path in RESULT_FILES + OPTIONAL_RESULT_FILES + FIGURE_FILES + CHECKPOINT_FILES:
        copy_if_exists(rel_path, run_dir / rel_path)
    metadata = {
        'config': str(config_path),
        'seed': seed,
        'elapsed_seconds': elapsed_s,
        'command': command,
        'returncode': returncode,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (run_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def load_overall_table(run_dir: Path, seed: int) -> pd.DataFrame:
    table = pd.read_csv(run_dir / 'results' / 'overall_performance_comparison.csv')
    table.insert(0, 'Seed', seed)
    return table


def load_ablation_table(run_dir: Path, seed: int) -> pd.DataFrame:
    table = pd.read_csv(run_dir / 'results' / 'ablation.csv')
    table.insert(0, 'Seed', seed)
    return table


def load_optional_table(
    run_dir: Path,
    relative_path: Path,
    seed: int,
) -> pd.DataFrame | None:
    path = run_dir / relative_path
    if not path.exists():
        return None
    table = pd.read_csv(path)
    table.insert(0, 'Seed', seed)
    return table


def build_grouped_mean_ci_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate every numeric metric by seed for curves and stress tables."""

    excluded = set(group_columns) | {'Seed'}
    numeric_columns = [
        column
        for column in frame.select_dtypes(include='number').columns
        if column not in excluded
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = dict(zip(group_columns, key_values))
        row['Seed Count'] = int(group['Seed'].nunique())
        row['Seeds'] = ','.join(str(int(seed)) for seed in sorted(group['Seed'].unique()))
        for metric in numeric_columns:
            mean, std, lower, upper = mean_t_interval(group[metric], confidence)
            row[f'{metric} Mean'] = mean
            row[f'{metric} Std'] = std
            row[f'{metric} {confidence:.0%} CI Lower'] = lower
            row[f'{metric} {confidence:.0%} CI Upper'] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def build_stratified_paired_inference_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
    lower_is_better: set[str],
    method_column: str = 'method',
) -> pd.DataFrame:
    """Run paired seed inference separately within each declared scenario.

    Calling ``build_paired_inference_table`` inside each stratum keeps the
    Holm family local to one baseline and one evaluation scenario. Episodes
    and slots therefore never become independent replicates.
    """

    rows: list[pd.DataFrame] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        available_metrics = [metric for metric in metrics if metric in group.columns]
        if not available_metrics:
            continue
        normalized = group.rename(columns={method_column: 'Method'})
        inference = build_paired_inference_table(
            normalized,
            available_metrics,
            lower_is_better,
        )
        if inference.empty:
            continue
        for column, value in reversed(list(zip(group_columns, key_values))):
            inference.insert(0, column, value)
        rows.append(inference)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_training_curves_with_ci(
    summary: pd.DataFrame,
    output_base: Path,
) -> None:
    metric_specs = [
        ('eval_avg_delay', 'Average Delay'),
        ('eval_p95_delay', 'P95 Delay'),
        ('eval_avg_profit', 'Server Return'),
        ('eval_violation_ratio', 'Latency-Violation Ratio'),
    ]
    palette = {
        'Queue-Unaware QH-MAL': '#c53030',
        'Queue-Aware MADDPG': '#8c564b',
        'Queue-Aware MATD3': '#d53f8c',
        'Proposed Method': '#2f855a',
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        mean_column = f'{metric} Mean'
        lower_column = f'{metric} 95% CI Lower'
        upper_column = f'{metric} 95% CI Upper'
        if mean_column not in summary.columns:
            ax.set_visible(False)
            continue
        for method in palette:
            subset = summary[summary['method'] == method].sort_values('episode')
            if subset.empty:
                continue
            x = subset['episode'].to_numpy(dtype=float)
            mean = subset[mean_column].to_numpy(dtype=float)
            lower = subset[lower_column].to_numpy(dtype=float)
            upper = subset[upper_column].to_numpy(dtype=float)
            color = palette[method]
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.5 if method == 'Proposed Method' else 2.0,
                label=method,
            )
            ax.fill_between(x, lower, upper, color=color, alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel('Training Episode')
        ax.set_ylabel(title)
        ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.99))
    fig.savefig(output_base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.08)
    fig.savefig(output_base.with_suffix('.png'), dpi=240, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def build_ablation_relative_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired per-seed gains of the full model over each ablation."""
    rows = []
    for seed, seed_df in df.groupby('Seed', sort=True):
        indexed = seed_df.set_index('Variant')
        if 'Full Model' not in indexed.index:
            continue
        full = indexed.loc['Full Model']
        for variant, ablated in indexed.drop(index='Full Model').iterrows():
            row = {'Seed': int(seed), 'Variant': variant}
            for raw, label, higher_is_better in ABLATION_METRICS:
                full_value = float(full[raw])
                ablated_value = float(ablated[raw])
                denominator = abs(ablated_value) + 1.0e-9
                if higher_is_better:
                    gain = (full_value - ablated_value) / denominator * 100.0
                else:
                    gain = (ablated_value - full_value) / denominator * 100.0
                row[f'{label} Relative Gain (%)'] = gain
            rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_gain_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gain_columns = [column for column in df.columns if column.endswith('Relative Gain (%)')]
    for variant, group in df.groupby('Variant', sort=False):
        row = {'Variant': variant, 'Seeds': int(group['Seed'].nunique())}
        for column in gain_columns:
            row[f'{column} Mean'] = float(group[column].mean())
            row[f'{column} Std'] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby('Method', sort=False):
        row = {'Method': method}
        for metric in AGGREGATE_METRICS:
            row[f'{metric} Mean'] = float(group[metric].mean())
            row[f'{metric} Std'] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            row[f'{metric} Mean+/-Std'] = f"{group[metric].mean():.4f} +/- {(group[metric].std(ddof=1) if len(group) > 1 else 0.0):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_win_count_table(df: pd.DataFrame) -> pd.DataFrame:
    proposed = df[df['Method'] == 'Proposed Method'].set_index('Seed')
    baselines = [
        'Random Offloading',
        'Greedy Offloading',
        'Fixed-Pricing Queue-Aware Greedy',
        'DPP Joint Pricing-Offloading',
        'Stackelberg Joint Pricing-Offloading',
        'Queue-Unaware QH-MAL',
        'Queue-Aware MADDPG',
        'Queue-Aware MATD3',
    ]
    rows = []
    for baseline in baselines:
        baseline_df = df[df['Method'] == baseline].set_index('Seed')
        shared = proposed.index.intersection(baseline_df.index)
        if shared.empty:
            continue
        row = {'Compared Against': baseline, 'Seeds': int(len(shared))}
        for metric in METRICS:
            prop_vals = proposed.loc[shared, metric]
            base_vals = baseline_df.loc[shared, metric]
            if metric in LOWER_IS_BETTER:
                wins = int((prop_vals < base_vals).sum())
            else:
                wins = int((prop_vals > base_vals).sum())
            ties = int((prop_vals == base_vals).sum())
            row[f'{metric} Wins'] = wins
            row[f'{metric} Ties'] = ties
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    proposed = df[df['Method'] == 'Proposed Method'].set_index('Seed')
    queue_unaware = df[df['Method'] == 'Queue-Unaware QH-MAL'].set_index('Seed')
    shared = proposed.index.intersection(queue_unaware.index)
    rows = []
    for seed in shared:
        row = {'Seed': int(seed)}
        for metric in METRICS + TRADEOFF_METRICS:
            row[f'{metric} Delta vs Queue-Unaware'] = float(proposed.loc[seed, metric] - queue_unaware.loc[seed, metric])
        rows.append(row)
    return pd.DataFrame(rows)


def append_seed_metrics(leaderboard_rows: list[dict[str, object]], run_dir: Path, seed: int, elapsed_s: float, returncode: int) -> None:
    overall = pd.read_csv(run_dir / 'results' / 'overall_performance_comparison.csv')
    proposed = overall.loc[overall['Method'] == 'Proposed Method'].iloc[0]
    row = {'seed': seed, 'elapsed_seconds': round(elapsed_s, 2), 'returncode': returncode}
    for metric in AGGREGATE_METRICS:
        row[metric] = proposed[metric]
    leaderboard_rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a single config across multiple random seeds and aggregate mean/std tables.')
    parser.add_argument('--config', default='configs/paper_results.json', help='Training config JSON to evaluate across seeds.')
    parser.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS, help='Explicit seed list. Defaults to the five independent seeds reported in the paper.')
    parser.add_argument(
        '--ablation-seeds',
        nargs='*',
        type=int,
        default=DEFAULT_ABLATION_SEEDS,
        help='Subset that runs the expensive component ablation; defaults to the five seeds fixed before evaluation.',
    )
    parser.add_argument('--device', default='cuda:0', help='Device override passed to run.py, e.g. cuda:0 or cpu.')
    parser.add_argument('--output-root', default='experiments/multiseed', help='Directory for archived runs and aggregate tables.')
    parser.add_argument('--run-sweeps', action='store_true', help='Also run lambda/ablation sweeps. Default is main experiment only.')
    parser.add_argument(
        '--run-ablation',
        action='store_true',
        help='Run and aggregate the component ablation without repeating penalty sweeps.',
    )
    parser.add_argument('--keep-going', action='store_true', help='Continue remaining seeds if one run fails.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them.')
    args = parser.parse_args()

    root = Path.cwd()
    config_path = (root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    run_root = root / args.output_root / config_path.stem
    run_root.mkdir(parents=True, exist_ok=True)

    overall_tables: list[pd.DataFrame] = []
    ablation_tables: list[pd.DataFrame] = []
    training_tables: list[pd.DataFrame] = []
    generalization_tables: list[pd.DataFrame] = []
    overload_tables: list[pd.DataFrame] = []
    leaderboard_rows: list[dict[str, object]] = []

    for index, seed in enumerate(args.seeds, start=1):
        command = [sys.executable, 'run.py', '--train-config', str(config_path), '--device', args.device, '--seed', str(seed)]
        # The paper's long-horizon stress test is a separate fixed-policy,
        # seed-7 experiment handled by run_buffer_stress_experiment.py.
        command.append('--skip-stress')
        if not args.run_sweeps:
            command.append('--skip-sweeps')
        if args.run_ablation and not args.run_sweeps and seed in set(args.ablation_seeds):
            command.append('--run-ablation')
        label = f'[{index}/{len(args.seeds)}] seed={seed}'
        print(f"{label} -> {' '.join(command)}", flush=True)
        if args.dry_run:
            continue

        clear_previous_outputs()
        started = time.time()
        completed = subprocess.run(command, cwd=root)
        elapsed_s = time.time() - started

        seed_dir = run_root / f'seed_{seed}'
        archive_run(seed_dir, config_path, seed, elapsed_s, command, completed.returncode)
        if (seed_dir / 'results' / 'overall_performance_comparison.csv').exists():
            overall_tables.append(load_overall_table(seed_dir, seed))
            append_seed_metrics(leaderboard_rows, seed_dir, seed, elapsed_s, completed.returncode)
        if (seed_dir / 'results' / 'ablation.csv').exists():
            ablation_tables.append(load_ablation_table(seed_dir, seed))
        training_table = load_optional_table(
            seed_dir,
            Path('results/training_history.csv'),
            seed,
        )
        if training_table is not None:
            training_tables.append(training_table)
        generalization_table = load_optional_table(
            seed_dir,
            Path('results/distribution_shift_generalization.csv'),
            seed,
        )
        if generalization_table is not None:
            generalization_tables.append(generalization_table)
        overload_table = load_optional_table(
            seed_dir,
            Path('results/overload_stability.csv'),
            seed,
        )
        if overload_table is not None:
            overload_tables.append(overload_table)

        if completed.returncode != 0:
            print(f'{label} failed after {elapsed_s / 3600:.2f}h. Outputs archived to {seed_dir}.', flush=True)
            if not args.keep_going:
                break
            continue
        print(f'{label} finished in {elapsed_s / 3600:.2f}h. Outputs archived to {seed_dir}.', flush=True)

    if not overall_tables:
        return

    all_runs_df = pd.concat(overall_tables, ignore_index=True)
    all_runs_df.to_csv(run_root / 'all_runs_overall.csv', index=False)

    mean_std_df = build_mean_std_table(all_runs_df)
    mean_std_df.to_csv(run_root / 'overall_mean_std.csv', index=False)

    mean_ci_df = build_mean_ci_table(all_runs_df, AGGREGATE_METRICS)
    mean_ci_df.to_csv(run_root / 'overall_mean_95ci.csv', index=False)

    inference_df = build_paired_inference_table(
        all_runs_df,
        MAIN_INFERENCE_METRICS,
        LOWER_IS_BETTER,
    )
    inference_df.to_csv(run_root / 'paired_significance_tests.csv', index=False)

    win_count_df = build_win_count_table(all_runs_df)
    win_count_df.to_csv(run_root / 'proposed_win_counts.csv', index=False)

    delta_df = build_delta_table(all_runs_df)
    delta_df.to_csv(run_root / 'proposed_vs_queue_unaware_deltas.csv', index=False)

    if leaderboard_rows:
        pd.DataFrame(leaderboard_rows).to_csv(run_root / 'proposed_per_seed.csv', index=False)

    if ablation_tables:
        all_ablation_df = pd.concat(ablation_tables, ignore_index=True)
        all_ablation_df.to_csv(run_root / 'ablation_all_seeds.csv', index=False)
        ablation_gain_df = build_ablation_relative_gain_table(all_ablation_df)
        ablation_gain_df.to_csv(run_root / 'ablation_relative_gain_by_seed.csv', index=False)
        build_ablation_gain_summary(ablation_gain_df).to_csv(
            run_root / 'ablation_relative_gain_mean_std.csv',
            index=False,
        )

    if training_tables:
        all_training_df = pd.concat(training_tables, ignore_index=True)
        all_training_df.to_csv(run_root / 'training_curves_all_seeds.csv', index=False)
        training_curve_summary = build_grouped_mean_ci_table(
            all_training_df,
            ['method', 'episode'],
        )
        training_curve_summary.to_csv(
            run_root / 'training_curves_mean_95ci.csv',
            index=False,
        )
        plot_training_curves_with_ci(
            training_curve_summary,
            run_root / 'training_curves_mean_95ci',
        )

    if generalization_tables:
        all_generalization_df = pd.concat(generalization_tables, ignore_index=True)
        all_generalization_df.to_csv(
            run_root / 'generalization_all_seeds.csv',
            index=False,
        )
        build_grouped_mean_ci_table(
            all_generalization_df,
            [
                'scenario',
                'arrival_scale',
                'arrival_distribution',
                'channel_distribution',
                'method',
            ],
        ).to_csv(run_root / 'generalization_mean_95ci.csv', index=False)
        generalization_inference = build_stratified_paired_inference_table(
            all_generalization_df,
            [
                'scenario',
                'arrival_scale',
                'arrival_distribution',
                'channel_distribution',
            ],
            GENERALIZATION_INFERENCE_METRICS,
            GENERALIZATION_LOWER_IS_BETTER,
        )
        generalization_inference.to_csv(
            run_root / 'generalization_paired_significance_tests.csv',
            index=False,
        )

    if overload_tables:
        all_overload_df = pd.concat(overload_tables, ignore_index=True)
        all_overload_df.to_csv(run_root / 'overload_all_seeds.csv', index=False)
        build_grouped_mean_ci_table(
            all_overload_df,
            ['method', 'horizon_slots', 'arrival_scale'],
        ).to_csv(run_root / 'overload_mean_95ci.csv', index=False)
        overload_inference = build_stratified_paired_inference_table(
            all_overload_df,
            ['horizon_slots', 'arrival_scale'],
            OVERLOAD_INFERENCE_METRICS,
            OVERLOAD_LOWER_IS_BETTER,
        )
        overload_inference.to_csv(
            run_root / 'overload_paired_significance_tests.csv',
            index=False,
        )

    summary = {
        'config': str(config_path),
        'seeds': list(args.seeds),
        'ablation_seeds': [
            seed for seed in args.seeds if seed in set(args.ablation_seeds)
        ],
        'num_completed_runs': len(overall_tables),
        'aggregate_files': {
            'all_runs_overall': str(run_root / 'all_runs_overall.csv'),
            'overall_mean_std': str(run_root / 'overall_mean_std.csv'),
            'overall_mean_95ci': str(run_root / 'overall_mean_95ci.csv'),
            'paired_significance_tests': str(run_root / 'paired_significance_tests.csv'),
            'proposed_per_seed': str(run_root / 'proposed_per_seed.csv'),
            'proposed_win_counts': str(run_root / 'proposed_win_counts.csv'),
            'proposed_vs_queue_unaware_deltas': str(run_root / 'proposed_vs_queue_unaware_deltas.csv'),
            'ablation_all_seeds': str(run_root / 'ablation_all_seeds.csv') if ablation_tables else None,
            'ablation_relative_gain_by_seed': str(run_root / 'ablation_relative_gain_by_seed.csv') if ablation_tables else None,
            'ablation_relative_gain_mean_std': str(run_root / 'ablation_relative_gain_mean_std.csv') if ablation_tables else None,
            'training_curves_all_seeds': str(run_root / 'training_curves_all_seeds.csv') if training_tables else None,
            'training_curves_mean_95ci': str(run_root / 'training_curves_mean_95ci.csv') if training_tables else None,
            'training_curves_figure': str(run_root / 'training_curves_mean_95ci.pdf') if training_tables else None,
            'generalization_all_seeds': str(run_root / 'generalization_all_seeds.csv') if generalization_tables else None,
            'generalization_mean_95ci': str(run_root / 'generalization_mean_95ci.csv') if generalization_tables else None,
            'generalization_paired_significance_tests': str(run_root / 'generalization_paired_significance_tests.csv') if generalization_tables else None,
            'overload_all_seeds': str(run_root / 'overload_all_seeds.csv') if overload_tables else None,
            'overload_mean_95ci': str(run_root / 'overload_mean_95ci.csv') if overload_tables else None,
            'overload_paired_significance_tests': str(run_root / 'overload_paired_significance_tests.csv') if overload_tables else None,
        },
        'inference_protocol': {
            'confidence_intervals': 'two-sided 95% Student-t intervals across independent training seeds',
            'paired_test': 'exact two-sided paired sign-flip randomization test',
            'multiplicity': 'Holm correction within each baseline across reported performance metrics',
            'interpretation': 'positive paired advantage favors Proposed Method; statistical significance is not assumed',
        },
    }
    (run_root / 'multiseed_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote aggregate results to {run_root}', flush=True)


if __name__ == '__main__':
    main()
