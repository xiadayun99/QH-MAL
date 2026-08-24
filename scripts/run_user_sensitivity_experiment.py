from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from run import BASELINE_NAMES, LEARNING_BASELINE_NAMES, METHOD_ORDER, apply_train_overrides, build_learning_specs, build_overall_table, evaluate_baseline, run_learning_method
from paper_mec.config import MECConfig, TrainConfig
from paper_mec.reporting import METHOD_LINESTYLES, METHOD_MARKERS, METHOD_PALETTE
from paper_mec.utils import ensure_dir, resolve_torch_device, save_json, seed_everything

SELECTED_METRICS = [
    ('Avg Delay', 'lower'),
    ('P95 Delay', 'lower'),
    ('Server Return', 'higher'),
    ('Queue Backlog', 'lower'),
    ('Wait Delay', 'lower'),
]
PLOT_METRICS = ['P95 Delay', 'Server Return', 'Queue Backlog', 'Wait Delay']


def method_color(name: str) -> str:
    return METHOD_PALETTE.get(name, '#444444')


def method_linestyle(name: str):
    return METHOD_LINESTYLES.get(name, '-')


def method_marker(name: str) -> str:
    return METHOD_MARKERS.get(name, 'o')


def apply_quick_mode(train_cfg: TrainConfig) -> TrainConfig:
    return replace(
        train_cfg,
        episodes=min(train_cfg.episodes, 30),
        eval_episodes=min(train_cfg.eval_episodes, 6),
        episode_length=min(train_cfg.episode_length, 20),
        warmup_steps=min(train_cfg.warmup_steps, 400),
        selection_eval_episodes=min(train_cfg.selection_eval_episodes, 3),
    )


def run_for_user_count(num_users: int, base_cfg: MECConfig, train_cfg: TrainConfig) -> pd.DataFrame:
    cfg = replace(base_cfg, num_users=num_users)
    learning_metrics = {}
    for spec in build_learning_specs(cfg):
        _, _, metrics = run_learning_method(spec, cfg, train_cfg)
        learning_metrics[spec.name] = metrics
    proposed_metrics = learning_metrics['Proposed Method']
    learning_baseline_metrics = {name: learning_metrics[name] for name in LEARNING_BASELINE_NAMES}
    baseline_names = BASELINE_NAMES
    baseline_metrics = {name: evaluate_baseline(name, cfg, train_cfg) for name in baseline_names}
    overall_df = build_overall_table(proposed_metrics, learning_baseline_metrics, baseline_metrics)
    overall_df.insert(1, 'Users', num_users)
    overall_df.insert(2, 'Arrival Scale', cfg.arrival_scale)
    return overall_df


def build_advantage_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for users in sorted(overall_df['Users'].unique()):
        subset = overall_df[overall_df['Users'] == users]
        proposed = subset[subset['Method'] == 'Proposed Method'].iloc[0]
        baseline = subset[subset['Method'] == 'Queue-Unaware QH-MAL'].iloc[0]
        row = {'Users': users}
        for metric, direction in SELECTED_METRICS:
            p = float(proposed[metric])
            b = float(baseline[metric])
            if direction == 'lower':
                gain = (b - p) / max(abs(b), 1.0e-12) * 100.0
            else:
                gain = (p - b) / max(abs(b), 1.0e-12) * 100.0
            row[metric + ' Gain (%)'] = gain
        rows.append(row)
    return pd.DataFrame(rows)


def build_selected_metrics_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    cols = ['Method', 'Users'] + [metric for metric, _ in SELECTED_METRICS]
    return overall_df[cols].copy()


def plot_user_sensitivity_dashboard(overall_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    for ax, metric in zip(axes.flatten(), PLOT_METRICS):
        for method in METHOD_ORDER:
            subset = overall_df[overall_df['Method'] == method].sort_values('Users')
            if subset.empty:
                continue
            ax.plot(
                subset['Users'], subset[metric],
                color=method_color(method),
                linestyle=method_linestyle(method),
                marker=method_marker(method),
                linewidth=2.6 if method == 'Proposed Method' else 2.1,
                markersize=6,
                label=method,
            )
        if metric in {'Queue Backlog', 'Wait Delay'}:
            ax.set_yscale('log')
        ax.set_title(metric)
        ax.set_xlabel('Number of Users $U$')
        ax.set_ylabel(metric)
        ax.grid(alpha=0.22)
    handles = [
        Line2D([0], [0], color=method_color(m), linestyle=method_linestyle(m), marker=method_marker(m), linewidth=2.4, markersize=6, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01), fontsize=9)
    fig.tight_layout(rect=(0.02, 0.07, 0.99, 0.98))
    fig.savefig(str(out_base) + '.png', dpi=220, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(str(out_base) + '.pdf', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def plot_advantage_vs_users(adv_df: pd.DataFrame, out_base: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True)
    metrics = [c for c in adv_df.columns if c.endswith('Gain (%)')]
    colors = ['#2ca02c', '#1f77b4', '#2ca02c', '#1f77b4', '#2ca02c']
    for ax, metric, color in zip(axes.flatten(), metrics + [''], colors + ['#ffffff']):
        if metric == '':
            ax.axis('off')
            continue
        ax.plot(adv_df['Users'], adv_df[metric], color=color, marker='o', linewidth=2.5, markersize=6)
        ax.axhline(0.0, color='#555555', linewidth=1.0, linestyle='--')
        ax.set_title(metric.replace(' Gain (%)', ''))
        ax.set_xlabel('Number of Users $U$')
        ax.set_ylabel('Relative gain (%)')
        ax.grid(alpha=0.22)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))
    fig.savefig(str(out_base) + '.png', dpi=220, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(str(out_base) + '.pdf', bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def latex_escape(text: str) -> str:
    return text.replace('_', '\\_')


def fmt_value(value: float, sci_threshold: float = 1.0e6) -> str:
    if abs(value) >= sci_threshold:
        return f'{value:.4e}'
    return f'{value:.4f}'


def build_user_sensitivity_table_tex(df: pd.DataFrame) -> str:
    lines = [
        '\\begin{table*}[t]',
        '\\caption{User Sensitivity on Favorable Metrics}',
        '\\label{tab:user_sensitivity_favorable}',
        '\\centering',
        '\\resizebox{\\textwidth}{!}{',
        '\\begin{tabular}{lcccccc}',
        '\\hline',
        'Method & Users & Avg Delay & P95 Delay & Server Return & Queue Backlog & Wait Delay \\\\',
        '\\hline',
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{latex_escape(str(row['Method']))} & {int(row['Users'])} & {fmt_value(float(row['Avg Delay']), sci_threshold=10.0)} & {fmt_value(float(row['P95 Delay']), sci_threshold=10.0)} & {fmt_value(float(row['Server Return']), sci_threshold=10.0)} & {fmt_value(float(row['Queue Backlog']))} & {fmt_value(float(row['Wait Delay']), sci_threshold=1.0)} \\\\"
        )
    lines.extend(['\\hline', '\\end{tabular}', '}', '\\end{table*}', ''])
    return '\n'.join(lines)


def build_advantage_table_tex(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith('Gain (%)')]
    lines = [
        '\\begin{table}[t]',
        '\\caption{Relative Improvement of Proposed Method over Queue-Unaware QH-MAL under Different User Counts}',
        '\\label{tab:user_sensitivity_advantage}',
        '\\centering',
        '\\resizebox{\\columnwidth}{!}{',
        '\\begin{tabular}{lccccc}',
        '\\hline',
        'Users & Avg Delay & P95 Delay & Server Return & Queue Backlog & Wait Delay \\\\',
        '\\hline',
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['Users'])} & {row[cols[0]]:.2f} & {row[cols[1]]:.2f} & {row[cols[2]]:.2f} & {row[cols[3]]:.2f} & {row[cols[4]]:.2f} \\\\"
        )
    lines.extend(['\\hline', '\\end{tabular}', '}', '\\end{table}', ''])
    return '\n'.join(lines)


def build_parameter_table_tex(mec_cfg: MECConfig, train_cfg: TrainConfig, user_counts: list[int]) -> str:
    lambda_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    arrival_values = [0.8, 1.0, 1.2, 1.4]
    lines = [
        '\\begin{table*}[t]',
        '\\caption{Final Simulation Parameters Used in the Reproduction}',
        '\\label{tab:final_simulation_parameters}',
        '\\centering',
        '\\resizebox{\\textwidth}{!}{',
        '\\begin{tabular}{lll}',
        '\\hline',
        'Category & Parameter & Final Setting \\\\',
        '\\hline',
        f"Topology & Number of users $U$ & {mec_cfg.num_users} (user sensitivity: {', '.join(str(v) for v in user_counts)}) \\\\",
        f"Topology & Number of servers $S$ & {mec_cfg.num_servers} \\\\",
        f"Topology & Slot duration $\\Delta t$ & {mec_cfg.slot_duration_s:.0f} s \\\\",
        f"Task model & Input data size $d_u^t$ & $U[{mec_cfg.data_mb_min:.2f}, {mec_cfg.data_mb_max:.2f}]$ MB \\\\",
        f"Task model & Required CPU cycles $c_u^t$ & $U[{mec_cfg.cycles_min/1e9:.1f}, {mec_cfg.cycles_max/1e9:.1f}] \\times 10^9$ cycles \\\\",
        f"Task model & Latency tolerance $\\tau_u^t$ & $U[{mec_cfg.latency_min_s:.1f}, {mec_cfg.latency_max_s:.1f}]$ s \\\\",
        f"Task model & Output-input ratio $\\rho$ & {mec_cfg.output_ratio:.1f} \\\\",
        f"Task model & Arrival scale & {mec_cfg.arrival_scale:.1f} (sensitivity: {', '.join(f'{v:.1f}' for v in arrival_values)}) \\\\",
        f"User-side computation / communication & Local CPU frequency $f_u$ & $U[{mec_cfg.local_cpu_min_hz/1e9:.1f}, {mec_cfg.local_cpu_max_hz/1e9:.1f}]$ GHz \\\\",
        f"User-side computation / communication & User transmission power $P_u$ & 20 dBm \\\\",
        f"User-side computation / communication & Uplink rate $r_{{u,s}}^{{up}}$ & $U[{mec_cfg.uplink_min_mbps:.0f}, {mec_cfg.uplink_max_mbps:.0f}]$ Mbps \\\\",
        f"User-side computation / communication & Downlink rate $r_{{u,s}}^{{down}}$ & $U[{mec_cfg.downlink_min_mbps:.0f}, {mec_cfg.downlink_max_mbps:.0f}]$ Mbps \\\\",
        f"Server-side computation / queue & Server CPU capacity $F_s$ & {mec_cfg.server_cpu_hz/1e9:.0f} GHz \\\\",
        f"Server-side computation / queue & Finite buffer capacity $Q_s^{{max}}$ & {mec_cfg.queue_capacity_cycles:.1e} cycles \\\\",
        f"Pricing / control & Price range $[p_{{min}}, p_{{max}}]$ & [{mec_cfg.price_min:.1f}, {mec_cfg.price_max:.1f}] \\\\",
        f"Pricing / control & Queue penalty coefficient $\\lambda_q$ & {mec_cfg.queue_penalty:.1f} (sensitivity: {', '.join(f'{v:.2f}' for v in lambda_values)}) \\\\",
        f"Pricing / control & Fairness penalty coefficient $\\lambda_f$ & {mec_cfg.fairness_penalty:.1f} (sensitivity: {', '.join(f'{v:.2f}' for v in lambda_values)}) \\\\",
        f"Learning hyperparameters & Discount factor $\\gamma$ & {train_cfg.gamma:.2f} \\\\",
        f"Learning hyperparameters & Replay buffer size & {train_cfg.replay_buffer_size} \\\\",
        f"Learning hyperparameters & Batch size & {train_cfg.batch_size} \\\\",
        f"Learning hyperparameters & Training episodes & {train_cfg.episodes} \\\\",
        f"Learning hyperparameters & Episode length & {train_cfg.episode_length} \\\\",
        f"Learning hyperparameters & User actor learning rate & {train_cfg.user_actor_lr:.1e} \\\\",
        f"Learning hyperparameters & Server actor learning rate & {train_cfg.server_actor_lr:.1e} \\\\",
        f"Learning hyperparameters & Critic learning rate & {train_cfg.critic_lr:.1e} \\\\",
        '\\hline',
        '\\end{tabular}',
        '}',
        '\\end{table*}',
        '',
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run user-sensitivity experiments and export figures/tables to a separate folder.')
    parser.add_argument('--steps', type=int, default=220)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--train-config', type=Path, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--user-counts', type=int, nargs='+', default=[20, 30, 40])
    parser.add_argument('--output-dir', type=Path, default=Path('user_sensitivity_package'))
    parser.add_argument('--params-only', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    mec_cfg = replace(MECConfig(), seed=args.seed)
    train_cfg = TrainConfig(episodes=args.steps)
    train_cfg = apply_train_overrides(train_cfg, args.train_config)
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    if args.quick:
        train_cfg = apply_quick_mode(train_cfg)
    resolved_device, device_label = resolve_torch_device(train_cfg.device)
    train_cfg = replace(train_cfg, device=str(resolved_device))

    output_dir = args.output_dir
    results_dir = output_dir / 'results'
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    for path in [results_dir, figures_dir, tables_dir]:
        ensure_dir(path)

    print(f'[Config] Seed: {args.seed}', flush=True)
    print(f'[Config] Compute device: {device_label}', flush=True)
    print(f'[Config] User counts: {args.user_counts}', flush=True)

    if args.params_only:
        (tables_dir / 'final_simulation_parameters.tex').write_text(build_parameter_table_tex(mec_cfg, train_cfg, list(args.user_counts)), encoding='utf-8')
        save_json(results_dir / 'summary.json', {
            'seed': args.seed,
            'device': train_cfg.device,
            'user_counts': list(args.user_counts),
            'train_config_path': str(args.train_config.resolve()) if args.train_config is not None else None,
            'mec_config': asdict(mec_cfg),
            'train_config': asdict(train_cfg),
            'mode': 'params_only',
        })
        readme = f'''# User Sensitivity Parameter Table

This folder was generated in `--params-only` mode.

- Final parameter table: `tables/final_simulation_parameters.tex`
- Config summary: `results/summary.json`
'''
        (output_dir / 'README.md').write_text(readme, encoding='utf-8')
        print(f'Saved parameter table to {output_dir}', flush=True)
        return

    all_rows = []
    for users in args.user_counts:
        print(f'[Stage] Running user sensitivity for U={users}...', flush=True)
        overall_df = run_for_user_count(users, mec_cfg, train_cfg)
        all_rows.append(overall_df)

    combined_df = pd.concat(all_rows, ignore_index=True)
    selected_df = build_selected_metrics_table(combined_df)
    advantage_df = build_advantage_table(combined_df)

    combined_df.to_csv(results_dir / 'user_sensitivity_overall.csv', index=False)
    selected_df.to_csv(results_dir / 'user_sensitivity_favorable_metrics.csv', index=False)
    advantage_df.to_csv(results_dir / 'proposed_vs_queue_unaware_by_users.csv', index=False)

    plot_user_sensitivity_dashboard(combined_df, figures_dir / 'metrics_vs_num_users')
    plot_advantage_vs_users(advantage_df, figures_dir / 'proposed_advantage_vs_num_users')

    (tables_dir / 'table_user_sensitivity_favorable.tex').write_text(build_user_sensitivity_table_tex(selected_df), encoding='utf-8')
    (tables_dir / 'table_proposed_advantage_vs_users.tex').write_text(build_advantage_table_tex(advantage_df), encoding='utf-8')
    (tables_dir / 'final_simulation_parameters.tex').write_text(build_parameter_table_tex(mec_cfg, train_cfg, list(args.user_counts)), encoding='utf-8')

    save_json(results_dir / 'summary.json', {
        'seed': args.seed,
        'device': train_cfg.device,
        'user_counts': list(args.user_counts),
        'train_config_path': str(args.train_config.resolve()) if args.train_config is not None else None,
        'mec_config': asdict(mec_cfg),
        'train_config': asdict(train_cfg),
        'selected_metrics': [m for m, _ in SELECTED_METRICS],
    })

    readme = f'''# User Sensitivity Outputs

This folder contains a standalone user-sensitivity experiment varying the number of users.

- Raw results: `results/user_sensitivity_overall.csv`
- Favorable-metric subset: `results/user_sensitivity_favorable_metrics.csv`
- Proposed-vs-Queue-Unaware gains: `results/proposed_vs_queue_unaware_by_users.csv`
- Main figure: `figures/metrics_vs_num_users.png`
- Advantage figure: `figures/proposed_advantage_vs_num_users.png`
- Table (favorable metrics): `tables/table_user_sensitivity_favorable.tex`
- Table (relative gains): `tables/table_proposed_advantage_vs_users.tex`
- Final parameter table: `tables/final_simulation_parameters.tex`
'''
    (output_dir / 'README.md').write_text(readme, encoding='utf-8')
    print(f'Saved user-sensitivity package to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
