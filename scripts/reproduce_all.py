from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label: str, command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f'[{label}] {" ".join(command)}', flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the final five-seed paper protocol and regenerate all manuscript figures.')
    parser.add_argument('--config', default='configs/paper_results.json', help='Training config used throughout the reproduction.')
    parser.add_argument('--device', default='cuda:0', help='Device passed to all underlying scripts.')
    parser.add_argument('--multiseed-output-root', default='experiments/multiseed', help='Root directory used by the multiseed runner.')
    parser.add_argument('--user-output-dir', default='user_sensitivity_package_final', help='Output directory for the user-sensitivity experiment.')
    parser.add_argument('--buffer-output-dir', default='buffer_stress_results', help='Output directory for the buffer and stress experiments.')
    parser.add_argument('--paper-assets-dir', '--out-dir', dest='paper_assets_dir', default='paper_assets', help='Directory for the six final paper figures.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_stem = Path(args.config).stem

    multiseed_cmd = [
        sys.executable,
        'scripts/run_multiseed_evaluation.py',
        '--config', args.config,
        '--device', args.device,
        '--output-root', args.multiseed_output_root,
        '--seeds', '7', '11', '19', '23', '31',
        '--ablation-seeds', '7', '11', '19', '23', '31',
        '--run-ablation',
    ]
    plot_cmd = [
        sys.executable,
        'plot_from_results.py',
        '--multiseed-dir', f'{args.multiseed_output_root}/{config_stem}',
        '--user-results', f'{args.user_output_dir}/results/user_sensitivity_overall.csv',
        '--buffer-results', f'{args.buffer_output_dir}/buffer_capacity_sensitivity.csv',
        '--out-dir', args.paper_assets_dir,
    ]
    user_cmd = [
        sys.executable,
        'scripts/run_user_sensitivity_experiment.py',
        '--train-config', args.config,
        '--device', args.device,
        '--seed', '7',
        '--user-counts', '20', '30', '40',
        '--output-dir', args.user_output_dir,
    ]
    buffer_cmd = [
        sys.executable,
        'scripts/run_buffer_stress_experiment.py',
        '--train-config', args.config,
        '--device', args.device,
        '--seed', '7',
        '--buffer-windows', '0.5', '1.0', '2.0',
        '--horizon', '10000',
        '--stress-arrival-scale', '1.6',
        '--output-dir', args.buffer_output_dir,
    ]

    run_step('multiseed', multiseed_cmd, root, args.dry_run)
    run_step('user-sensitivity', user_cmd, root, args.dry_run)
    run_step('buffer-and-stress', buffer_cmd, root, args.dry_run)
    run_step('plot', plot_cmd, root, args.dry_run)


if __name__ == '__main__':
    main()
