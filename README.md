# QH-MAL: Reproducible MEC Experiments

Reproduction package for **“Queue-Aware Pricing and Offloading for Mobile Edge Computing with Heterogeneous Multi-Agent Learning.”** It implements the manuscript's nine-method comparison, five-seed evaluation, sensitivity studies, finite-buffer stress test, ablations, and final figure pipeline.

## What this repository reproduces

- Main comparison over seeds `7, 11, 19, 23, 31`, with 80 deterministic evaluation episodes per seed.
- QH-MAL, QU-QH-MAL, QA-MADDPG, QA-MATD3, three heuristics, DPP-JPO, and SG-JPO.
- Arrival-intensity and user-population sensitivity.
- Buffer-window sensitivity and the fixed-policy `10,000`-slot stress test at load `1.6`.
- Five single-factor ablations and the six figures used in the final manuscript.

The authoritative training preset is [`configs/paper_results.json`](configs/paper_results.json). In particular, it fixes the `600 x 80` training budget and the user/server actor learning rates to `8e-5 / 4e-5`.

Human-readable tables and figures use **server return** for $\Pi_s^t$, which combines collected revenue, allocation fairness, and post-decision congestion. The machine-readable field `avg_profit` is retained only for compatibility with existing raw metrics and checkpoints. It does not denote monetary revenue alone.

## Installation

The reported environment uses Ubuntu 24.04.3, Python 3.10, PyTorch 2.11.0, and CUDA 13.0.

```bash
conda env create -f environment.yml
conda activate qh-mal
python scripts/check_environment.py --strict
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Fast validation

The first command exercises the finite-buffer and CPU projections without training. The second runs the test suite. The third performs a small CPU training smoke run.

```bash
python scripts/smoke_environment.py
python -m pytest -q
python run.py --train-config configs/smoke_test.json --device cpu --quick --skip-sweeps --skip-arrival-sweep --skip-stress
```

## Full paper reproduction

```bash
python scripts/reproduce_all.py --config configs/paper_results.json --device cuda:0
```

This command runs the five-seed main comparison and ablations, the seed-7 population and buffer studies, the `10,000`-slot stress test, and all final plots. On hardware comparable to the reported RTX PRO 6000 system, plan for several tens of GPU-hours.

The stages can also be run separately:

```bash
# Five-seed main comparison, arrival sensitivity, and ablations
python scripts/run_multiseed_evaluation.py \
  --config configs/paper_results.json \
  --device cuda:0 \
  --run-ablation

# Seed-7 user-count study: U = 20, 30, 40
python scripts/run_user_sensitivity_experiment.py \
  --train-config configs/paper_results.json \
  --device cuda:0 \
  --seed 7 \
  --output-dir user_sensitivity_package_final

# Seed-7 buffer study and stress test
python scripts/run_buffer_stress_experiment.py \
  --train-config configs/paper_results.json \
  --device cuda:0 \
  --seed 7 \
  --buffer-windows 0.5 1.0 2.0 \
  --horizon 10000 \
  --stress-arrival-scale 1.6

# Final manuscript figures
python plot_from_results.py \
  --multiseed-dir experiments/multiseed/paper_results \
  --user-results user_sensitivity_package_final/results/user_sensitivity_overall.csv \
  --buffer-results buffer_stress_results/buffer_capacity_sensitivity.csv \
  --out-dir paper_assets
```

## Outputs

| Output | Location |
|---|---|
| Per-seed runs and checkpoints | `experiments/multiseed/paper_results/seed_<seed>/` |
| Five-seed aggregate tables | `experiments/multiseed/paper_results/` |
| User-count study | `user_sensitivity_package_final/` |
| Buffer and stress study | `buffer_stress_results/` |
| Final figures 2–7 | `paper_assets/` |
| Manuscript reference values | `reference_results/` |

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the experiment-to-figure mapping and [`CODE_PAPER_ALIGNMENT.md`](CODE_PAPER_ALIGNMENT.md) for the implementation mapping.

## Reproducibility notes

- Training seeds are independent; evaluation realizations are matched across methods within each seed.
- Confidence intervals are two-sided 95% Student-t intervals over the five seed-level means. Episodes are not treated as independent replicates.
- GPU kernels can introduce small numerical differences across hardware and driver versions. Compare regenerated values with the confidence intervals in `reference_results/`, not by bitwise equality.
- Version `1.1.0` changes public terminology and release tooling only. The environment dynamics, algorithms, hyperparameters, seeds, and reference values are identical to version `1.0.0`.
- No software license has been selected on behalf of the authors. Add the intended license before making the repository public.
