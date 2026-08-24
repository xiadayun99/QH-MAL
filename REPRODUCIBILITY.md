# Reproducibility Protocol

## Authoritative settings

| Item | Final setting |
|---|---|
| Users / servers | `20 / 4` |
| Buffer capacity | `1.2e10` cycles per server |
| Training budget | `600` episodes × `80` slots |
| Evaluation | `80` deterministic episodes per seed |
| Main seeds | `7, 11, 19, 23, 31` |
| User / server actor LR | `8e-5 / 4e-5` |
| Actor update intervals | `1 / 2` |
| Arrival scales | `0.8, 1.0, 1.2, 1.4` |
| Population study | `U = 20, 30, 40`, seed `7` |
| Buffer windows | `0.5, 1.0, 2.0` s, seed `7` |
| Stress test | `10,000` slots, load `1.6`, seed `7` |

`scripts/check_environment.py --strict` verifies the configuration values before a long run.

## Method set

The main table contains exactly nine methods:

1. Random Offloading
2. Greedy Offloading
3. Fixed-Pricing Queue-Aware Greedy
4. DPP Joint Pricing-Offloading (DPP-JPO)
5. Stackelberg Joint Pricing-Offloading (SG-JPO)
6. Queue-Unaware QH-MAL (QU-QH-MAL)
7. Queue-Aware MADDPG (QA-MADDPG)
8. Queue-Aware MATD3 (QA-MATD3)
9. QH-MAL (`Proposed Method` in raw tables)

Human-readable comparison columns and plot labels use `Server Return` for $\Pi_s^t$. The lower-case key `avg_profit` remains in internal episode and sensitivity records for backward compatibility. The plotting script accepts both the final `Server Return` column and the legacy `Profit` column.

## Experiment-to-artifact mapping

| Manuscript artifact | Required raw output | Generator |
|---|---|---|
| Table: overall performance | `all_runs_overall.csv`, `overall_mean_95ci.csv` | `run_multiseed_evaluation.py` |
| Fig. 2: overall comparison | `all_runs_overall.csv` | `plot_from_results.py` |
| Fig. 3: QH-MAL vs. QU-QH-MAL | `all_runs_overall.csv` | `plot_from_results.py` |
| Fig. 4: arrival sensitivity | `seed_*/results/arrival_sensitivity.csv` | `plot_from_results.py` |
| Fig. 5: population sensitivity | `user_sensitivity_overall.csv` | `plot_from_results.py` |
| Fig. 6: buffer sensitivity | `buffer_capacity_sensitivity.csv` | `plot_from_results.py` |
| Stress-test paragraph | `long_horizon_stress_test.csv` | `run_buffer_stress_experiment.py` |
| Fig. 7: component ablations | `ablation_relative_gain_mean_std.csv` | `plot_from_results.py` |

## Statistical unit

For the main and arrival experiments, a replicate is one independently trained seed. Each seed-level number is the mean of 80 deterministic evaluation episodes. Two-sided 95% Student-t intervals are computed across the five seed-level means. Paired differences use shared seed IDs. Population and buffer results are fixed seed-7 sensitivity studies and are not presented as five-seed confidence intervals.

With five paired seeds, the minimum attainable two-sided exact sign-flip p-value is `0.0625`; the manuscript therefore emphasizes paired effect sizes and confidence intervals rather than claiming significance from this test.

## Completion checks

Before using regenerated results:

1. Confirm all five `seed_<id>/run_metadata.json` files have return code `0`.
2. Confirm `multiseed_summary.json` reports the exact five seeds.
3. Confirm all nine methods appear in `all_runs_overall.csv` for every seed.
4. Confirm `Overflow Count` and `Hard Constraint Violation Rate` are zero.
5. Confirm the stress table reports `horizon_slots = 10000` and `arrival_scale = 1.6`.
6. Compare aggregate values with `reference_results/` using the reported confidence intervals.
