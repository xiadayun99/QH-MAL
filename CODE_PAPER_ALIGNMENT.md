# Code–Paper Alignment

The final manuscript and this repository use the same five-seed protocol and the same `configs/paper_results.json` preset.

| Manuscript element | Executable implementation |
|---|---|
| Arrival-before-service finite queue | `paper_mec/env.py` |
| Tentative request vs. executed admission | `MECEnvironment._apply_admission_projection` |
| Price, utilization, and allocation-shape projections | `paper_mec/env.py` |
| Categorical user actor and continuous server actors | `paper_mec/models.py`, `paper_mec/trainer.py` |
| Straight-through Gumbel–Softmax update | `HeteroOffPolicyTrainer._straight_through_user_actions` |
| Role-specific twin critics and update intervals `(1, 2)` | `paper_mec/trainer.py` |
| Nine-method comparison | `run.py` |
| Server return $\Pi_s^t$ | `paper_mec/env.py` (`avg_profit` internally), exported as `Server Return` |
| Five-seed aggregation and confidence intervals | `scripts/run_multiseed_evaluation.py` |
| Population sensitivity | `scripts/run_user_sensitivity_experiment.py` |
| Buffer sensitivity and `10,000`-slot stress test | `scripts/run_buffer_stress_experiment.py` |
| Final figures 2–7 | `plot_from_results.py` |

## Hard-feasibility boundary

The environment, not a learned queue predictor, applies the queue recurrence. Before reward calculation and replay insertion, deterministic projections enforce:

- valid local/candidate-server requests;
- price and allocation-control intervals;
- finite pre-service admission capacity;
- nonnegative CPU shares whose sum does not exceed server capacity.

Rejected edge requests execute locally; workload is not silently discarded. The finite-buffer result is therefore an implementation-linked feasibility certificate, not a Lyapunov-convergence claim.

## Controlled learning comparisons

QU-QH-MAL retains the proposed learning structure but removes queue observations, queue-induced perceived delay, and the queue reward term. QA-MADDPG and QA-MATD3 use the same queue-aware inputs, action interface, role rewards, feasibility projections, training budget, evaluation realizations, and checkpoint rule as QH-MAL. Their comparison isolates critic and update structure rather than information access.

## Fixed-dimensional allocation interface

Each server actor emits two allocation controls: aggregate utilization and allocation shape. The environment maps them permutation-equivariantly to the currently admitted users. The centralized action width is therefore fixed at `U * (S + 1) + 3 * S`, even when the admitted-user set changes.
