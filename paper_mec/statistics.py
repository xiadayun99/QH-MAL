from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def mean_t_interval(
    values: np.ndarray | pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Return mean, sample standard deviation, and two-sided Student-t CI."""

    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(sample.mean())
    if sample.size == 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(sample.std(ddof=1))
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=sample.size - 1))
    half_width = critical * std / np.sqrt(sample.size)
    return mean, std, mean - half_width, mean + half_width


def exact_paired_sign_flip_pvalue(advantages: np.ndarray | pd.Series) -> float:
    """Exact two-sided randomization p-value for paired seed advantages.

    The null distribution flips the sign of each seed-paired difference.  The
    implementation enumerates all assignments up to 20 seeds, which covers the
    paper protocol exactly and avoids an asymptotic small-sample claim.
    """

    values = np.asarray(advantages, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    tolerance = 1.0e-15
    if values.size <= 20:
        exceed = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=values.size):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            exceed += int(statistic >= observed - tolerance)
            total += 1
        return float(exceed / total)

    rng = np.random.default_rng(20260812)
    draws = 200_000
    signs = rng.choice((-1.0, 1.0), size=(draws, values.size))
    statistics = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(statistics >= observed - tolerance)) / (draws + 1))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-adjust a family of finite p-values while preserving input order."""

    adjusted = [float("nan")] * len(pvalues)
    finite = [(index, value) for index, value in enumerate(pvalues) if np.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    family_size = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min((family_size - rank) * float(value), 1.0)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def build_mean_ci_table(
    frame: pd.DataFrame,
    metrics: list[str],
    confidence: float = 0.95,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in frame.groupby("Method", sort=False):
        row: dict[str, object] = {
            "Method": method,
            "Seed Count": int(group["Seed"].nunique()),
            "Seeds": ",".join(str(int(seed)) for seed in sorted(group["Seed"].unique())),
        }
        for metric in metrics:
            mean, std, lower, upper = mean_t_interval(group[metric], confidence)
            row[f"{metric} Mean"] = mean
            row[f"{metric} Std"] = std
            row[f"{metric} {confidence:.0%} CI Lower"] = lower
            row[f"{metric} {confidence:.0%} CI Upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_inference_table(
    frame: pd.DataFrame,
    metrics: list[str],
    lower_is_better: set[str],
    reference: str = "Proposed Method",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Build paired seed tests; positive advantages always favor reference."""

    reference_df = frame[frame["Method"] == reference].set_index("Seed")
    rows: list[dict[str, object]] = []
    for method in frame["Method"].drop_duplicates():
        if method == reference:
            continue
        baseline_df = frame[frame["Method"] == method].set_index("Seed")
        shared = reference_df.index.intersection(baseline_df.index).sort_values()
        if shared.empty:
            continue
        family_start = len(rows)
        for metric in metrics:
            reference_values = reference_df.loc[shared, metric].to_numpy(dtype=np.float64)
            baseline_values = baseline_df.loc[shared, metric].to_numpy(dtype=np.float64)
            advantages = (
                baseline_values - reference_values
                if metric in lower_is_better
                else reference_values - baseline_values
            )
            mean, std, lower, upper = mean_t_interval(advantages, confidence)
            effect_size = mean / std if std > 1.0e-15 else (
                float("inf") if mean > 0.0 else float("-inf") if mean < 0.0 else 0.0
            )
            rows.append(
                {
                    "Reference": reference,
                    "Compared Against": method,
                    "Metric": metric,
                    "Seed Count": int(len(shared)),
                    "Seeds": ",".join(str(int(seed)) for seed in shared),
                    "Mean Paired Advantage": mean,
                    "Paired Advantage Std": std,
                    f"Paired Advantage {confidence:.0%} CI Lower": lower,
                    f"Paired Advantage {confidence:.0%} CI Upper": upper,
                    "Paired Effect Size dz": effect_size,
                    "Exact Two-Sided Sign-Flip p": exact_paired_sign_flip_pvalue(advantages),
                    "Minimum Attainable Two-Sided p": float(2.0 / (2 ** len(shared))),
                }
            )
        family_rows = rows[family_start:]
        adjusted = holm_adjust(
            [float(row["Exact Two-Sided Sign-Flip p"]) for row in family_rows]
        )
        for row, adjusted_p in zip(family_rows, adjusted):
            row["Holm-Adjusted p"] = adjusted_p
            row["Holm Significant at 0.05"] = bool(adjusted_p < 0.05)
    return pd.DataFrame(rows)
