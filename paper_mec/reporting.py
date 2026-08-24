from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from paper_mec.utils import ensure_dir


sns.set_theme(style="whitegrid")

DEFAULT_METHOD_ORDER = [
    "Random Offloading",
    "Greedy Offloading",
    "Fixed-Pricing Queue-Aware Greedy",
    "DPP Joint Pricing-Offloading",
    "Stackelberg Joint Pricing-Offloading",
    "Queue-Unaware QH-MAL",
    "Queue-Aware MADDPG",
    "Queue-Aware MATD3",
    "Proposed Method",
]

METHOD_PALETTE = {
    "Random Offloading": "#7f7f7f",
    "Greedy Offloading": "#1f77b4",
    "Fixed-Pricing Queue-Aware Greedy": "#ff7f0e",
    "DPP Joint Pricing-Offloading": "#9467bd",
    "Stackelberg Joint Pricing-Offloading": "#17becf",
    "Queue-Unaware QH-MAL": "#d62728",
    "Queue-Aware MADDPG": "#8c564b",
    "Queue-Aware MATD3": "#e377c2",
    "Proposed Method": "#2ca02c",
}

METHOD_LINESTYLES = {
    "Random Offloading": (0, (1, 1)),
    "Greedy Offloading": (0, (5, 2)),
    "Fixed-Pricing Queue-Aware Greedy": (0, (3, 1, 1, 1)),
    "DPP Joint Pricing-Offloading": (0, (5, 1)),
    "Stackelberg Joint Pricing-Offloading": (0, (2, 1)),
    "Queue-Unaware QH-MAL": "-",
    "Queue-Aware MADDPG": (0, (4, 1, 1, 1)),
    "Queue-Aware MATD3": (0, (6, 2)),
    "Proposed Method": "-",
}

METHOD_MARKERS = {
    "Random Offloading": "o",
    "Greedy Offloading": "s",
    "Fixed-Pricing Queue-Aware Greedy": "^",
    "DPP Joint Pricing-Offloading": "P",
    "Stackelberg Joint Pricing-Offloading": "h",
    "Queue-Unaware QH-MAL": "D",
    "Queue-Aware MADDPG": "v",
    "Queue-Aware MATD3": "*",
    "Proposed Method": "X",
}


def _resolve_series_order(df: pd.DataFrame, column: str, preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    observed = [str(item) for item in pd.Series(df[column]).dropna().unique().tolist()]
    ordered = [item for item in preferred if item in observed]
    ordered.extend(item for item in observed if item not in ordered)
    return ordered


def _method_color(label: str) -> str:
    return METHOD_PALETTE.get(label, "#444444")


def _method_linestyle(label: str):
    return METHOD_LINESTYLES.get(label, "-")


def _method_marker(label: str) -> str:
    return METHOD_MARKERS.get(label, "o")


def _legend_handles(order: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for label in order:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_method_color(label),
                linestyle=_method_linestyle(label),
                marker=_method_marker(label),
                linewidth=2.4,
                markersize=6,
                label=label,
            )
        )
    return handles


def _apply_metric_scale(ax: plt.Axes, metric: str) -> None:
    metric = metric.removeprefix("plot_")
    if metric == "avg_queue_backlog":
        ax.set_yscale("symlog", linthresh=1.0)
    elif metric == "avg_waiting_delay":
        ax.set_yscale("symlog", linthresh=1.0e-6)


def _format_axis_label(name: str) -> str:
    name = name.removeprefix("plot_")
    mapping = {
        "lambda_q": "Queue Penalty $\\lambda_q$",
        "lambda_f": "Fairness Penalty $\\lambda_f$",
        "arrival_scale": "Arrival Scale",
        "p95_delay": "P95 Delay",
        "avg_profit": "Server Return",
        "violation_ratio": "Violation Ratio",
        "avg_queue_backlog": "Queue Backlog",
        "avg_waiting_delay": "Waiting Delay",
        "fairness": "Fairness",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _plot_method_lines(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, series_order: list[str], markers: bool) -> None:
    for label in series_order:
        subset = df[df["method"] == label].sort_values(x_col)
        if subset.empty:
            continue
        plot_kwargs = {
            "color": _method_color(label),
            "linestyle": _method_linestyle(label),
            "linewidth": 2.6 if label == "Proposed Method" else 2.2,
            "label": label,
        }
        if markers:
            plot_kwargs["marker"] = _method_marker(label)
            plot_kwargs["markersize"] = 6
        ax.plot(subset[x_col], subset[y_col], **plot_kwargs)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def plot_training_curve(
    df: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    y_label: str,
    baseline_lines: dict[str, float] | None = None,
    series_order: list[str] | None = None,
) -> None:
    ensure_dir(out_path.parent)
    resolved_order = _resolve_series_order(df, "method", series_order or DEFAULT_METHOD_ORDER)

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    _plot_method_lines(ax, df, "episode", value_col, resolved_order, markers=False)

    if baseline_lines:
        baseline_order = [name for name in DEFAULT_METHOD_ORDER if name in baseline_lines]
        baseline_order.extend(name for name in baseline_lines if name not in baseline_order)
        for label in baseline_order:
            value = baseline_lines[label]
            ax.axhline(
                value,
                color=_method_color(label),
                linestyle=_method_linestyle(label),
                linewidth=1.8,
                alpha=0.95,
            )

    _apply_metric_scale(ax, value_col)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    handles = _legend_handles(resolved_order + [name for name in DEFAULT_METHOD_ORDER if baseline_lines and name in baseline_lines and name not in resolved_order])
    ax.legend(handles=handles, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_metric_vs_x(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    out_path: Path,
    title: str,
    series_order: list[str] | None = None,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    if hue_col == "method":
        resolved_order = _resolve_series_order(df, hue_col, series_order or DEFAULT_METHOD_ORDER)
        _plot_method_lines(ax, df, x_col, y_col, resolved_order, markers=True)
        ax.legend(handles=_legend_handles(resolved_order), loc="best", fontsize=8)
    else:
        subset = df.sort_values(x_col)
        ax.plot(subset[x_col], subset[y_col], color="#1f77b4", linewidth=2.4, marker="o", markersize=6)
    _apply_metric_scale(ax, y_col)
    ax.set_title(title)
    ax.set_xlabel(_format_axis_label(x_col))
    ax.set_ylabel(_format_axis_label(y_col))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_arrival_dashboard(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    series_order: list[str] | None = None,
) -> None:
    ensure_dir(out_path.parent)
    resolved_order = _resolve_series_order(df, "method", series_order or DEFAULT_METHOD_ORDER)
    metrics = [
        ("p95_delay", "P95 Delay"),
        ("avg_profit", "Server Return"),
        ("violation_ratio", "Violation Ratio"),
        ("avg_queue_backlog", "Queue Backlog"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True)
    for ax, (metric, label) in zip(axes.flatten(), metrics):
        _plot_method_lines(ax, df, "arrival_scale", metric, resolved_order, markers=True)
        _apply_metric_scale(ax, metric)
        ax.set_title(label)
        ax.set_xlabel("Arrival Scale")
        ax.set_ylabel(label)
    handles = _legend_handles(resolved_order)
    fig.suptitle(title, y=0.98)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.01), fontsize=9)
    fig.tight_layout(rect=(0.02, 0.08, 0.985, 0.955))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_penalty_dashboard(
    df: pd.DataFrame,
    x_col: str,
    out_path: Path,
    title: str,
    left_metric: str,
    left_label: str,
    right_metric: str,
    right_label: str,
    default_x: float = 0.2,
) -> None:
    ensure_dir(out_path.parent)
    subset = df.sort_values(x_col)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharex=True)
    for ax, metric, label, color in [
        (axes[0], left_metric, left_label, "#1f77b4"),
        (axes[1], right_metric, right_label, "#d62728"),
    ]:
        ax.plot(subset[x_col], subset[metric], color=color, linewidth=2.6, marker="o", markersize=6)
        ax.fill_between(subset[x_col], subset[metric], color=color, alpha=0.12)
        ax.axvline(default_x, color="#444444", linestyle="--", linewidth=1.2)
        default_row = subset.iloc[(subset[x_col] - default_x).abs().argmin()]
        ax.scatter([default_row[x_col]], [default_row[metric]], color="#111111", s=28, zorder=5)
        _apply_metric_scale(ax, metric)
        ax.set_title(label)
        ax.set_xlabel(_format_axis_label(x_col))
        ax.set_ylabel(label)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.03, 0.02, 0.985, 0.94))
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
