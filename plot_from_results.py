from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
MULTI = ROOT / "experiments" / "multiseed" / "paper_results"
USER = ROOT / "user_sensitivity_package_final" / "results" / "user_sensitivity_overall.csv"
BUFFER = ROOT / "buffer_stress_results" / "buffer_capacity_sensitivity.csv"
OUT = ROOT / "paper_assets"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.2,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    }
)


METHODS = [
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
SHORT = {
    "Random Offloading": "Random",
    "Greedy Offloading": "Greedy",
    "Fixed-Pricing Queue-Aware Greedy": "Fixed-QA",
    "DPP Joint Pricing-Offloading": "DPP-JPO",
    "Stackelberg Joint Pricing-Offloading": "SG-JPO",
    "Queue-Unaware QH-MAL": "QU-QH-MAL",
    "Queue-Aware MADDPG": "QA-MADDPG",
    "Queue-Aware MATD3": "QA-MATD3",
    "Proposed Method": "QH-MAL",
}
COLORS = {
    "Random Offloading": "#B9BDC5",
    "Greedy Offloading": "#9EA4AE",
    "Fixed-Pricing Queue-Aware Greedy": "#7F8792",
    "DPP Joint Pricing-Offloading": "#D39C3F",
    "Stackelberg Joint Pricing-Offloading": "#B97832",
    "Queue-Unaware QH-MAL": "#C85A5A",
    "Queue-Aware MADDPG": "#6A83B8",
    "Queue-Aware MATD3": "#8A6DB0",
    "Proposed Method": "#159D8C",
}


def canonicalize_public_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept legacy result tables while exposing the paper's final terminology."""

    renames: dict[str, str] = {}
    for column in frame.columns:
        if column == "Profit":
            canonical = "Server Return"
        elif column.startswith("Profit "):
            canonical = "Server Return" + column[len("Profit") :]
        else:
            continue
        if canonical in frame.columns:
            raise ValueError(
                f"Result table contains both legacy column {column!r} and {canonical!r}."
            )
        renames[column] = canonical
    return frame.rename(columns=renames)


def ci95(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(t.ppf(0.975, arr.size - 1) * arr.std(ddof=1) / np.sqrt(arr.size))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def overall_figure(all_runs: pd.DataFrame) -> None:
    df = all_runs.copy()
    summaries = []
    for method in METHODS:
        d = df[df["Method"] == method]
        row = {"Method": method}
        for metric in ["Avg Delay", "Server Return", "Queue Backlog", "Decision Time (ms)"]:
            row[f"{metric} Mean"] = d[metric].mean()
            row[f"{metric} CI"] = ci95(d[metric])
        summaries.append(row)
    s = pd.DataFrame(summaries).set_index("Method").loc[METHODS]

    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.8), constrained_layout=True)
    specs = [
        ("Avg Delay", "Average delay (s)", False),
        ("Server Return", "Server return", False),
        ("Queue Backlog", "Mean backlog ($10^6$ cycles)", False),
        ("Decision Time (ms)", "Decision time (ms, log scale)", True),
    ]
    y = np.arange(len(METHODS))
    for idx, (ax, (metric, xlabel, log_scale)) in enumerate(zip(axes.flat, specs)):
        vals = s[f"{metric} Mean"].to_numpy().copy()
        errs = s[f"{metric} CI"].to_numpy().copy()
        if metric == "Queue Backlog":
            vals /= 1e6
            errs /= 1e6
        ax.barh(
            y,
            vals,
            xerr=errs,
            color=[COLORS[m] for m in METHODS],
            edgecolor="white",
            linewidth=0.35,
            capsize=1.7,
            error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
        )
        ax.set_yticks(y)
        if idx % 2 == 0:
            ax.set_yticklabels([SHORT[m] for m in METHODS])
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        if log_scale:
            ax.set_xscale("log")
        panel_label(ax, f"({chr(97 + idx)})")
    save(fig, "fig2_overall")


def queue_ablation_figure(all_runs: pd.DataFrame) -> None:
    qh = all_runs[all_runs["Method"] == "Proposed Method"].set_index("Seed")
    qu = all_runs[all_runs["Method"] == "Queue-Unaware QH-MAL"].set_index("Seed")
    metrics = [
        ("Avg Delay", "Avg delay", "lower"),
        ("P95 Delay", "P95 delay", "lower"),
        ("Energy", "Energy", "lower"),
        ("Payment", "User payment", "lower"),
        ("Server Return", "Server return", "higher"),
        ("Fairness", "Fairness", "higher"),
        ("Queue Backlog", "Queue backlog", "lower"),
        ("Wait Delay", "Wait delay", "lower"),
        ("Violation Ratio", "Violation ratio", "lower"),
        ("Max Buffer Occupancy", "Max occupancy", "lower"),
        ("Local Fallback Ratio", "Fallback ratio", "lower"),
        ("CPU Utilization", "CPU utilization", "higher"),
    ]
    means, cis, labels = [], [], []
    for col, label, direction in metrics:
        if direction == "lower":
            gain = 100 * (qu[col] - qh[col]) / qu[col].abs()
        else:
            gain = 100 * (qh[col] - qu[col]) / qu[col].abs()
        means.append(gain.mean())
        cis.append(ci95(gain))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(7.12, 3.35), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        means,
        xerr=cis,
        color=["#159D8C" if v >= 0 else "#C85A5A" for v in means],
        alpha=0.92,
        capsize=2,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative improvement of QH-MAL over QU-QH-MAL (%)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, means):
        x = value + (0.7 if value >= 0 else -0.7)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=6.5,
        )
    save(fig, "fig3_queue_ablation")


def aggregate_arrival() -> pd.DataFrame:
    frames = []
    for seed in [7, 11, 19, 23, 31]:
        p = MULTI / f"seed_{seed}" / "results" / "arrival_sensitivity.csv"
        d = pd.read_csv(p)
        d["Seed"] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def line_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    ylabel: str,
    methods: list[str],
    scale: float = 1.0,
    with_ci: bool = True,
) -> None:
    markers = ["o", "s", "^", "D"]
    for method, marker in zip(methods, markers):
        d = df[df["method"] == method]
        xs = sorted(d[xcol].unique())
        means, cis = [], []
        for x in xs:
            vals = d.loc[d[xcol] == x, ycol] / scale
            means.append(vals.mean())
            cis.append(ci95(vals) if with_ci else 0.0)
        ax.errorbar(
            xs,
            means,
            yerr=cis if with_ci else None,
            label=SHORT[method],
            marker=marker,
            color=COLORS[method],
            capsize=2,
            elinewidth=0.75,
        )
    ax.set_xlabel("Arrival scale" if xcol == "arrival_scale" else "Number of users")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)


def arrival_figure(arrival: pd.DataFrame) -> None:
    methods = [
        "Queue-Unaware QH-MAL",
        "Queue-Aware MADDPG",
        "Queue-Aware MATD3",
        "Proposed Method",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.55))
    specs = [
        ("avg_delay", "Average delay (s)", 1.0),
        ("avg_profit", "Server return", 1.0),
        ("avg_queue_backlog", "Mean backlog ($10^6$ cycles)", 1e6),
        ("fallback_ratio", "Fallback ratio (%)", 0.01),
    ]
    for idx, (ax, (metric, ylabel, scale)) in enumerate(zip(axes.flat, specs)):
        line_panel(ax, arrival, "arrival_scale", metric, ylabel, methods, scale=scale)
        panel_label(ax, f"({chr(97 + idx)})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.88, wspace=0.30, hspace=0.34)
    save(fig, "fig4_arrival_sensitivity")


def user_figure() -> None:
    d = canonicalize_public_metric_columns(pd.read_csv(USER)).rename(
        columns={
            "Method": "method",
            "Users": "users",
            "Avg Delay": "avg_delay",
            "P95 Delay": "p95_delay",
            "Server Return": "avg_profit",
            "Queue Backlog": "avg_queue_backlog",
        }
    )
    methods = [
        "Queue-Unaware QH-MAL",
        "Queue-Aware MADDPG",
        "Queue-Aware MATD3",
        "Proposed Method",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.55))
    specs = [
        ("avg_delay", "Average delay (s)", 1.0),
        ("p95_delay", "P95 delay (s)", 1.0),
        ("avg_profit", "Server return", 1.0),
        ("avg_queue_backlog", "Mean backlog ($10^6$ cycles)", 1e6),
    ]
    for idx, (ax, (metric, ylabel, scale)) in enumerate(zip(axes.flat, specs)):
        line_panel(ax, d, "users", metric, ylabel, methods, scale=scale, with_ci=False)
        ax.set_xticks([20, 30, 40])
        panel_label(ax, f"({chr(97 + idx)})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.88, wspace=0.30, hspace=0.34)
    save(fig, "fig5_user_sensitivity")


def buffer_figure() -> None:
    d = pd.read_csv(BUFFER)
    x = d["buffer_window_s"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.12, 4.35), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, d["avg_delay"], marker="o", label="Average")
    ax.plot(x, d["p95_delay"], marker="s", label="P95")
    ax.set_ylabel("Delay (s)")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(x, d["avg_energy"], marker="o", color="#C85A5A")
    ax.set_ylabel("Energy")
    ax = axes[1, 0]
    ax.plot(x, d["avg_profit"], marker="o", color="#159D8C")
    ax.set_ylabel("Server return")
    ax = axes[1, 1]
    ax.plot(x, 100 * d["fallback_ratio"], marker="o", label="Fallback")
    ax.plot(x, 100 * d["max_queue_occupancy"], marker="s", label="Max occupancy")
    ax.set_ylabel("Ratio (%)")
    ax.legend(frameon=False)
    for idx, ax in enumerate(axes.flat):
        ax.set_xlabel("Buffer window (s)")
        ax.set_xticks(x)
        ax.grid(True)
        ax.set_axisbelow(True)
        panel_label(ax, f"({chr(97 + idx)})")
    save(fig, "fig6_buffer_sensitivity")


def ablation_figure() -> None:
    d = canonicalize_public_metric_columns(
        pd.read_csv(MULTI / "ablation_relative_gain_mean_std.csv")
    )
    variants = [
        "w/o Queue Observations",
        "w/o Queue Penalty",
        "w/o Fairness Penalty",
        "w/o Slower Server Updates",
        "w/o Role-Specific Actors",
    ]
    d = d.set_index("Variant").loc[variants].reset_index()
    metrics = [
        ("Avg Delay", "Avg delay"),
        ("P95 Delay", "P95 delay"),
        ("Server Return", "Server return"),
        ("Fairness", "Fairness"),
        ("Queue Backlog", "Backlog"),
        ("Wait Delay", "Wait delay"),
        ("Violation Ratio", "Violation"),
    ]
    means = np.column_stack(
        [d[f"{m} Relative Gain (%) Mean"].to_numpy() for m, _ in metrics]
    )
    stds = np.column_stack(
        [d[f"{m} Relative Gain (%) Std"].to_numpy() for m, _ in metrics]
    )
    row_labels = [
        "No queue observations",
        "No queue penalty",
        "No fairness penalty",
        "No slower server updates",
        "No role-specific actors",
    ]
    fig, ax = plt.subplots(figsize=(7.12, 2.65), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=85)
    im = ax.imshow(means, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            value = means[i, j]
            text_color = "white" if value > 55 or value < -7 else "#222222"
            ax.text(
                j,
                i,
                f"{value:.1f}\n$\\pm${stds[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Gain of full QH-MAL (%)")
    ax.set_xlabel("Positive values favor the full method")
    save(fig, "fig7_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the six figures used in the final manuscript from raw experiment tables."
    )
    parser.add_argument(
        "--multiseed-dir",
        type=Path,
        default=Path("experiments/multiseed/paper_results"),
        help="Directory containing all_runs_overall.csv and per-seed result folders.",
    )
    parser.add_argument(
        "--user-results",
        type=Path,
        default=Path("user_sensitivity_package_final/results/user_sensitivity_overall.csv"),
    )
    parser.add_argument(
        "--buffer-results",
        type=Path,
        default=Path("buffer_stress_results/buffer_capacity_sensitivity.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path

    global MULTI, USER, BUFFER, OUT
    MULTI = resolve(args.multiseed_dir)
    USER = resolve(args.user_results)
    BUFFER = resolve(args.buffer_results)
    OUT = resolve(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    required = [
        MULTI / "all_runs_overall.csv",
        USER,
        BUFFER,
        MULTI / "ablation_relative_gain_mean_std.csv",
    ] + [
        MULTI / f"seed_{seed}" / "results" / "arrival_sensitivity.csv"
        for seed in [7, 11, 19, 23, 31]
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing reproduction inputs:\n{formatted}")

    all_runs = canonicalize_public_metric_columns(
        pd.read_csv(MULTI / "all_runs_overall.csv")
    )
    missing_methods = [method for method in METHODS if method not in set(all_runs["Method"])]
    if missing_methods:
        raise ValueError(f"all_runs_overall.csv is missing methods: {missing_methods}")
    observed_seeds = sorted(int(seed) for seed in all_runs["Seed"].unique())
    expected_seeds = [7, 11, 19, 23, 31]
    if observed_seeds != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, found {observed_seeds}")
    overall_figure(all_runs)
    queue_ablation_figure(all_runs)
    arrival_figure(aggregate_arrival())
    user_figure()
    buffer_figure()
    ablation_figure()


if __name__ == "__main__":
    main()
