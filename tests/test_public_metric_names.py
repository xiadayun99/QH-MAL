from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plot_from_results import canonicalize_public_metric_columns


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_return_columns_are_read_without_changing_values() -> None:
    legacy = pd.DataFrame(
        {
            "Method": ["Proposed Method"],
            "Profit": [5.7942],
            "Profit Relative Gain (%) Mean": [30.22],
        }
    )

    current = canonicalize_public_metric_columns(legacy)

    assert "Profit" not in current.columns
    assert current.loc[0, "Server Return"] == pytest.approx(5.7942)
    assert current.loc[0, "Server Return Relative Gain (%) Mean"] == pytest.approx(
        30.22
    )


def test_ambiguous_legacy_and_current_columns_are_rejected() -> None:
    ambiguous = pd.DataFrame({"Profit": [1.0], "Server Return": [1.0]})

    with pytest.raises(ValueError, match="both legacy column"):
        canonicalize_public_metric_columns(ambiguous)


def test_reference_results_use_final_manuscript_term() -> None:
    overall = pd.read_csv(ROOT / "reference_results" / "overall_performance_reference.csv")

    assert "Server Return Mean" in overall.columns
    assert "Profit Mean" not in overall.columns
