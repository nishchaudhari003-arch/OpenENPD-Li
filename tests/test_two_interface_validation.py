"""
Tests for the two-interface Foo 2023 LM-C pH 7 validation diagnostics.

These verify the comparison table, DataFrame, metrics, and solver-diagnostic
summary. They assert the workflow is honest: failed solves stay explicit,
successful-only RMSE is None when nothing converges, and the negative-Li
diagnostic reflects the actual predictions. They do NOT assert that the
two-interface model converges or reproduces any rejection sign.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejection_for_flux
from openenpd.plots import (
    plot_two_interface_foo2023_lmc_ph7_rejection_comparison,
    plot_two_interface_foo2023_lmc_ph7_scaled_residual,
    plot_two_interface_foo2023_lmc_ph7_solve_success,
)
from openenpd.two_interface_validation import (
    COMPARISON_COLUMNS,
    two_interface_foo2023_lmc_ph7_comparison_dataframe,
    two_interface_foo2023_lmc_ph7_comparison_rows,
    two_interface_foo2023_lmc_ph7_metrics,
    two_interface_foo2023_lmc_ph7_solver_diagnostics,
)


# ---------------------------------------------------------------------------
# 1. comparison rows
# ---------------------------------------------------------------------------

def test_comparison_rows_return_four_rows_with_required_keys():
    rows = two_interface_foo2023_lmc_ph7_comparison_rows()
    assert len(rows) == 4
    for row in rows:
        for key in COMPARISON_COLUMNS:
            assert key in row


def test_comparison_rows_experimental_values_match_case():
    case = foo2023_lmc_ph7_case()
    experimental = case["experimental_data"]
    rows = two_interface_foo2023_lmc_ph7_comparison_rows()

    for index, row in enumerate(rows):
        assert row["pressure_bar"] == experimental["pressure_bar"][index]
        assert row["Jw_LMH"] == experimental["Jw_LMH"][index]
        assert row["Jw_um_s"] == experimental["Jw_um_s"][index]
        assert row["experimental_R_Li"] == experimental["R_Li"][index]
        assert row["experimental_R_Mg"] == experimental["R_Mg"][index]


def test_comparison_rows_have_baseline_and_two_interface_diagnostics():
    rows = two_interface_foo2023_lmc_ph7_comparison_rows()
    for row in rows:
        # Baseline predictions present and finite.
        assert math.isfinite(row["baseline_R_Li"])
        assert math.isfinite(row["baseline_R_Mg"])
        # Two-interface diagnostics present.
        assert isinstance(row["two_interface_success"], bool)
        assert isinstance(row["two_interface_message"], str) and row["two_interface_message"]
        assert row["two_interface_R_Li"] is not None
        assert row["two_interface_R_Mg"] is not None
        assert math.isfinite(row["two_interface_scaled_residual_norm"])
        assert math.isfinite(row["two_interface_scaled_residual_max_abs"])
        assert math.isfinite(row["two_interface_raw_residual_norm"])


# ---------------------------------------------------------------------------
# 2. dataframe
# ---------------------------------------------------------------------------

def test_dataframe_shape_and_columns():
    dataframe = two_interface_foo2023_lmc_ph7_comparison_dataframe()
    assert dataframe.shape[0] == 4
    # Deterministic column order.
    assert list(dataframe.columns) == list(COMPARISON_COLUMNS)


def test_dataframe_is_deterministic():
    first = two_interface_foo2023_lmc_ph7_comparison_dataframe()
    second = two_interface_foo2023_lmc_ph7_comparison_dataframe()
    # Numeric columns compare equal element-wise.
    for column in [
        "baseline_R_Li",
        "two_interface_R_Li",
        "two_interface_scaled_residual_max_abs",
    ]:
        assert list(first[column]) == list(second[column])


# ---------------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------------

def test_baseline_metrics_are_finite():
    metrics = two_interface_foo2023_lmc_ph7_metrics()
    assert math.isfinite(metrics["baseline_R_Li_rmse"])
    assert math.isfinite(metrics["baseline_R_Mg_rmse"])
    assert math.isfinite(metrics["baseline_total_rmse"])


def test_best_effort_metrics_finite_and_labeled_diagnostic():
    metrics = two_interface_foo2023_lmc_ph7_metrics()
    assert math.isfinite(metrics["two_interface_best_effort_R_Li_rmse"])
    assert math.isfinite(metrics["two_interface_best_effort_R_Mg_rmse"])
    assert math.isfinite(metrics["two_interface_best_effort_total_rmse"])
    assert metrics["best_effort_is_diagnostic_only"] is True


def test_successful_only_metrics_none_when_no_successful_solves():
    metrics = two_interface_foo2023_lmc_ph7_metrics()
    if metrics["number_of_successful_two_interface_solves"] == 0:
        assert metrics["two_interface_successful_R_Li_rmse"] is None
        assert metrics["two_interface_successful_R_Mg_rmse"] is None
        assert metrics["two_interface_successful_total_rmse"] is None
    else:
        assert math.isfinite(metrics["two_interface_successful_total_rmse"])


def test_solve_counts_sum_to_four():
    metrics = two_interface_foo2023_lmc_ph7_metrics()
    assert metrics["number_of_fluxes"] == 4
    assert (
        metrics["number_of_successful_two_interface_solves"]
        + metrics["number_of_failed_two_interface_solves"]
        == 4
    )


# ---------------------------------------------------------------------------
# 4. failure transparency
# ---------------------------------------------------------------------------

def test_failed_solves_remain_explicit_and_not_dropped():
    rows = two_interface_foo2023_lmc_ph7_comparison_rows()
    diagnostics = two_interface_foo2023_lmc_ph7_solver_diagnostics()

    # Every flux is represented (nothing dropped), regardless of success.
    assert len(rows) == 4
    assert (
        diagnostics["number_of_successful_solves"]
        + diagnostics["number_of_failed_solves"]
        == 4
    )
    # Any failed row keeps success=False and a non-empty message and still
    # carries best-effort rejections rather than being removed.
    failed = [r for r in rows if not r["two_interface_success"]]
    for row in failed:
        assert row["two_interface_success"] is False
        assert row["two_interface_message"]
        assert row["two_interface_R_Li"] is not None


# ---------------------------------------------------------------------------
# 5. hard-cutoff diagnostics
# ---------------------------------------------------------------------------

def test_hard_cutoff_species_present_in_each_row():
    rows = two_interface_foo2023_lmc_ph7_comparison_rows()
    for row in rows:
        assert "hard_cutoff_species" in row
        # Under the current physics Mg2+ is the size-excluded species.
        assert "Mg2+" in row["hard_cutoff_species"]


def test_hard_cutoff_species_in_diagnostics_summary():
    diagnostics = two_interface_foo2023_lmc_ph7_solver_diagnostics()
    assert len(diagnostics["hard_cutoff_species_by_flux"]) == 4
    for cutoff in diagnostics["hard_cutoff_species_by_flux"]:
        assert "Mg2+" in cutoff


# ---------------------------------------------------------------------------
# 6. negative Li rejection diagnostic
# ---------------------------------------------------------------------------

def test_negative_li_rejection_diagnostic_recorded():
    diagnostics = two_interface_foo2023_lmc_ph7_solver_diagnostics()
    assert "negative_li_rejection_reproduced" in diagnostics
    assert isinstance(diagnostics["negative_li_rejection_reproduced"], bool)
    assert "negative_li_rejection_in_successful_solve" in diagnostics

    rows = two_interface_foo2023_lmc_ph7_comparison_rows()
    expected_any_negative = any(
        row["two_interface_R_Li"] is not None and row["two_interface_R_Li"] < 0.0
        for row in rows
    )
    assert diagnostics["negative_li_rejection_reproduced"] == expected_any_negative
    # The summary text mentions negative Li rejection explicitly.
    assert "negative li rejection" in diagnostics["summary_text"].lower()


# ---------------------------------------------------------------------------
# 7. plot tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "plot_function",
    [
        plot_two_interface_foo2023_lmc_ph7_rejection_comparison,
        plot_two_interface_foo2023_lmc_ph7_scaled_residual,
        plot_two_interface_foo2023_lmc_ph7_solve_success,
    ],
)
def test_two_interface_plots_return_figure_and_axes(plot_function):
    figure, axes = plot_function()
    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    plt.close(figure)


@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (plot_two_interface_foo2023_lmc_ph7_rejection_comparison, "ti_comparison.png"),
        (plot_two_interface_foo2023_lmc_ph7_scaled_residual, "ti_residual.png"),
        (plot_two_interface_foo2023_lmc_ph7_solve_success, "ti_success.png"),
    ],
)
def test_two_interface_plots_save_nonempty_file(plot_function, filename, tmp_path):
    output_path = tmp_path / filename
    figure, _ = plot_function(output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(figure)


# ---------------------------------------------------------------------------
# 8. no mutation
# ---------------------------------------------------------------------------

def test_case_dictionary_factory_unchanged_by_validation_workflow():
    reference = foo2023_lmc_ph7_case()

    two_interface_foo2023_lmc_ph7_comparison_rows()
    two_interface_foo2023_lmc_ph7_metrics()
    two_interface_foo2023_lmc_ph7_solver_diagnostics()

    assert foo2023_lmc_ph7_case() == reference


# ---------------------------------------------------------------------------
# 9. baseline protection
# ---------------------------------------------------------------------------

def test_baseline_solver_output_unchanged_by_validation_workflow():
    case = foo2023_lmc_ph7_case()
    flux = case["experimental_data"]["Jw_um_s"][0] * 1e-6

    def baseline_rejections():
        model_inputs = prepare_case_transport_inputs(foo2023_lmc_ph7_case())
        prediction = predict_rejection_for_flux(
            model_inputs=model_inputs,
            water_flux_m_s=flux,
            temperature_K=case["temperature_K"],
        )
        return dict(prediction["rejections"])

    before = baseline_rejections()
    two_interface_foo2023_lmc_ph7_metrics()  # run the full workflow
    after = baseline_rejections()

    assert before.keys() == after.keys()
    for ion in before:
        assert before[ion] == pytest.approx(after[ion])
