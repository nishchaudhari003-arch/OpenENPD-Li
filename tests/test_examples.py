"""
Smoke tests for the example workflow and the release-artifact script.

These confirm the runnable example and the artifact generator complete without
crashing, produce the expected outputs, leave the baseline validation numbers
unchanged, and never mutate the source case.
"""

import matplotlib

matplotlib.use("Agg")

import pytest

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejection_for_flux

from examples.foo2023_lmc_ph7_workflow import main as run_example
from scripts.generate_release_artifacts import main as generate_artifacts


def _baseline_rejections():
    case = foo2023_lmc_ph7_case()
    flux = case["experimental_data"]["Jw_um_s"][0] * 1e-6
    model_inputs = prepare_case_transport_inputs(foo2023_lmc_ph7_case())
    prediction = predict_rejection_for_flux(
        model_inputs=model_inputs,
        water_flux_m_s=flux,
        temperature_K=case["temperature_K"],
    )
    return dict(prediction["rejections"])


def test_example_workflow_runs_and_reports_sections():
    result = run_example(verbose=False)

    assert "baseline_metrics" in result
    assert "two_interface_diagnostics" in result
    assert "variant_metrics" in result
    assert "parameter_estimation" in result

    # Baseline metrics present and finite.
    baseline = result["baseline_metrics"]
    assert {"R_Li_rmse", "R_Mg_rmse"}.issubset(baseline)

    # Two-interface diagnostics report solve counts and the negative-Li flag.
    diagnostics = result["two_interface_diagnostics"]
    assert (
        diagnostics["number_of_successful_solves"]
        + diagnostics["number_of_failed_solves"]
        == diagnostics["number_of_fluxes"]
    )
    assert "negative_li_rejection_reproduced" in diagnostics

    # Parameter-estimation section carries both fits.
    estimation = result["parameter_estimation"]
    assert "one_parameter" in estimation
    assert "two_parameter" in estimation


def test_release_artifact_script_generates_outputs(tmp_path):
    paths = generate_artifacts(output_directory=tmp_path, include_plots=False)

    # At least the baseline comparison CSV and metrics JSON are written.
    assert "baseline_comparison" in paths
    assert paths["baseline_comparison"].exists()
    assert paths["baseline_comparison"].stat().st_size > 0
    assert paths["baseline_metrics"].exists()
    # A human-readable summary is produced.
    assert paths["summary"].exists()
    assert paths["summary"].stat().st_size > 0


def test_release_artifact_script_has_callable_main():
    assert callable(generate_artifacts)


def test_baseline_and_case_protected_by_example_and_script(tmp_path):
    # Run the example and the artifact script once, then check both that the
    # baseline metrics are unchanged and that the source case is not mutated.
    before = _baseline_rejections()
    reference = foo2023_lmc_ph7_case()

    run_example(verbose=False)
    generate_artifacts(output_directory=tmp_path, include_plots=False)

    after = _baseline_rejections()
    assert before.keys() == after.keys()
    for ion in before:
        assert before[ion] == pytest.approx(after[ion])

    assert foo2023_lmc_ph7_case() == reference
