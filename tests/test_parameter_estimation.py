"""
Tests for bounded parameter estimation and identifiability diagnostics.

These verify the estimation configuration/validation, objective construction,
one- and two-parameter fits, bound handling, identifiability diagnostics,
objective scans, the negative-Li diagnostic, immutability, baseline protection,
and the documentation smoke test. They keep runs fast with small seed grids and
low ``max_nfev``, and do NOT assert validation success or negative-Li
reproduction.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.constitutive_variants import ConstitutiveVariantConfig
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejection_for_flux
from openenpd.plots import (
    plot_objective_scan_1d,
    plot_objective_scan_2d,
    plot_parameter_fit_rejection,
)
from openenpd.parameter_estimation import (
    EstimationConfig,
    EstimationResult,
    ParameterSpec,
    build_foo2023_estimation_objective,
    fit_foo2023_parameters,
    foo2023_parameter_estimation_summary,
    objective_scan_1d,
    objective_scan_2d,
    parameter_spec,
    run_foo2023_single_parameter_fits,
    run_foo2023_two_parameter_fits,
)


def _fast_config(*names):
    return EstimationConfig(
        parameter_specs=tuple(parameter_spec(name) for name in names),
        seed_grid_points=3,
        max_nfev=20,
    )


# ---------------------------------------------------------------------------
# 1. ParameterSpec and EstimationConfig
# ---------------------------------------------------------------------------

def test_parameter_spec_factory_uses_registry_defaults():
    spec = parameter_spec("ion_radius_scale")
    assert spec.name == "ion_radius_scale"
    assert spec.lower_bound == 0.40
    assert spec.upper_bound == 1.20
    assert spec.lower_bound <= spec.initial_value <= spec.upper_bound


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "not_a_parameter", "initial_value": 1.0, "lower_bound": 0.0, "upper_bound": 1.0},
        {"name": "ion_radius_scale", "initial_value": 1.0, "lower_bound": 1.0, "upper_bound": 1.0},
        {"name": "ion_radius_scale", "initial_value": 1.0, "lower_bound": -0.1, "upper_bound": 1.2},
        {"name": "ion_radius_scale", "initial_value": 5.0, "lower_bound": 0.4, "upper_bound": 1.2},
    ],
)
def test_invalid_parameter_spec_raises(kwargs):
    with pytest.raises(ValueError):
        ParameterSpec(**kwargs)


def test_parameter_spec_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        parameter_spec("mystery_parameter")


def test_estimation_config_valid_and_invalid():
    config = _fast_config("ion_radius_scale")
    assert config.number_of_parameters == 1
    # Duplicate names.
    with pytest.raises(ValueError):
        EstimationConfig(
            parameter_specs=(parameter_spec("ion_radius_scale"), parameter_spec("ion_radius_scale"))
        )
    # Unknown species.
    with pytest.raises(ValueError):
        EstimationConfig(
            parameter_specs=(parameter_spec("ion_radius_scale"),),
            species_to_fit=("Na+",),
        )
    # Negative penalty.
    with pytest.raises(ValueError):
        EstimationConfig(
            parameter_specs=(parameter_spec("ion_radius_scale"),),
            failed_solve_penalty=-1.0,
        )


def test_more_than_two_parameters_triggers_identifiability_warning():
    config = EstimationConfig(
        parameter_specs=(
            parameter_spec("ion_radius_scale"),
            parameter_spec("pore_radius_scale"),
            parameter_spec("hindrance_floor"),
        )
    )
    warnings = config.identifiability_warnings()
    assert any("parameters" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# 2. objective construction
# ---------------------------------------------------------------------------

def test_objective_vector_length_and_finiteness():
    config = _fast_config("pore_radius_scale")
    objective, metadata = build_foo2023_estimation_objective(config)
    # 8 rejection residuals (Li/Mg x 4 fluxes) + 4 per-flux penalty slots.
    vector = objective([1.25])
    assert len(vector) == 8 + 4
    assert metadata["number_of_data_points"] == 8
    assert all(math.isfinite(v) for v in vector)


def test_objective_penalty_slots_flag_failed_solves():
    config = _fast_config("pore_radius_scale")
    objective, _ = build_foo2023_estimation_objective(config)
    # At pore scale 1.0 the Mg hard cutoff persists and all four solves fail,
    # so every per-flux penalty slot equals the failed-solve penalty.
    vector = objective([1.0])
    penalty_slots = vector[-4:]
    assert penalty_slots == [config.failed_solve_penalty] * 4


# ---------------------------------------------------------------------------
# 3. one-parameter fit
# ---------------------------------------------------------------------------

def test_one_parameter_fit_runs_and_reports_fields():
    result = fit_foo2023_parameters(_fast_config("pore_radius_scale"))
    assert isinstance(result, EstimationResult)
    assert "pore_radius_scale" in result.parameter_values
    assert isinstance(result.success, bool)
    assert result.total_rmse is None or math.isfinite(result.total_rmse)
    assert result.number_of_data_points == 8
    assert result.degrees_of_freedom == 7
    assert (
        result.number_of_successful_solves + result.number_of_failed_solves == 4
    )
    assert len(result.residual_vector) == 8
    assert "warnings" in result.identifiability_diagnostics


# ---------------------------------------------------------------------------
# 4. two-parameter fit
# ---------------------------------------------------------------------------

def test_two_parameter_fit_runs_with_identifiability_diagnostics():
    result = fit_foo2023_parameters(_fast_config("ion_radius_scale", "pore_radius_scale"))
    assert set(result.parameter_values.keys()) == {"ion_radius_scale", "pore_radius_scale"}
    assert result.number_of_parameters == 2
    assert result.degrees_of_freedom == 6
    diagnostics = result.identifiability_diagnostics
    # A condition number is available when the optimizer returns a Jacobian.
    assert diagnostics["jacobian_condition_number"] is None or math.isfinite(
        diagnostics["jacobian_condition_number"]
    )


# ---------------------------------------------------------------------------
# 5. bounds
# ---------------------------------------------------------------------------

def test_fitted_parameters_stay_within_bounds():
    result = fit_foo2023_parameters(_fast_config("ion_radius_scale", "pore_radius_scale"))
    for name, value in result.parameter_values.items():
        lower, upper = result.bounds[name]
        assert lower <= value <= upper


def test_bound_hitting_is_reported():
    # Restrict ion_radius_scale so the (smaller-is-better) optimum hits the lower
    # bound; the fit should report it in parameters_at_bounds.
    config = EstimationConfig(
        parameter_specs=(
            ParameterSpec(
                name="ion_radius_scale",
                initial_value=0.95,
                lower_bound=0.90,
                upper_bound=1.00,
            ),
        ),
        seed_grid_points=3,
        max_nfev=20,
    )
    result = fit_foo2023_parameters(config)
    assert "ion_radius_scale" in result.parameters_at_bounds
    assert any(
        "bounds" in w.lower() for w in result.identifiability_diagnostics["warnings"]
    )


# ---------------------------------------------------------------------------
# 6. identifiability diagnostics
# ---------------------------------------------------------------------------

def test_degrees_of_freedom_computed_correctly():
    one = fit_foo2023_parameters(_fast_config("pore_radius_scale"))
    two = fit_foo2023_parameters(_fast_config("ion_radius_scale", "pore_radius_scale"))
    assert one.identifiability_diagnostics["degrees_of_freedom"] == 8 - 1
    assert two.identifiability_diagnostics["degrees_of_freedom"] == 8 - 2


def test_condition_number_available_for_two_parameter_fit():
    result = fit_foo2023_parameters(_fast_config("ion_radius_scale", "pore_radius_scale"))
    cond = result.identifiability_diagnostics["jacobian_condition_number"]
    # Two-parameter fit with a converged Jacobian exposes a finite condition number.
    assert cond is not None
    assert cond >= 1.0


# ---------------------------------------------------------------------------
# 7. objective scans
# ---------------------------------------------------------------------------

def test_objective_scan_1d_rows_columns_and_determinism():
    values = [1.0, 1.25, 1.5]
    scan = objective_scan_1d("pore_radius_scale", values=values)
    assert list(scan["pore_radius_scale"]) == values
    for column in [
        "total_rmse",
        "R_Li_rmse",
        "R_Mg_rmse",
        "number_successful_solves",
        "number_failed_solves",
        "reproduces_negative_Li_rejection",
        "Mg_hard_cutoff_present",
    ]:
        assert column in scan.columns
    # Solve counts sum to four on every row.
    for _, row in scan.iterrows():
        assert row["number_successful_solves"] + row["number_failed_solves"] == 4
    # Deterministic.
    again = objective_scan_1d("pore_radius_scale", values=values)
    assert list(again["total_rmse"]) == list(scan["total_rmse"])


def test_objective_scan_2d_rows_columns():
    scan = objective_scan_2d(
        "ion_radius_scale",
        "pore_radius_scale",
        values_x=[0.7, 1.0],
        values_y=[1.0, 1.5],
    )
    assert len(scan) == 4  # full 2x2 grid
    assert "ion_radius_scale" in scan.columns
    assert "pore_radius_scale" in scan.columns
    assert "total_rmse" in scan.columns
    for _, row in scan.iterrows():
        assert row["number_successful_solves"] + row["number_failed_solves"] == 4


def test_objective_scan_rejects_bad_names():
    with pytest.raises(ValueError):
        objective_scan_1d("nope", values=[1.0])
    with pytest.raises(ValueError):
        objective_scan_2d("ion_radius_scale", "ion_radius_scale")


# ---------------------------------------------------------------------------
# 8. negative Li diagnostic
# ---------------------------------------------------------------------------

def test_negative_li_diagnostic_present_in_fit_and_scan():
    result = fit_foo2023_parameters(_fast_config("pore_radius_scale"))
    assert isinstance(result.reproduces_negative_Li_rejection, bool)
    assert isinstance(
        result.reproduces_negative_Li_rejection_in_successful_solve, bool
    )

    scan = objective_scan_1d("pore_radius_scale", values=[1.0, 1.5])
    assert "reproduces_negative_Li_rejection" in scan.columns

    summary = foo2023_parameter_estimation_summary(
        single_names=["pore_radius_scale"],
        pairs=[("ion_radius_scale", "pore_radius_scale")],
        max_nfev=20,
    )
    assert "any_fit_reproduces_negative_Li_rejection" in summary
    assert isinstance(summary["any_fit_reproduces_negative_Li_rejection"], bool)


# ---------------------------------------------------------------------------
# 9. no mutation
# ---------------------------------------------------------------------------

def test_case_and_base_variant_not_mutated():
    reference_case = foo2023_lmc_ph7_case()
    base_variant = ConstitutiveVariantConfig()

    config = EstimationConfig(
        parameter_specs=(parameter_spec("fixed_charge_scale"), parameter_spec("ion_radius_scale")),
        base_variant_config=base_variant,
        seed_grid_points=3,
        max_nfev=15,
    )
    fit_foo2023_parameters(config)
    objective_scan_1d("fixed_charge_scale", values=[0.5, 1.0])

    assert foo2023_lmc_ph7_case() == reference_case
    # Frozen base variant is unchanged.
    assert base_variant == ConstitutiveVariantConfig()


# ---------------------------------------------------------------------------
# 10. baseline protection
# ---------------------------------------------------------------------------

def test_baseline_solver_output_unchanged_by_estimation():
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
    fit_foo2023_parameters(_fast_config("pore_radius_scale"))
    after = baseline_rejections()

    assert before.keys() == after.keys()
    for ion in before:
        assert before[ion] == pytest.approx(after[ion])


# ---------------------------------------------------------------------------
# 11. documentation smoke test
# ---------------------------------------------------------------------------

def test_documentation_exists_with_conservative_language():
    doc = Path(__file__).resolve().parents[1] / "docs" / "parameter_estimation_identifiability.md"
    assert doc.exists()
    text = doc.read_text().lower()
    assert "diagnostic" in text
    assert "identifiability" in text
    assert "not validation success" in text or "not a validation" in text


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def test_scan_and_fit_plots_return_and_save(tmp_path):
    scan_1d = objective_scan_1d("pore_radius_scale", values=[1.0, 1.25, 1.5])
    scan_2d = objective_scan_2d(
        "ion_radius_scale", "pore_radius_scale", values_x=[0.7, 1.0], values_y=[1.0, 1.5]
    )
    result = fit_foo2023_parameters(_fast_config("pore_radius_scale"))

    figure, axes = plot_objective_scan_1d(scan_1d, "pore_radius_scale")
    assert isinstance(figure, Figure) and isinstance(axes, Axes)
    plt.close(figure)

    figure, axes = plot_objective_scan_2d(scan_2d, "ion_radius_scale", "pore_radius_scale")
    assert isinstance(figure, Figure) and isinstance(axes, Axes)
    plt.close(figure)

    path = tmp_path / "fit_rejection.png"
    figure, _ = plot_parameter_fit_rejection(result, output_path=path)
    assert path.exists() and path.stat().st_size > 0
    plt.close(figure)


def test_batch_runners_return_results():
    single = run_foo2023_single_parameter_fits(
        names=["pore_radius_scale"], max_nfev=15
    )
    pairs = run_foo2023_two_parameter_fits(
        pairs=[("ion_radius_scale", "pore_radius_scale")], max_nfev=15
    )
    assert len(single) == 1 and isinstance(single[0], EstimationResult)
    assert len(pairs) == 1 and isinstance(pairs[0], EstimationResult)
