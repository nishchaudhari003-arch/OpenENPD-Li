"""
Tests for configurable constitutive steric/hindrance model variants.

These verify the variant configuration and factor functions, that the default
reproduces the hard baseline, that softened variants relax the Mg2+ hard cutoff,
and that the diagnostic workflow (comparison rows, metrics, scans, plots) is
honest about solve success and the negative-Li question. They do NOT assert the
corrected model is physically true or validated.
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
from openenpd.two_interface import (
    assemble_two_interface_state,
    two_interface_inputs_from_case,
)
from openenpd.two_interface_solver import solve_foo2023_lmc_ph7_all_fluxes
from openenpd.plots import (
    plot_constitutive_variant_li_mg_rmse,
    plot_constitutive_variant_mg_hard_cutoff,
    plot_constitutive_variant_total_rmse,
)
from openenpd.constitutive_variants import (
    ConstitutiveVariantConfig,
    build_two_interface_state_variant,
    default_variants,
    effective_size_ratio,
    hindrance_factor_variant,
    run_foo2023_lmc_ph7_variant_comparison,
    scan_hindrance_floor,
    scan_ion_radius_scale,
    scan_pore_radius_scale,
    scan_steric_floor,
    steric_partition_factor_variant,
    variant_foo2023_lmc_ph7_comparison_rows,
    variant_foo2023_lmc_ph7_metrics,
    variant_ions_and_cutoff,
)

# Mg2+ size ratio for Foo 2023: 0.428 / 0.416 ~ 1.029 (hard cutoff).
MG_LAMBDA = 0.428 / 0.416


# ---------------------------------------------------------------------------
# 1. config defaults and validation
# ---------------------------------------------------------------------------

def test_default_config_reproduces_hard_baseline_settings():
    config = ConstitutiveVariantConfig()
    assert config.steric_model == "hard"
    assert config.hindrance_model == "hard"
    assert config.ion_radius_scale == 1.0
    assert config.pore_radius_scale == 1.0
    assert config.steric_floor == 0.0
    assert config.hindrance_floor == 0.0
    assert config.pore_radius_nm_override is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steric_model": "quadratic"},
        {"hindrance_model": "bogus"},
        {"ion_radius_scale": 0.0},
        {"ion_radius_scale": -1.0},
        {"pore_radius_scale": 0.0},
        {"pore_radius_nm_override": 0.0},
        {"steric_floor": -0.1},
        {"steric_floor": 1.5},
        {"hindrance_floor": -0.001},
        {"soft_cutoff_width": 0.0},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        ConstitutiveVariantConfig(**kwargs)


# ---------------------------------------------------------------------------
# 2. steric floor behavior
# ---------------------------------------------------------------------------

def test_steric_hard_matches_existing_form():
    hard = ConstitutiveVariantConfig()
    assert steric_partition_factor_variant(1.03, hard) == 0.0
    assert steric_partition_factor_variant(0.5, hard) == pytest.approx(0.25)


def test_steric_floor_never_below_floor_and_softens_mg():
    floor = ConstitutiveVariantConfig(steric_model="floor", steric_floor=0.02)
    # Mg cutoff softened: hard would be 0, floor gives 0.02.
    assert steric_partition_factor_variant(MG_LAMBDA, floor) == pytest.approx(0.02)
    # Never below the floor across a range of lambda.
    for lam in [0.1, 0.5, 0.95, 1.0, 1.5, 3.0]:
        assert steric_partition_factor_variant(lam, floor) >= 0.02


def test_steric_soft_positive_and_reduces_to_hard():
    soft = ConstitutiveVariantConfig(steric_model="soft", soft_cutoff_width=0.1)
    # Smooth and strictly positive beyond the cutoff.
    assert steric_partition_factor_variant(MG_LAMBDA, soft) > 0.0
    # As width -> 0 it recovers the hard value below cutoff.
    narrow = ConstitutiveVariantConfig(steric_model="soft", soft_cutoff_width=1e-4)
    assert steric_partition_factor_variant(0.5, narrow) == pytest.approx(0.25, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. hindrance floor behavior
# ---------------------------------------------------------------------------

def test_hindrance_hard_matches_existing_form():
    hard = ConstitutiveVariantConfig()
    assert hindrance_factor_variant(1.03, hard) == 0.0
    assert hindrance_factor_variant(0.5, hard) == pytest.approx(0.25)


def test_hindrance_floor_never_below_floor():
    floor = ConstitutiveVariantConfig(hindrance_model="floor", hindrance_floor=0.05)
    for lam in [0.1, 0.5, 1.0, 1.5, 3.0]:
        assert hindrance_factor_variant(lam, floor) >= 0.05


# ---------------------------------------------------------------------------
# 4. radius scaling behavior + no mutation
# ---------------------------------------------------------------------------

def test_ion_radius_scale_changes_lambda_expected_direction():
    base = ConstitutiveVariantConfig()
    smaller = ConstitutiveVariantConfig(ion_radius_scale=0.5)
    lam_base = effective_size_ratio(0.428, 0.416, base)
    lam_small = effective_size_ratio(0.428, 0.416, smaller)
    assert lam_small < lam_base
    assert lam_small == pytest.approx(0.5 * lam_base)


def test_pore_radius_scale_changes_lambda_expected_direction():
    base = ConstitutiveVariantConfig()
    wider = ConstitutiveVariantConfig(pore_radius_scale=2.0)
    lam_base = effective_size_ratio(0.428, 0.416, base)
    lam_wide = effective_size_ratio(0.428, 0.416, wider)
    assert lam_wide < lam_base
    assert lam_wide == pytest.approx(lam_base / 2.0)


def test_pore_radius_override_wins_over_scale():
    config = ConstitutiveVariantConfig(pore_radius_scale=2.0, pore_radius_nm_override=0.5)
    assert effective_size_ratio(0.5, 0.416, config) == pytest.approx(1.0)


def test_variant_state_builder_does_not_mutate_case():
    case = foo2023_lmc_ph7_case()
    reference = foo2023_lmc_ph7_case()
    config = ConstitutiveVariantConfig(ion_radius_scale=0.8)

    build_two_interface_state_variant(
        case,
        permeate_concentrations_mol_L=dict(case["bulk_concentrations_mol_L"]),
        feed_side_donnan_V=-0.01,
        permeate_side_donnan_V=-0.005,
        config=config,
    )
    assert case == reference


# ---------------------------------------------------------------------------
# 5. integration with Foo 2023
# ---------------------------------------------------------------------------

def test_hard_variant_state_matches_baseline_assembler():
    case = foo2023_lmc_ph7_case()
    permeate = {k: 0.5 * v for k, v in case["bulk_concentrations_mol_L"].items()}
    hard = ConstitutiveVariantConfig()

    variant_state = build_two_interface_state_variant(
        case, permeate, -0.02, -0.01, hard
    )
    baseline_state = assemble_two_interface_state(
        two_interface_inputs_from_case(
            case,
            permeate_concentrations_mol_L=permeate,
            feed_side_donnan_V=-0.02,
            permeate_side_donnan_V=-0.01,
        )
    )

    assert variant_state.hard_cutoff_species == baseline_state.hard_cutoff_species
    for ion in baseline_state.ions:
        assert variant_state.feed_membrane_concentrations_mol_L[ion] == pytest.approx(
            baseline_state.feed_membrane_concentrations_mol_L[ion]
        )
        assert variant_state.permeate_membrane_concentrations_mol_L[ion] == pytest.approx(
            baseline_state.permeate_membrane_concentrations_mol_L[ion]
        )


def test_hard_variant_solve_matches_baseline_solve():
    baseline = solve_foo2023_lmc_ph7_all_fluxes()
    hard = solve_foo2023_lmc_ph7_all_fluxes(variant=ConstitutiveVariantConfig())
    for base_result, hard_result in zip(baseline, hard):
        assert base_result.success == hard_result.success
        for ion in ("Li+", "Mg2+", "Cl-"):
            assert base_result.predicted_rejections[ion] == pytest.approx(
                hard_result.predicted_rejections[ion]
            )


def test_hard_baseline_variant_keeps_mg_hard_cutoff():
    case = foo2023_lmc_ph7_case()
    _, cutoff = variant_ions_and_cutoff(case, ConstitutiveVariantConfig())
    assert "Mg2+" in cutoff


def test_at_least_one_softened_variant_removes_mg_hard_cutoff():
    case = foo2023_lmc_ph7_case()
    # Steric floor removes the steric hard cutoff for Mg.
    _, floor_cutoff = variant_ions_and_cutoff(
        case, ConstitutiveVariantConfig(steric_model="floor", steric_floor=0.01)
    )
    assert "Mg2+" not in floor_cutoff
    # Ion-radius scaling also removes it (geometry).
    _, scaled_cutoff = variant_ions_and_cutoff(
        case, ConstitutiveVariantConfig(ion_radius_scale=0.85)
    )
    assert "Mg2+" not in scaled_cutoff


def test_each_variant_returns_four_rows():
    for config in default_variants():
        rows = variant_foo2023_lmc_ph7_comparison_rows(config)
        assert len(rows) == 4
        for row in rows:
            assert row["variant_label"] == config.label
            assert "predicted_R_Li" in row
            assert "predicted_R_Mg" in row
            assert "solve_success" in row
            assert "hard_cutoff_species" in row


# ---------------------------------------------------------------------------
# 6. metrics
# ---------------------------------------------------------------------------

def test_metrics_returned_per_variant_with_expected_fields():
    for config in default_variants():
        metrics = variant_foo2023_lmc_ph7_metrics(config)
        assert metrics["variant_label"] == config.label
        # RMSE finite or explicitly None.
        for key in ("R_Li_rmse", "R_Mg_rmse", "total_rmse"):
            assert metrics[key] is None or math.isfinite(metrics[key])
        # Counts sum to four.
        assert (
            metrics["number_successful_solves"]
            + metrics["number_failed_solves"]
            == 4
        )
        # Best-effort flag is a bool and True exactly when not all solves succeed.
        assert isinstance(metrics["rmse_is_best_effort_diagnostic"], bool)
        assert metrics["rmse_is_best_effort_diagnostic"] == (
            metrics["number_successful_solves"] < 4
        )


def test_run_variant_comparison_covers_all_default_variants():
    rows = run_foo2023_lmc_ph7_variant_comparison()
    labels = {row["variant_label"] for row in rows}
    expected = {config.label for config in default_variants()}
    assert labels == expected
    assert len(rows) == 4 * len(default_variants())


# ---------------------------------------------------------------------------
# 7. negative Li diagnostic
# ---------------------------------------------------------------------------

def test_negative_li_diagnostic_present_and_boolean():
    for config in default_variants():
        metrics = variant_foo2023_lmc_ph7_metrics(config)
        assert "reproduces_negative_Li_rejection" in metrics
        assert isinstance(metrics["reproduces_negative_Li_rejection"], bool)
        assert isinstance(
            metrics["reproduces_negative_Li_rejection_in_successful_solve"], bool
        )


# ---------------------------------------------------------------------------
# sensitivity scans (short value lists to keep tests fast)
# ---------------------------------------------------------------------------

def test_scans_return_one_entry_per_value():
    ion_scan = scan_ion_radius_scale(values=[0.7, 1.0])
    pore_scan = scan_pore_radius_scale(values=[1.0, 1.5])
    steric_scan = scan_steric_floor(values=[0.0, 0.05])
    hindrance_scan = scan_hindrance_floor(values=[0.0, 0.05])

    for scan, n in [(ion_scan, 2), (pore_scan, 2), (steric_scan, 2), (hindrance_scan, 2)]:
        assert len(scan) == n
        for entry in scan:
            assert (
                entry["number_successful_solves"] + entry["number_failed_solves"] == 4
            )
            assert "Mg_hard_cutoff_present" in entry
            assert "reproduces_negative_Li_rejection" in entry


def test_ion_radius_scan_relaxes_mg_cutoff_at_small_scale():
    scan = scan_ion_radius_scale(values=[0.5, 1.0])
    by_scale = {entry["ion_radius_scale"]: entry for entry in scan}
    assert by_scale[1.0]["Mg_hard_cutoff_present"] is True
    assert by_scale[0.5]["Mg_hard_cutoff_present"] is False


# ---------------------------------------------------------------------------
# 8. plot tests
# ---------------------------------------------------------------------------

def _two_variant_metrics():
    configs = [
        ConstitutiveVariantConfig(label="hard_baseline"),
        ConstitutiveVariantConfig(pore_radius_scale=1.25, label="pore_scale"),
    ]
    return [variant_foo2023_lmc_ph7_metrics(c) for c in configs]


@pytest.mark.parametrize(
    "plot_function",
    [
        plot_constitutive_variant_total_rmse,
        plot_constitutive_variant_li_mg_rmse,
        plot_constitutive_variant_mg_hard_cutoff,
    ],
)
def test_variant_plots_return_figure_and_axes(plot_function):
    metrics = _two_variant_metrics()
    figure, axes = plot_function(metrics)
    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    plt.close(figure)


@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (plot_constitutive_variant_total_rmse, "variant_total_rmse.png"),
        (plot_constitutive_variant_li_mg_rmse, "variant_li_mg_rmse.png"),
        (plot_constitutive_variant_mg_hard_cutoff, "variant_mg_cutoff.png"),
    ],
)
def test_variant_plots_save_nonempty_file(plot_function, filename, tmp_path):
    metrics = _two_variant_metrics()
    output_path = tmp_path / filename
    figure, _ = plot_function(metrics, output_path=output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(figure)


# ---------------------------------------------------------------------------
# 9. baseline protection
# ---------------------------------------------------------------------------

def test_baseline_solver_output_unchanged_by_variant_workflow():
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
    run_foo2023_lmc_ph7_variant_comparison(
        configs=[ConstitutiveVariantConfig(ion_radius_scale=0.8)]
    )
    after = baseline_rejections()

    assert before.keys() == after.keys()
    for ion in before:
        assert before[ion] == pytest.approx(after[ion])


# ---------------------------------------------------------------------------
# 10. no mutation of original case data after scans
# ---------------------------------------------------------------------------

def test_original_case_unchanged_after_scans():
    reference = foo2023_lmc_ph7_case()
    scan_ion_radius_scale(values=[0.7, 1.0])
    scan_pore_radius_scale(values=[1.0, 1.5])
    scan_steric_floor(values=[0.0, 0.05])
    scan_hindrance_floor(values=[0.0, 0.05])
    assert foo2023_lmc_ph7_case() == reference
