"""
Tests for the minimal nonlinear two-interface ENP-Donnan solver.

These tests verify the solver *infrastructure*: options validation, unknown
packing/unpacking, residual scaling, convergence on a controlled symmetric
case, structured (non-crashing) failures, the Foo 2023 wrappers, protection of
the existing baseline solver, and input immutability.

They deliberately do NOT assert that the Foo 2023 case converges or reproduces
any particular rejection sign -- the milestone treats that as an open physics
question.
"""

import math

import pytest

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejection_for_flux
from openenpd.two_interface import (
    assemble_two_interface_state,
    two_interface_inputs_from_case,
)
from openenpd.two_interface_residuals import (
    TwoInterfaceTransportInputs,
    feed_side_electroneutrality_residual,
    zero_current_residual,
)
from openenpd.hindrance import hindrance_factors
from openenpd.transport import active_layer_thickness_m
from openenpd.two_interface_solver import (
    ResidualScales,
    TwoInterfaceSolveResult,
    TwoInterfaceSolverOptions,
    compute_residual_scales,
    pack_unknowns,
    solve_foo2023_lmc_ph7_all_fluxes,
    solve_foo2023_lmc_ph7_flux,
    solve_two_interface_for_flux,
    unknown_labels,
    unpack_unknowns,
)


# ---------------------------------------------------------------------------
# controlled test fixtures (synthetic; NOT literature validation cases)
# ---------------------------------------------------------------------------

def _controlled_symmetric_case():
    """
    A controlled symmetric monovalent salt: zero fixed charge, equal
    diffusivities, a large pore (no hard cutoff), and pore dielectric equal to
    bulk (so the dielectric factor is 1). By symmetry this system has a
    well-defined solution with small residuals.
    """
    return {
        "case_id": "controlled_symmetric",
        "temperature_K": 298.15,
        "bulk_concentrations_mol_L": {"A+": 0.10, "B-": 0.10},
        "charges": {"A+": 1, "B-": -1},
        "ion_radii_nm": {"A+": 0.10, "B-": 0.10},
        "diffusivities_m2_s": {"A+": 1.0e-9, "B-": 1.0e-9},
        "membrane_parameters": {
            "pore_radius_nm": 1.0,
            "pore_dielectric_constant": 80.1,
            "fixed_charge_mol_L": 0.0,
            "active_layer_thickness_nm": 100.0,
        },
    }


def _all_hard_cutoff_case():
    """A case where every ion radius exceeds the pore radius (all excluded)."""
    return {
        "case_id": "controlled_all_cutoff",
        "temperature_K": 298.15,
        "bulk_concentrations_mol_L": {"A+": 0.10, "B-": 0.10},
        "charges": {"A+": 1, "B-": -1},
        "ion_radii_nm": {"A+": 0.60, "B-": 0.60},
        "diffusivities_m2_s": {"A+": 1.0e-9, "B-": 1.0e-9},
        "membrane_parameters": {
            "pore_radius_nm": 0.40,
            "pore_dielectric_constant": 80.1,
            "fixed_charge_mol_L": 0.0,
            "active_layer_thickness_nm": 100.0,
        },
    }


# ---------------------------------------------------------------------------
# 1. options creation
# ---------------------------------------------------------------------------

def test_default_options_are_valid():
    options = TwoInterfaceSolverOptions()
    assert options.max_nfev >= 1
    assert options.tolerance > 0.0
    assert options.potential_bounds_V > 0.0
    assert options.concentration_lower_bound_mol_L >= 0.0
    assert options.concentration_upper_factor > 1.0
    assert options.flux_scale_floor_mol_m2_s > 0.0
    assert options.include_permeate_electroneutrality_in_solve is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": -1.0},
        {"tolerance": 0.0},
        {"potential_bounds_V": 0.0},
        {"potential_bounds_V": -0.1},
        {"concentration_lower_bound_mol_L": -1e-6},
        {"concentration_upper_factor": 1.0},
        {"concentration_upper_factor": 0.5},
        {"max_nfev": 0},
        {"flux_scale_floor_mol_m2_s": 0.0},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(ValueError):
        TwoInterfaceSolverOptions(**kwargs)


# ---------------------------------------------------------------------------
# 2. unknown vector packing / unpacking
# ---------------------------------------------------------------------------

def test_pack_unpack_roundtrip():
    active = ("A+", "B-")
    permeate = {"A+": 0.031, "B-": 0.029}
    x = pack_unknowns(permeate, -0.012, -0.004, 0.006, active)

    assert len(x) == len(active) + 3
    assert unknown_labels(active) == (
        "C_p[A+]",
        "C_p[B-]",
        "feed_donnan_V",
        "permeate_donnan_V",
        "membrane_potential_drop_V",
    )

    out_permeate, feed_donnan, perm_donnan, dpsi_m = unpack_unknowns(x, active)
    assert out_permeate == permeate
    assert feed_donnan == pytest.approx(-0.012)
    assert perm_donnan == pytest.approx(-0.004)
    assert dpsi_m == pytest.approx(0.006)


def test_unpack_rejects_wrong_length():
    with pytest.raises(ValueError):
        unpack_unknowns([0.1, 0.1], ("A+", "B-"))  # needs 5 entries


def test_solved_concentrations_stay_nonnegative():
    result = solve_two_interface_for_flux(
        _controlled_symmetric_case(), water_flux_m_s=1.0e-5
    )
    for value in result.permeate_concentrations_mol_L.values():
        assert value >= 0.0


# ---------------------------------------------------------------------------
# 3. scaled residual vector + raw/scaled exposure
# ---------------------------------------------------------------------------

def test_residual_scales_are_positive_and_typed():
    options = TwoInterfaceSolverOptions()
    scales = compute_residual_scales({"A+": 0.1, "B-": 0.1}, 1.0e-5, options)
    assert isinstance(scales, ResidualScales)
    assert scales.concentration_scale_mol_L == pytest.approx(0.2)
    assert scales.flux_scale_mol_m2_s > 0.0


def test_flux_scale_floor_applied_for_zero_flux():
    options = TwoInterfaceSolverOptions(flux_scale_floor_mol_m2_s=1e-10)
    scales = compute_residual_scales({"A+": 0.1, "B-": 0.1}, 0.0, options)
    assert scales.flux_scale_mol_m2_s == pytest.approx(1e-10)


def test_scaled_and_raw_residuals_both_exposed_and_finite():
    result = solve_two_interface_for_flux(
        _controlled_symmetric_case(), water_flux_m_s=1.0e-5
    )
    # Scaled solve vector is finite.
    assert len(result.scaled_residual_vector) == len(result.scaled_residual_labels)
    assert all(math.isfinite(v) for v in result.scaled_residual_vector)
    # Raw residuals are exposed (full named set including the diagnostic).
    assert "feed_side_electroneutrality" in result.raw_residuals
    assert "permeate_electroneutrality" in result.raw_residuals
    assert all(math.isfinite(v) for v in result.raw_residuals.values())
    assert math.isfinite(result.residual_norm_scaled)
    assert math.isfinite(result.residual_norm_raw)


def test_scaling_does_not_mutate_raw_residuals():
    """The reported raw residuals equal an independent unscaled recomputation."""
    case = _controlled_symmetric_case()
    result = solve_two_interface_for_flux(case, water_flux_m_s=1.0e-5)

    # Rebuild the state/transport at the reported solution and recompute the
    # RAW feed-side electroneutrality and zero-current residuals directly.
    permeate = dict(result.permeate_concentrations_mol_L)
    state = assemble_two_interface_state(
        two_interface_inputs_from_case(
            case,
            permeate_concentrations_mol_L=permeate,
            feed_side_donnan_V=result.feed_donnan_potential_V,
            permeate_side_donnan_V=result.permeate_donnan_potential_V,
        )
    )
    params = case["membrane_parameters"]
    hindrance = hindrance_factors(
        ion_radii_nm=case["ion_radii_nm"], pore_radius_nm=params["pore_radius_nm"]
    )
    transport = TwoInterfaceTransportInputs(
        diffusivities_m2_s=case["diffusivities_m2_s"],
        diffusive_hindrance=hindrance["diffusive"],
        convective_hindrance=hindrance["convective"],
        thickness_m=active_layer_thickness_m(case),
        water_flux_m_s=1.0e-5,
        membrane_potential_drop_V=result.membrane_potential_drop_V,
        temperature_K=case["temperature_K"],
    )

    assert result.raw_residuals["feed_side_electroneutrality"] == pytest.approx(
        feed_side_electroneutrality_residual(state)
    )
    assert result.raw_residuals["zero_current"] == pytest.approx(
        zero_current_residual(state, transport)
    )


# ---------------------------------------------------------------------------
# 4. controlled symmetric case converges
# ---------------------------------------------------------------------------

def test_controlled_symmetric_case_converges():
    result = solve_two_interface_for_flux(
        _controlled_symmetric_case(), water_flux_m_s=1.0e-5
    )
    assert isinstance(result, TwoInterfaceSolveResult)
    assert result.success is True
    assert result.residual_max_abs_scaled <= TwoInterfaceSolverOptions().tolerance
    assert result.hard_cutoff_species == ()
    # By symmetry the anion and cation permeate concentrations match.
    assert result.permeate_concentrations_mol_L["A+"] == pytest.approx(
        result.permeate_concentrations_mol_L["B-"], rel=1e-6
    )


def test_controlled_case_is_deterministic():
    a = solve_two_interface_for_flux(_controlled_symmetric_case(), 1.0e-5)
    b = solve_two_interface_for_flux(_controlled_symmetric_case(), 1.0e-5)
    assert a.permeate_concentrations_mol_L["A+"] == pytest.approx(
        b.permeate_concentrations_mol_L["A+"]
    )
    assert a.residual_max_abs_scaled == pytest.approx(b.residual_max_abs_scaled)


# ---------------------------------------------------------------------------
# 5. structured failure
# ---------------------------------------------------------------------------

def test_all_hard_cutoff_returns_structured_failure():
    result = solve_two_interface_for_flux(
        _all_hard_cutoff_case(), water_flux_m_s=1.0e-5
    )
    assert isinstance(result, TwoInterfaceSolveResult)
    assert result.success is False
    assert "hard-cutoff" in result.message.lower()
    assert set(result.hard_cutoff_species) == {"A+", "B-"}
    assert result.active_ions == ()
    # Excluded ions are reported at zero permeate concentration.
    assert result.permeate_concentrations_mol_L["A+"] == 0.0
    # No optimizer was run.
    assert result.optimizer_nfev is None


def test_under_iterated_solve_reports_non_convergence_without_crashing():
    # Too few evaluations for the (already hard) Foo system to converge.
    options = TwoInterfaceSolverOptions(max_nfev=1, tolerance=1e-12)
    result = solve_foo2023_lmc_ph7_flux(8.05e-6, options=options)
    assert isinstance(result, TwoInterfaceSolveResult)
    assert result.success is False
    assert "not converged" in result.message.lower()


# ---------------------------------------------------------------------------
# 6. Foo 2023 one-flux smoke test
# ---------------------------------------------------------------------------

def test_foo2023_one_flux_runs_and_is_structured():
    result = solve_foo2023_lmc_ph7_flux(8.05e-6)

    assert isinstance(result, TwoInterfaceSolveResult)
    assert isinstance(result.success, bool)
    assert isinstance(result.message, str) and result.message
    # Required diagnostics are present.
    assert set(result.predicted_rejections.keys()) == {"Li+", "Mg2+", "Cl-"}
    assert math.isfinite(result.residual_norm_scaled)
    assert math.isfinite(result.residual_max_abs_scaled)
    assert math.isfinite(result.residual_norm_raw)
    # Mg2+ is the hard-cutoff species for this case.
    assert "Mg2+" in result.hard_cutoff_species
    assert result.permeate_concentrations_mol_L["Mg2+"] == 0.0


# ---------------------------------------------------------------------------
# 7. Foo 2023 all-flux wrapper
# ---------------------------------------------------------------------------

def test_foo2023_all_fluxes_returns_four_structured_results():
    results = solve_foo2023_lmc_ph7_all_fluxes()
    assert len(results) == 4

    case = foo2023_lmc_ph7_case()
    expected_fluxes = [v * 1e-6 for v in case["experimental_data"]["Jw_um_s"]]
    for result, expected_flux in zip(results, expected_fluxes):
        assert isinstance(result, TwoInterfaceSolveResult)
        assert result.water_flux_m_s == pytest.approx(expected_flux)
        # Li and Mg rejection keys are present.
        assert "Li+" in result.predicted_rejections
        assert "Mg2+" in result.predicted_rejections
        # Failed solves stay explicit rather than being dropped.
        assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# 8. baseline protection
# ---------------------------------------------------------------------------

def test_baseline_solver_output_unchanged_by_two_interface_solve():
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
    solve_two_interface_for_flux(case, flux)  # run the new solver
    after = baseline_rejections()

    assert before.keys() == after.keys()
    for ion in before:
        assert before[ion] == pytest.approx(after[ion])
    # Documented baseline behaviour: strongly positive Li rejection.
    assert before["Li+"] > 0.0


# ---------------------------------------------------------------------------
# 9. no mutation of the case dictionary
# ---------------------------------------------------------------------------

def test_case_dictionary_not_mutated_by_solve():
    case = foo2023_lmc_ph7_case()
    reference = foo2023_lmc_ph7_case()

    solve_two_interface_for_flux(case, 8.05e-6)
    assert case == reference

    solve_foo2023_lmc_ph7_all_fluxes()
    assert foo2023_lmc_ph7_case() == reference
