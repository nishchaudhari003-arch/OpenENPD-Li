"""
Tests for the two-interface ENP-Donnan residual functions.

These tests exercise residual *construction* only. There is no nonlinear solve
here (Milestone 2), so residuals are evaluated for provided Donnan potentials,
a provided membrane potential drop, and a provided permeate composition. They
verify that each residual matches a direct manual calculation, that the vector
is deterministic and correctly sized, that diagnostics (norm, finiteness,
hard-cutoff) are surfaced, and that no input is mutated.
"""

import math

import pytest

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.electroneutrality import charge_balance
from openenpd.two_interface import (
    TwoInterfaceInputs,
    assemble_two_interface_state,
    two_interface_inputs_from_case,
)
from openenpd.two_interface_residuals import (
    TwoInterfaceResiduals,
    TwoInterfaceTransportInputs,
    compute_two_interface_residuals,
    convective_permeate_fluxes,
    enp_ion_fluxes,
    feed_side_electroneutrality_residual,
    flux_closure_residuals,
    permeate_electroneutrality_residual,
    permeate_side_electroneutrality_residual,
    residual_labels,
    residual_norm,
    residual_vector,
    two_interface_transport_inputs_from_case,
    zero_current_residual,
)


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _controlled_state():
    """A controlled, hard-cutoff-free two-interface state (all ions fit the pore)."""
    inputs = TwoInterfaceInputs(
        feed_bulk_concentrations_mol_L={"A+": 0.10, "B-": 0.10},
        permeate_concentrations_mol_L={"A+": 0.04, "B-": 0.04},
        charges={"A+": 1, "B-": -1},
        ion_radii_nm={"A+": 0.15, "B-": 0.15},
        pore_radius_nm=1.0,
        fixed_charge_mol_L=-0.02,
        feed_side_donnan_V=-0.010,
        permeate_side_donnan_V=-0.004,
        pore_dielectric_constant=45.0,
        temperature_K=298.15,
    )
    return assemble_two_interface_state(inputs)


def _controlled_transport():
    """Transport inputs matching the controlled state, with a nonzero water flux."""
    return TwoInterfaceTransportInputs(
        diffusivities_m2_s={"A+": 1.0e-9, "B-": 2.0e-9},
        diffusive_hindrance={"A+": 0.7225, "B-": 0.7225},
        convective_hindrance={"A+": 0.7225, "B-": 0.7225},
        thickness_m=1.0e-7,
        water_flux_m_s=1.0e-5,
        membrane_potential_drop_V=-0.005,
        temperature_K=298.15,
    )


def _placeholder_permeate(case, fraction=0.5):
    return {
        ion: fraction * concentration
        for ion, concentration in case["bulk_concentrations_mol_L"].items()
    }


def _foo_state_and_transport(flux_index=0):
    case = foo2023_lmc_ph7_case()
    permeate = _placeholder_permeate(case)
    state = assemble_two_interface_state(
        two_interface_inputs_from_case(
            case,
            permeate_concentrations_mol_L=permeate,
            feed_side_donnan_V=-0.02,
            permeate_side_donnan_V=-0.01,
        )
    )
    water_flux = case["experimental_data"]["Jw_um_s"][flux_index] * 1e-6
    transport = two_interface_transport_inputs_from_case(
        case,
        water_flux_m_s=water_flux,
        membrane_potential_drop_V=-0.005,
    )
    return state, transport


# ---------------------------------------------------------------------------
# 1. residual object creation
# ---------------------------------------------------------------------------

def test_residual_object_has_named_fields_vector_and_finite_norm():
    result = compute_two_interface_residuals(
        _controlled_state(), _controlled_transport()
    )

    assert isinstance(result, TwoInterfaceResiduals)
    # Named scalar fields exist.
    assert math.isfinite(result.feed_side_electroneutrality_mol_L)
    assert math.isfinite(result.permeate_side_electroneutrality_mol_L)
    assert math.isfinite(result.permeate_electroneutrality_mol_L)
    assert math.isfinite(result.zero_current_mol_m2_s)
    # Vector exists and norm is finite for this controlled case.
    assert len(result.residual_vector) > 0
    assert math.isfinite(result.l2_norm)
    assert result.is_finite is True
    # named_residuals mapping is populated and read-only.
    assert "feed_side_electroneutrality" in result.named_residuals
    with pytest.raises(TypeError):
        result.named_residuals["feed_side_electroneutrality"] = 0.0


def test_transport_inputs_reject_inconsistent_ions_and_bad_scalars():
    with pytest.raises(ValueError):
        TwoInterfaceTransportInputs(
            diffusivities_m2_s={"A+": 1e-9, "B-": 2e-9},
            diffusive_hindrance={"A+": 0.5},  # missing B-
            convective_hindrance={"A+": 0.5, "B-": 0.5},
            thickness_m=1e-7,
            water_flux_m_s=1e-5,
            membrane_potential_drop_V=0.0,
            temperature_K=298.15,
        )
    with pytest.raises(ValueError):
        TwoInterfaceTransportInputs(
            diffusivities_m2_s={"A+": 1e-9},
            diffusive_hindrance={"A+": 0.5},
            convective_hindrance={"A+": 0.5},
            thickness_m=0.0,  # nonpositive
            water_flux_m_s=1e-5,
            membrane_potential_drop_V=0.0,
            temperature_K=298.15,
        )


# ---------------------------------------------------------------------------
# 2-4. electroneutrality residuals match direct manual calculation
# ---------------------------------------------------------------------------

def test_feed_side_electroneutrality_matches_manual():
    state = _controlled_state()
    manual = (
        sum(
            state.charges[ion] * state.feed_membrane_concentrations_mol_L[ion]
            for ion in state.ions
        )
        + state.fixed_charge_mol_L
    )
    assert feed_side_electroneutrality_residual(state) == pytest.approx(manual)
    # Consistent with the value the Milestone-1 assembler already stored.
    assert feed_side_electroneutrality_residual(state) == pytest.approx(
        state.feed_side_charge_residual_mol_L
    )


def test_permeate_side_electroneutrality_matches_manual():
    state = _controlled_state()
    manual = (
        sum(
            state.charges[ion] * state.permeate_membrane_concentrations_mol_L[ion]
            for ion in state.ions
        )
        + state.fixed_charge_mol_L
    )
    assert permeate_side_electroneutrality_residual(state) == pytest.approx(manual)
    assert permeate_side_electroneutrality_residual(state) == pytest.approx(
        state.permeate_side_charge_residual_mol_L
    )


def test_permeate_electroneutrality_matches_manual():
    state = _controlled_state()
    manual = sum(
        state.charges[ion] * state.permeate_concentrations_mol_L[ion]
        for ion in state.ions
    )
    assert permeate_electroneutrality_residual(state) == pytest.approx(manual)
    # No fixed charge in the free solution -> equals charge_balance directly.
    assert permeate_electroneutrality_residual(state) == pytest.approx(
        charge_balance(state.permeate_concentrations_mol_L, state.charges)
    )


# ---------------------------------------------------------------------------
# 5. residual vector size is deterministic
# ---------------------------------------------------------------------------

def test_residual_vector_size_is_deterministic():
    state = _controlled_state()
    transport = _controlled_transport()

    expected_size = 4 + len(state.ions)  # 3 EN + zero-current + N flux closures
    vector = residual_vector(state, transport)

    assert len(vector) == expected_size
    assert len(residual_labels(state)) == expected_size
    # Deterministic: identical inputs give an identical vector.
    assert residual_vector(state, transport) == vector

    result = compute_two_interface_residuals(state, transport)
    assert len(result.residual_vector) == expected_size
    assert result.residual_vector == vector
    assert result.residual_labels == residual_labels(state)


def test_foo_residual_vector_size():
    state, transport = _foo_state_and_transport()
    # Foo 2023 LM-C has 3 ions -> 4 + 3 = 7 residual equations.
    assert len(residual_vector(state, transport)) == 4 + len(state.ions)
    assert len(residual_vector(state, transport)) == 7


# ---------------------------------------------------------------------------
# 6. zero-current residual consistent with the species fluxes
# ---------------------------------------------------------------------------

def test_zero_current_residual_consistent_with_fluxes():
    state = _controlled_state()
    transport = _controlled_transport()

    result = compute_two_interface_residuals(state, transport)
    fluxes = result.enp_fluxes_mol_m2_s

    charge_weighted = sum(state.charges[ion] * fluxes[ion] for ion in state.ions)
    assert result.zero_current_mol_m2_s == pytest.approx(charge_weighted)
    # Standalone function agrees.
    assert zero_current_residual(state, transport) == pytest.approx(charge_weighted)


def test_zero_current_sign_flips_with_membrane_potential():
    """A controlled asymmetric case gives opposite-signed zero current for +/- Dpsi_m."""
    state = _controlled_state()
    base = _controlled_transport()

    def transport_with(drop):
        return TwoInterfaceTransportInputs(
            diffusivities_m2_s=dict(base.diffusivities_m2_s),
            diffusive_hindrance=dict(base.diffusive_hindrance),
            convective_hindrance=dict(base.convective_hindrance),
            thickness_m=base.thickness_m,
            water_flux_m_s=0.0,  # isolate diffusion + migration
            membrane_potential_drop_V=drop,
            temperature_K=base.temperature_K,
        )

    positive = zero_current_residual(state, transport_with(0.02))
    negative = zero_current_residual(state, transport_with(-0.02))
    # Migration term is linear in the field, so the sign follows Dpsi_m here.
    assert math.copysign(1.0, positive) != math.copysign(1.0, negative)


# ---------------------------------------------------------------------------
# 7. flux residuals
# ---------------------------------------------------------------------------

def test_flux_residual_keys_match_ions_and_are_finite():
    state = _controlled_state()
    transport = _controlled_transport()

    residuals = flux_closure_residuals(state, transport)
    assert set(residuals.keys()) == set(state.ions)
    assert all(math.isfinite(value) for value in residuals.values())


def test_flux_residual_equals_enp_minus_convective():
    state = _controlled_state()
    transport = _controlled_transport()

    enp = enp_ion_fluxes(state, transport)
    convective = convective_permeate_fluxes(state, transport)
    residuals = flux_closure_residuals(state, transport)

    for ion in state.ions:
        assert residuals[ion] == pytest.approx(enp[ion] - convective[ion])


# ---------------------------------------------------------------------------
# 8. hard-cutoff species surfaced; Foo 2023 does not crash
# ---------------------------------------------------------------------------

def test_hard_cutoff_species_surfaced_in_diagnostics():
    state, transport = _foo_state_and_transport()
    result = compute_two_interface_residuals(state, transport)

    assert "Mg2+" in result.hard_cutoff_species
    # Active set excludes the hard-cutoff species.
    assert "Mg2+" not in result.active_ions
    assert "Li+" in result.active_ions
    # A diagnostic message names the cutoff species.
    assert any("Mg2+" in message for message in result.messages)
    # Construction still produced a finite residual for every ion.
    assert set(result.flux_residuals_mol_m2_s.keys()) == set(state.ions)
    assert result.is_finite is True


def test_hard_cutoff_flux_reduces_to_negative_convective():
    """Mg2+ has zero ENP flux, so its flux residual is exactly -C_p*J_v."""
    state, transport = _foo_state_and_transport()
    result = compute_two_interface_residuals(state, transport)

    assert result.enp_fluxes_mol_m2_s["Mg2+"] == pytest.approx(0.0)
    assert result.flux_residuals_mol_m2_s["Mg2+"] == pytest.approx(
        -result.convective_permeate_fluxes_mol_m2_s["Mg2+"]
    )


# ---------------------------------------------------------------------------
# 9. no mutation of case dictionary or input mappings
# ---------------------------------------------------------------------------

def test_case_dictionary_not_mutated_by_residual_construction():
    case = foo2023_lmc_ph7_case()
    reference = foo2023_lmc_ph7_case()

    permeate = _placeholder_permeate(case)
    state = assemble_two_interface_state(
        two_interface_inputs_from_case(case, permeate_concentrations_mol_L=permeate)
    )
    transport = two_interface_transport_inputs_from_case(
        case, water_flux_m_s=1e-5, membrane_potential_drop_V=-0.005
    )
    compute_two_interface_residuals(state, transport)

    assert case == reference


def test_transport_inputs_do_not_alias_source_dicts():
    diffusivities = {"A+": 1e-9, "B-": 2e-9}
    transport = TwoInterfaceTransportInputs(
        diffusivities_m2_s=diffusivities,
        diffusive_hindrance={"A+": 0.5, "B-": 0.5},
        convective_hindrance={"A+": 0.5, "B-": 0.5},
        thickness_m=1e-7,
        water_flux_m_s=1e-5,
        membrane_potential_drop_V=0.0,
        temperature_K=298.15,
    )
    diffusivities["A+"] = 999.0
    assert transport.diffusivities_m2_s["A+"] == 1e-9
    with pytest.raises(TypeError):
        transport.diffusivities_m2_s["A+"] = 1.0


# ---------------------------------------------------------------------------
# standalone helpers agree with the assembled result
# ---------------------------------------------------------------------------

def test_standalone_vector_and_norm_match_result():
    state = _controlled_state()
    transport = _controlled_transport()
    result = compute_two_interface_residuals(state, transport)

    assert residual_vector(state, transport) == result.residual_vector
    assert residual_norm(state, transport) == pytest.approx(result.l2_norm)
    assert result.max_abs_residual == pytest.approx(
        max(abs(value) for value in result.residual_vector)
    )


# ---------------------------------------------------------------------------
# 10. integration smoke test: Foo 2023 for one experimental water flux
# ---------------------------------------------------------------------------

def test_foo2023_residuals_for_one_experimental_flux():
    state, transport = _foo_state_and_transport(flux_index=0)
    result = compute_two_interface_residuals(state, transport)

    expected_ions = {"Li+", "Mg2+", "Cl-"}
    assert set(state.ions) == expected_ions
    assert set(result.flux_residuals_mol_m2_s.keys()) == expected_ions

    # Everything finite; diagnostics populated.
    assert result.is_finite is True
    assert math.isfinite(result.l2_norm)
    assert math.isfinite(result.max_abs_residual)
    assert "Li+" in result.flux_residuals_mol_m2_s
    assert "Mg2+" in result.flux_residuals_mol_m2_s
    assert len(result.messages) >= 1
