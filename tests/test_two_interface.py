"""
Tests for the two-interface ENP-Donnan state assembler.

These tests exercise the immutable, optimizer-free assembler in
``openenpd.two_interface``. They deliberately do NOT test any nonlinear solve
(there is none yet): the assembler only evaluates interface quantities in
closed form for provided Donnan potentials and permeate concentrations, and
reports the interface electroneutrality residuals explicitly.
"""

import math

import pytest

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.partitioning import total_partition_factor
from openenpd.two_interface import (
    TwoInterfaceInputs,
    TwoInterfaceState,
    assemble_two_interface_state,
    two_interface_inputs_from_case,
)


def _controlled_inputs():
    """
    A controlled, hard-cutoff-free synthetic case (all ions fit the pore).

    A large pore radius keeps every size ratio well below 1, so steric and
    hindrance factors stay positive and the assembler behaviour is easy to
    reason about.
    """
    return TwoInterfaceInputs(
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


def _placeholder_permeate(case, fraction=0.5):
    """Reasonable placeholder permeate composition: a fraction of the feed."""
    return {
        ion: fraction * concentration
        for ion, concentration in case["bulk_concentrations_mol_L"].items()
    }


# ---------------------------------------------------------------------------
# object / data-structure creation
# ---------------------------------------------------------------------------

def test_state_creation_returns_expected_type_and_ions():
    state = assemble_two_interface_state(_controlled_inputs())

    assert isinstance(state, TwoInterfaceState)
    assert state.ions == ("A+", "B-")
    assert set(state.feed_side_partition_factors) == {"A+", "B-"}
    assert set(state.permeate_side_partition_factors) == {"A+", "B-"}


def test_inputs_reject_inconsistent_ion_sets():
    with pytest.raises(ValueError):
        TwoInterfaceInputs(
            feed_bulk_concentrations_mol_L={"A+": 0.1, "B-": 0.1},
            permeate_concentrations_mol_L={"A+": 0.05},  # missing B-
            charges={"A+": 1, "B-": -1},
            ion_radii_nm={"A+": 0.15, "B-": 0.15},
            pore_radius_nm=1.0,
            fixed_charge_mol_L=-0.02,
            feed_side_donnan_V=0.0,
            permeate_side_donnan_V=0.0,
            pore_dielectric_constant=45.0,
            temperature_K=298.15,
        )


def test_inputs_reject_nonpositive_temperature():
    with pytest.raises(ValueError):
        TwoInterfaceInputs(
            feed_bulk_concentrations_mol_L={"A+": 0.1},
            permeate_concentrations_mol_L={"A+": 0.05},
            charges={"A+": 1},
            ion_radii_nm={"A+": 0.15},
            pore_radius_nm=1.0,
            fixed_charge_mol_L=0.0,
            feed_side_donnan_V=0.0,
            permeate_side_donnan_V=0.0,
            pore_dielectric_constant=45.0,
            temperature_K=0.0,
        )


# ---------------------------------------------------------------------------
# feed-side / permeate-side interface concentration consistency
# ---------------------------------------------------------------------------

def test_feed_side_interface_concentrations_consistent():
    inputs = _controlled_inputs()
    state = assemble_two_interface_state(inputs)

    for ion in state.ions:
        expected_factor = total_partition_factor(
            charge=inputs.charges[ion],
            ion_radius_nm=inputs.ion_radii_nm[ion],
            pore_radius_nm=inputs.pore_radius_nm,
            delta_psi_V=inputs.feed_side_donnan_V,
            pore_dielectric_constant=inputs.pore_dielectric_constant,
            temperature_K=inputs.temperature_K,
            bulk_dielectric_constant=inputs.bulk_dielectric_constant,
        )

        # Assembler's total factor matches the baseline partitioning module.
        assert state.feed_side_partition_factors[ion] == pytest.approx(
            expected_factor
        )

        # Membrane concentration is bulk feed times that factor.
        expected_conc = (
            inputs.feed_bulk_concentrations_mol_L[ion] * expected_factor
        )
        assert state.feed_membrane_concentrations_mol_L[ion] == pytest.approx(
            expected_conc
        )


def test_permeate_side_interface_concentrations_consistent():
    inputs = _controlled_inputs()
    state = assemble_two_interface_state(inputs)

    for ion in state.ions:
        expected_factor = total_partition_factor(
            charge=inputs.charges[ion],
            ion_radius_nm=inputs.ion_radii_nm[ion],
            pore_radius_nm=inputs.pore_radius_nm,
            delta_psi_V=inputs.permeate_side_donnan_V,
            pore_dielectric_constant=inputs.pore_dielectric_constant,
            temperature_K=inputs.temperature_K,
            bulk_dielectric_constant=inputs.bulk_dielectric_constant,
        )

        assert state.permeate_side_partition_factors[ion] == pytest.approx(
            expected_factor
        )

        # Membrane concentration uses the PROVIDED permeate composition.
        expected_conc = (
            inputs.permeate_concentrations_mol_L[ion] * expected_factor
        )
        assert state.permeate_membrane_concentrations_mol_L[ion] == pytest.approx(
            expected_conc
        )


def test_factor_decomposition_multiplies_to_total():
    state = assemble_two_interface_state(_controlled_inputs())

    for ion in state.ions:
        feed_product = (
            state.feed_side_donnan_factors[ion]
            * state.steric_factors[ion]
            * state.dielectric_factors[ion]
        )
        assert state.feed_side_partition_factors[ion] == pytest.approx(feed_product)

        permeate_product = (
            state.permeate_side_donnan_factors[ion]
            * state.steric_factors[ion]
            * state.dielectric_factors[ion]
        )
        assert state.permeate_side_partition_factors[ion] == pytest.approx(
            permeate_product
        )


# ---------------------------------------------------------------------------
# explicit interface charge residuals
# ---------------------------------------------------------------------------

def test_interface_charge_residuals_returned_and_finite():
    inputs = _controlled_inputs()
    state = assemble_two_interface_state(inputs)

    assert math.isfinite(state.feed_side_charge_residual_mol_L)
    assert math.isfinite(state.permeate_side_charge_residual_mol_L)

    # Residual equals Sum(z_i * c_i,m) + X, computed independently.
    expected_feed = (
        sum(
            inputs.charges[ion] * state.feed_membrane_concentrations_mol_L[ion]
            for ion in state.ions
        )
        + inputs.fixed_charge_mol_L
    )
    assert state.feed_side_charge_residual_mol_L == pytest.approx(expected_feed)

    expected_permeate = (
        sum(
            inputs.charges[ion]
            * state.permeate_membrane_concentrations_mol_L[ion]
            for ion in state.ions
        )
        + inputs.fixed_charge_mol_L
    )
    assert state.permeate_side_charge_residual_mol_L == pytest.approx(
        expected_permeate
    )


def test_charge_residual_tracks_fixed_charge_perturbation():
    """Adding delta to the fixed charge shifts the residual by delta."""
    base = _controlled_inputs()
    base_state = assemble_two_interface_state(base)

    delta = 0.05
    perturbed = TwoInterfaceInputs(
        feed_bulk_concentrations_mol_L=dict(base.feed_bulk_concentrations_mol_L),
        permeate_concentrations_mol_L=dict(base.permeate_concentrations_mol_L),
        charges=dict(base.charges),
        ion_radii_nm=dict(base.ion_radii_nm),
        pore_radius_nm=base.pore_radius_nm,
        fixed_charge_mol_L=base.fixed_charge_mol_L + delta,
        feed_side_donnan_V=base.feed_side_donnan_V,
        permeate_side_donnan_V=base.permeate_side_donnan_V,
        pore_dielectric_constant=base.pore_dielectric_constant,
        temperature_K=base.temperature_K,
        bulk_dielectric_constant=base.bulk_dielectric_constant,
    )
    perturbed_state = assemble_two_interface_state(perturbed)

    assert perturbed_state.feed_side_charge_residual_mol_L == pytest.approx(
        base_state.feed_side_charge_residual_mol_L + delta
    )
    assert perturbed_state.permeate_side_charge_residual_mol_L == pytest.approx(
        base_state.permeate_side_charge_residual_mol_L + delta
    )


# ---------------------------------------------------------------------------
# hard-cutoff species handling
# ---------------------------------------------------------------------------

def test_hard_cutoff_species_flagged_and_zeroed_for_foo_case():
    case = foo2023_lmc_ph7_case()
    inputs = two_interface_inputs_from_case(
        case,
        permeate_concentrations_mol_L=_placeholder_permeate(case),
        feed_side_donnan_V=-0.02,
        permeate_side_donnan_V=-0.01,
    )
    state = assemble_two_interface_state(inputs)

    # Mg2+ (0.428 nm) exceeds the NF270 pore radius (0.416 nm): fully excluded.
    assert "Mg2+" in state.hard_cutoff_species
    assert state.has_hard_cutoff is True

    # A flagged species has zero steric factor and zero membrane concentration
    # on BOTH interfaces, rather than silently vanishing unnoticed.
    assert state.steric_factors["Mg2+"] == 0.0
    assert state.feed_membrane_concentrations_mol_L["Mg2+"] == 0.0
    assert state.permeate_membrane_concentrations_mol_L["Mg2+"] == 0.0

    # Non-cutoff ions are not flagged and are not zeroed.
    assert "Li+" not in state.hard_cutoff_species
    assert state.feed_membrane_concentrations_mol_L["Li+"] > 0.0


def test_no_hard_cutoff_when_all_ions_fit():
    state = assemble_two_interface_state(_controlled_inputs())
    assert state.hard_cutoff_species == ()
    assert state.has_hard_cutoff is False


# ---------------------------------------------------------------------------
# no mutation of the case dictionary
# ---------------------------------------------------------------------------

def test_case_dictionary_not_mutated():
    case = foo2023_lmc_ph7_case()
    reference = foo2023_lmc_ph7_case()

    inputs = two_interface_inputs_from_case(
        case,
        permeate_concentrations_mol_L=_placeholder_permeate(case),
    )
    assemble_two_interface_state(inputs)

    assert case == reference


def test_state_mappings_are_read_only():
    state = assemble_two_interface_state(_controlled_inputs())

    with pytest.raises(TypeError):
        state.feed_membrane_concentrations_mol_L["A+"] = 1.0

    # Frozen dataclass: attributes cannot be reassigned either.
    with pytest.raises(Exception):
        state.feed_side_charge_residual_mol_L = 0.0


def test_inputs_do_not_alias_source_dicts():
    source = {"A+": 0.10, "B-": 0.10}
    inputs = TwoInterfaceInputs(
        feed_bulk_concentrations_mol_L=source,
        permeate_concentrations_mol_L={"A+": 0.04, "B-": 0.04},
        charges={"A+": 1, "B-": -1},
        ion_radii_nm={"A+": 0.15, "B-": 0.15},
        pore_radius_nm=1.0,
        fixed_charge_mol_L=-0.02,
        feed_side_donnan_V=0.0,
        permeate_side_donnan_V=0.0,
        pore_dielectric_constant=45.0,
        temperature_K=298.15,
    )

    # Mutating the source afterwards must not affect the stored inputs.
    source["A+"] = 999.0
    assert inputs.feed_bulk_concentrations_mol_L["A+"] == 0.10


# ---------------------------------------------------------------------------
# Foo 2023 assembly for one flux with placeholder permeate
# ---------------------------------------------------------------------------

def test_foo2023_assembles_for_one_flux_with_placeholder_permeate():
    case = foo2023_lmc_ph7_case()
    permeate = _placeholder_permeate(case, fraction=0.5)

    inputs = two_interface_inputs_from_case(
        case,
        permeate_concentrations_mol_L=permeate,
        feed_side_donnan_V=-0.02,
        permeate_side_donnan_V=-0.01,
    )
    state = assemble_two_interface_state(inputs)

    expected_ions = {"Li+", "Mg2+", "Cl-"}
    assert set(state.ions) == expected_ions
    assert set(state.feed_membrane_concentrations_mol_L) == expected_ions
    assert set(state.permeate_membrane_concentrations_mol_L) == expected_ions

    # Everything is finite and non-negative for these placeholder inputs.
    for ion in state.ions:
        assert math.isfinite(state.feed_membrane_concentrations_mol_L[ion])
        assert math.isfinite(state.permeate_membrane_concentrations_mol_L[ion])
        assert state.feed_membrane_concentrations_mol_L[ion] >= 0.0
        assert state.permeate_membrane_concentrations_mol_L[ion] >= 0.0

    assert math.isfinite(state.feed_side_charge_residual_mol_L)
    assert math.isfinite(state.permeate_side_charge_residual_mol_L)

    # Fixed charge is echoed straight from the case.
    assert state.fixed_charge_mol_L == case["membrane_parameters"][
        "fixed_charge_mol_L"
    ]
