"""
Membrane-interface calculation utilities for OpenENPD-Li.

This module connects published validation-case definitions to the
partitioning physics modules.
"""

from openenpd.partitioning import (
    solve_partitioning_potential,
    total_partition_factors,
    membrane_interface_concentrations,
    partitioning_charge_balance,
)


def compute_interface_state(case):
    """
    Compute membrane-interface state for a validation case.

    This includes:
    - Donnan potential solved with steric and dielectric effects included
    - total partition factors
    - membrane-interface concentrations
    - final electroneutrality residual

    Parameters
    ----------
    case : dict
        Validation case definition, such as foo2023_lmc_ph7_case().

    Returns
    -------
    dict
        Computed membrane-interface state.
    """
    bulk_concentrations = case["bulk_concentrations_mol_L"]
    charges = case["charges"]
    ion_radii_nm = case["ion_radii_nm"]
    params = case["membrane_parameters"]

    pore_radius_nm = params["pore_radius_nm"]
    fixed_charge_mol_L = params["fixed_charge_mol_L"]
    pore_dielectric_constant = params["pore_dielectric_constant"]
    temperature_K = case["temperature_K"]

    delta_psi_V = solve_partitioning_potential(
        bulk_concentrations=bulk_concentrations,
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=pore_radius_nm,
        fixed_charge_mol_L=fixed_charge_mol_L,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
    )

    partition_factors = total_partition_factors(
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=pore_radius_nm,
        delta_psi_V=delta_psi_V,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
    )

    membrane_concentrations = membrane_interface_concentrations(
        bulk_concentrations=bulk_concentrations,
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=pore_radius_nm,
        delta_psi_V=delta_psi_V,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
    )

    charge_residual = partitioning_charge_balance(
        delta_psi_V=delta_psi_V,
        bulk_concentrations=bulk_concentrations,
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=pore_radius_nm,
        fixed_charge_mol_L=fixed_charge_mol_L,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
    )

    return {
        "delta_psi_V": delta_psi_V,
        "partition_factors": partition_factors,
        "membrane_concentrations_mol_L": membrane_concentrations,
        "charge_residual_mol_L": charge_residual,
    }
