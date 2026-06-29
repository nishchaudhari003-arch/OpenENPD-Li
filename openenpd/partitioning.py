"""
Combined membrane partitioning utilities for OpenENPD-Li.

This module combines Donnan, steric, and dielectric/Born exclusion
effects into a total membrane-entry partition factor:

    K_i = K_Donnan,i * K_steric,i * K_dielectric,i
"""

from openenpd.donnan import donnan_partition_factor
from openenpd.steric import steric_partition_factor
from openenpd.dielectric import dielectric_partition_factor


def total_partition_factor(
    charge,
    ion_radius_nm,
    pore_radius_nm,
    delta_psi_V,
    pore_dielectric_constant,
    temperature_K,
    bulk_dielectric_constant=80.1,
):
    """
    Compute total membrane-entry partition factor for one ion.

    Parameters
    ----------
    charge : int or float
        Ion charge number.

    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    delta_psi_V : float
        Donnan potential difference in volts.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    temperature_K : float
        Temperature in kelvin.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    float
        Total dimensionless membrane-entry partition factor.
    """
    donnan_factor = donnan_partition_factor(
        charge=charge,
        delta_psi_V=delta_psi_V,
        temperature_K=temperature_K,
    )

    steric_factor = steric_partition_factor(
        ion_radius_nm=ion_radius_nm,
        pore_radius_nm=pore_radius_nm,
    )

    dielectric_factor = dielectric_partition_factor(
        charge=charge,
        ion_radius_nm=ion_radius_nm,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
        bulk_dielectric_constant=bulk_dielectric_constant,
    )

    return donnan_factor * steric_factor * dielectric_factor


def total_partition_factors(
    charges,
    ion_radii_nm,
    pore_radius_nm,
    delta_psi_V,
    pore_dielectric_constant,
    temperature_K,
    bulk_dielectric_constant=80.1,
):
    """
    Compute total membrane-entry partition factors for multiple ions.

    Parameters
    ----------
    charges : dict
        Dictionary of ion charge numbers.

    ion_radii_nm : dict
        Dictionary of ion effective radii in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    delta_psi_V : float
        Donnan potential difference in volts.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    temperature_K : float
        Temperature in kelvin.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    dict
        Dictionary of total partition factors.
    """
    factors = {}

    for ion, charge in charges.items():
        factors[ion] = total_partition_factor(
            charge=charge,
            ion_radius_nm=ion_radii_nm[ion],
            pore_radius_nm=pore_radius_nm,
            delta_psi_V=delta_psi_V,
            pore_dielectric_constant=pore_dielectric_constant,
            temperature_K=temperature_K,
            bulk_dielectric_constant=bulk_dielectric_constant,
        )

    return factors


def membrane_interface_concentrations(
    bulk_concentrations,
    charges,
    ion_radii_nm,
    pore_radius_nm,
    delta_psi_V,
    pore_dielectric_constant,
    temperature_K,
    bulk_dielectric_constant=80.1,
):
    """
    Compute membrane-interface concentrations after combined partitioning.

    Parameters
    ----------
    bulk_concentrations : dict
        Bulk ion concentrations in mol/L.

    charges : dict
        Dictionary of ion charge numbers.

    ion_radii_nm : dict
        Dictionary of ion effective radii in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    delta_psi_V : float
        Donnan potential difference in volts.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    temperature_K : float
        Temperature in kelvin.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    dict
        Membrane-interface concentrations in mol/L.
    """
    factors = total_partition_factors(
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=pore_radius_nm,
        delta_psi_V=delta_psi_V,
        pore_dielectric_constant=pore_dielectric_constant,
        temperature_K=temperature_K,
        bulk_dielectric_constant=bulk_dielectric_constant,
    )

    membrane_concentrations = {}

    for ion, concentration in bulk_concentrations.items():
        membrane_concentrations[ion] = concentration * factors[ion]

    return membrane_concentrations
