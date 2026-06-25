"""
Dielectric/Born exclusion utilities for OpenENPD-Li.

This module implements a Born-type dielectric exclusion penalty for
transferring an ion from bulk solution into a membrane pore with a
different dielectric constant.

The Born energy penalty is computed as:

    ΔG_i = N_A * z_i^2 * e^2 / (8π ε0 r_i) * (1/ε_p - 1/ε_b)

where:
- ΔG_i is the dielectric exclusion energy in J/mol
- N_A is Avogadro's number
- z_i is the ion charge number
- e is the elementary charge
- ε0 is vacuum permittivity
- r_i is the ion effective radius in meters
- ε_p is the membrane pore dielectric constant
- ε_b is the bulk solution dielectric constant

The corresponding partition factor is:

    K_i = exp(-ΔG_i / RT)
"""

import math

from openenpd.constants import R_GAS, EPSILON_0, AVOGADRO, FARADAY
from openenpd.units import nm_to_m


def elementary_charge():
    """
    Compute elementary charge from Faraday constant and Avogadro constant.

    Returns
    -------
    float
        Elementary charge in coulombs.
    """
    return FARADAY / AVOGADRO


def born_energy_j_per_mol(
    charge,
    ion_radius_nm,
    pore_dielectric_constant,
    bulk_dielectric_constant=80.1,
):
    """
    Compute Born dielectric exclusion energy.

    Parameters
    ----------
    charge : int or float
        Ion charge number.

    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    float
        Born energy penalty in J/mol.
    """
    radius_m = nm_to_m(ion_radius_nm)
    e_charge = elementary_charge()

    prefactor = (
        AVOGADRO
        * charge**2
        * e_charge**2
        / (8.0 * math.pi * EPSILON_0 * radius_m)
    )

    dielectric_term = (
        1.0 / pore_dielectric_constant
        - 1.0 / bulk_dielectric_constant
    )

    return prefactor * dielectric_term


def dielectric_partition_factor(
    charge,
    ion_radius_nm,
    pore_dielectric_constant,
    temperature_K,
    bulk_dielectric_constant=80.1,
):
    """
    Compute dielectric/Born partition factor.

    Parameters
    ----------
    charge : int or float
        Ion charge number.

    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    temperature_K : float
        Temperature in kelvin.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    float
        Dimensionless dielectric partition factor.
    """
    energy = born_energy_j_per_mol(
        charge=charge,
        ion_radius_nm=ion_radius_nm,
        pore_dielectric_constant=pore_dielectric_constant,
        bulk_dielectric_constant=bulk_dielectric_constant,
    )

    return math.exp(-energy / (R_GAS * temperature_K))


def dielectric_partition_factors(
    ion_radii_nm,
    charges,
    pore_dielectric_constant,
    temperature_K,
    bulk_dielectric_constant=80.1,
):
    """
    Compute dielectric partition factors for multiple ions.

    Parameters
    ----------
    ion_radii_nm : dict
        Dictionary of ion effective radii in nanometers.

    charges : dict
        Dictionary of ion charge numbers.

    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.

    temperature_K : float
        Temperature in kelvin.

    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    dict
        Dictionary of dielectric partition factors.
    """
    factors = {}

    for ion, radius_nm in ion_radii_nm.items():
        factors[ion] = dielectric_partition_factor(
            charge=charges[ion],
            ion_radius_nm=radius_nm,
            pore_dielectric_constant=pore_dielectric_constant,
            temperature_K=temperature_K,
            bulk_dielectric_constant=bulk_dielectric_constant,
        )

    return factors
