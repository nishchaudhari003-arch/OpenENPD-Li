"""
Steric partitioning utilities for OpenENPD-Li.

This module implements a simple steric partition coefficient for ions
entering cylindrical membrane pores:

    phi_i = (1 - lambda_i)^2

where:

    lambda_i = r_i / r_p

Here r_i is the ion effective radius and r_p is the membrane pore radius.

If lambda_i >= 1, the ion is fully excluded from the pore.
"""


def size_ratio(ion_radius_nm, pore_radius_nm):
    """
    Compute the ion-to-pore size ratio.

    Parameters
    ----------
    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    float
        Dimensionless size ratio lambda_i = r_i / r_p.
    """
    return ion_radius_nm / pore_radius_nm


def steric_partition_factor(ion_radius_nm, pore_radius_nm):
    """
    Compute the steric partition factor for an ion entering a pore.

    Parameters
    ----------
    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    float
        Dimensionless steric partition factor.
    """
    lambda_i = size_ratio(ion_radius_nm, pore_radius_nm)

    if lambda_i >= 1.0:
        return 0.0

    return (1.0 - lambda_i) ** 2


def steric_partition_factors(ion_radii_nm, pore_radius_nm):
    """
    Compute steric partition factors for multiple ions.

    Parameters
    ----------
    ion_radii_nm : dict
        Dictionary of ion effective radii in nanometers.
        Example: {"Li+": 0.382, "Mg2+": 0.428, "Cl-": 0.332}

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    dict
        Dictionary of steric partition factors.
    """
    factors = {}

    for ion, radius_nm in ion_radii_nm.items():
        factors[ion] = steric_partition_factor(
            ion_radius_nm=radius_nm,
            pore_radius_nm=pore_radius_nm,
        )

    return factors
