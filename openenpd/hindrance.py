"""
Hydrodynamic hindrance utilities for OpenENPD-Li.

These functions provide simple pore-size-dependent hindrance factors
for diffusion and convection.

The current implementation uses a conservative first baseline:

    K_d = (1 - lambda)^2
    K_c = (1 - lambda)^2

where:

    lambda = r_i / r_p

If lambda >= 1, the ion is fully hindered.

Later versions can replace these baseline expressions with more detailed
Deen/DSPM-DE hindrance correlations.
"""

from openenpd.steric import size_ratio


def diffusive_hindrance_factor(ion_radius_nm, pore_radius_nm):
    """
    Compute a baseline diffusive hindrance factor.

    Parameters
    ----------
    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    float
        Dimensionless diffusive hindrance factor.
    """
    lambda_i = size_ratio(
        ion_radius_nm=ion_radius_nm,
        pore_radius_nm=pore_radius_nm,
    )

    if lambda_i >= 1.0:
        return 0.0

    return (1.0 - lambda_i) ** 2


def convective_hindrance_factor(ion_radius_nm, pore_radius_nm):
    """
    Compute a baseline convective hindrance factor.

    Parameters
    ----------
    ion_radius_nm : float
        Effective ion radius in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    float
        Dimensionless convective hindrance factor.
    """
    lambda_i = size_ratio(
        ion_radius_nm=ion_radius_nm,
        pore_radius_nm=pore_radius_nm,
    )

    if lambda_i >= 1.0:
        return 0.0

    return (1.0 - lambda_i) ** 2


def hindrance_factors(ion_radii_nm, pore_radius_nm):
    """
    Compute diffusive and convective hindrance factors for multiple ions.

    Parameters
    ----------
    ion_radii_nm : dict
        Dictionary of ion effective radii in nanometers.

    pore_radius_nm : float
        Membrane pore radius in nanometers.

    Returns
    -------
    dict
        Dictionary containing diffusive and convective hindrance factors.
    """
    return {
        "diffusive": {
            ion: diffusive_hindrance_factor(
                ion_radius_nm=radius_nm,
                pore_radius_nm=pore_radius_nm,
            )
            for ion, radius_nm in ion_radii_nm.items()
        },
        "convective": {
            ion: convective_hindrance_factor(
                ion_radius_nm=radius_nm,
                pore_radius_nm=pore_radius_nm,
            )
            for ion, radius_nm in ion_radii_nm.items()
        },
    }
