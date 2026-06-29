"""
Rejection and permeate-concentration utilities for OpenENPD-Li.

This module provides simple membrane-scale relations for converting
ion fluxes into permeate concentrations and apparent/species rejection.

These functions are intentionally model-agnostic. They can be used with
experimental fluxes, ENP-predicted fluxes, or simplified analytical
transport approximations.
"""


def permeate_concentration_from_flux(ion_flux_mol_m2_s, water_flux_m_s):
    """
    Compute permeate concentration from ion flux and water flux.

    Parameters
    ----------
    ion_flux_mol_m2_s : float
        Ion molar flux in mol m^-2 s^-1.

    water_flux_m_s : float
        Water flux in m/s.

    Returns
    -------
    float
        Permeate concentration in mol/m^3.
    """
    if water_flux_m_s == 0.0:
        raise ValueError("Water flux must be nonzero.")

    return ion_flux_mol_m2_s / water_flux_m_s


def species_rejection(feed_concentration, permeate_concentration):
    """
    Compute apparent/species rejection.

        R_i = 1 - C_i,p / C_i,f

    Parameters
    ----------
    feed_concentration : float
        Feed concentration.

    permeate_concentration : float
        Permeate concentration.

    Returns
    -------
    float
        Dimensionless rejection.
    """
    if feed_concentration == 0.0:
        raise ValueError("Feed concentration must be nonzero.")

    return 1.0 - permeate_concentration / feed_concentration


def species_rejections(feed_concentrations, permeate_concentrations):
    """
    Compute species rejection for multiple ions.

    Parameters
    ----------
    feed_concentrations : dict
        Feed concentrations.

    permeate_concentrations : dict
        Permeate concentrations.

    Returns
    -------
    dict
        Species rejection for each ion.
    """
    rejections = {}

    for ion, feed_concentration in feed_concentrations.items():
        rejections[ion] = species_rejection(
            feed_concentration=feed_concentration,
            permeate_concentration=permeate_concentrations[ion],
        )

    return rejections


def separation_factor(rejection_a, rejection_b):
    """
    Compute a rejection-based separation factor.

    For lithium/magnesium selectivity, this can be used as:

        SF_Li_Mg = (1 - R_Li) / (1 - R_Mg)

    where larger values indicate stronger preferential lithium passage.

    Parameters
    ----------
    rejection_a : float
        Rejection of preferentially permeating species.

    rejection_b : float
        Rejection of more strongly retained species.

    Returns
    -------
    float
        Dimensionless separation factor.
    """
    denominator = 1.0 - rejection_b

    if denominator == 0.0:
        raise ValueError("Separation factor is undefined when rejection_b is 1.")

    return (1.0 - rejection_a) / denominator
