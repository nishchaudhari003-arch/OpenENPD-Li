"""
Extended Nernst–Planck transport utilities for OpenENPD-Li.

This module implements local ion flux expressions containing:
- diffusion
- electromigration
- convection

The local one-dimensional extended Nernst–Planck form used here is:

    J_i = -K_d,i D_i dc_i/dx
          -K_d,i D_i z_i F c_i/(RT) dpsi/dx
          +K_c,i c_i J_v

where:
- J_i is ion molar flux in mol m^-2 s^-1
- D_i is ion diffusivity in m^2 s^-1
- c_i is concentration in mol m^-3
- psi is electric potential in V
- J_v is water velocity/flux in m s^-1
- K_d,i is diffusive hindrance factor
- K_c,i is convective hindrance factor
"""

from openenpd.constants import R_GAS, FARADAY


def diffusive_flux(
    diffusivity_m2_s,
    concentration_gradient_mol_m4,
    diffusive_hindrance=1.0,
):
    """
    Compute diffusive contribution to ion flux.

    Parameters
    ----------
    diffusivity_m2_s : float
        Ion diffusivity in m^2/s.

    concentration_gradient_mol_m4 : float
        Concentration gradient dc/dx in mol/m^4.

    diffusive_hindrance : float, optional
        Diffusive hindrance factor.

    Returns
    -------
    float
        Diffusive flux in mol m^-2 s^-1.
    """
    return -diffusive_hindrance * diffusivity_m2_s * concentration_gradient_mol_m4


def electromigration_flux(
    diffusivity_m2_s,
    charge,
    concentration_mol_m3,
    potential_gradient_V_m,
    temperature_K,
    diffusive_hindrance=1.0,
):
    """
    Compute electromigration contribution to ion flux.

    Parameters
    ----------
    diffusivity_m2_s : float
        Ion diffusivity in m^2/s.

    charge : int or float
        Ion charge number.

    concentration_mol_m3 : float
        Ion concentration in mol/m^3.

    potential_gradient_V_m : float
        Electric potential gradient dpsi/dx in V/m.

    temperature_K : float
        Temperature in kelvin.

    diffusive_hindrance : float, optional
        Diffusive hindrance factor.

    Returns
    -------
    float
        Electromigration flux in mol m^-2 s^-1.
    """
    return (
        -diffusive_hindrance
        * diffusivity_m2_s
        * charge
        * FARADAY
        * concentration_mol_m3
        * potential_gradient_V_m
        / (R_GAS * temperature_K)
    )


def convective_flux(
    concentration_mol_m3,
    water_flux_m_s,
    convective_hindrance=1.0,
):
    """
    Compute convective contribution to ion flux.

    Parameters
    ----------
    concentration_mol_m3 : float
        Ion concentration in mol/m^3.

    water_flux_m_s : float
        Water flux in m/s.

    convective_hindrance : float, optional
        Convective hindrance factor.

    Returns
    -------
    float
        Convective flux in mol m^-2 s^-1.
    """
    return convective_hindrance * concentration_mol_m3 * water_flux_m_s


def enp_flux(
    diffusivity_m2_s,
    charge,
    concentration_mol_m3,
    concentration_gradient_mol_m4,
    potential_gradient_V_m,
    water_flux_m_s,
    temperature_K,
    diffusive_hindrance=1.0,
    convective_hindrance=1.0,
):
    """
    Compute total local extended Nernst–Planck ion flux.

    Returns
    -------
    float
        Total ion flux in mol m^-2 s^-1.
    """
    j_diff = diffusive_flux(
        diffusivity_m2_s=diffusivity_m2_s,
        concentration_gradient_mol_m4=concentration_gradient_mol_m4,
        diffusive_hindrance=diffusive_hindrance,
    )

    j_migration = electromigration_flux(
        diffusivity_m2_s=diffusivity_m2_s,
        charge=charge,
        concentration_mol_m3=concentration_mol_m3,
        potential_gradient_V_m=potential_gradient_V_m,
        temperature_K=temperature_K,
        diffusive_hindrance=diffusive_hindrance,
    )

    j_convection = convective_flux(
        concentration_mol_m3=concentration_mol_m3,
        water_flux_m_s=water_flux_m_s,
        convective_hindrance=convective_hindrance,
    )

    return j_diff + j_migration + j_convection
