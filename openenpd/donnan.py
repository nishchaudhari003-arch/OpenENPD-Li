"""
Donnan equilibrium utilities for OpenENPD-Li.

This module implements the ideal Donnan partitioning relation at a
solution/membrane interface:

    c_i,m = c_i,b * exp(-z_i F Δψ_D / RT)

where:
- c_i,m is the membrane-side concentration
- c_i,b is the bulk solution concentration
- z_i is the ion charge number
- Δψ_D is the Donnan potential difference
- F is Faraday's constant
- R is the gas constant
- T is temperature
"""

import math

from openenpd.constants import R_GAS, FARADAY


def thermal_voltage(temperature_K):
    """
    Compute the thermal voltage RT/F.

    Parameters
    ----------
    temperature_K : float
        Temperature in kelvin.

    Returns
    -------
    float
        Thermal voltage in volts.
    """
    return R_GAS * temperature_K / FARADAY


def donnan_partition_factor(charge, delta_psi_V, temperature_K):
    """
    Compute the ideal Donnan partition factor for one ion.

    Parameters
    ----------
    charge : int or float
        Ion charge number z_i.

    delta_psi_V : float
        Donnan potential difference in volts.

    temperature_K : float
        Temperature in kelvin.

    Returns
    -------
    float
        Dimensionless partition factor.
    """
    exponent = -charge * FARADAY * delta_psi_V / (R_GAS * temperature_K)
    return math.exp(exponent)


def donnan_partition_concentrations(concentrations, charges, delta_psi_V, temperature_K):
    """
    Compute membrane-side ion concentrations from bulk concentrations
    using ideal Donnan partitioning.

    Parameters
    ----------
    concentrations : dict
        Bulk concentrations in mol/L.

    charges : dict
        Ion charge numbers.

    delta_psi_V : float
        Donnan potential difference in volts.

    temperature_K : float
        Temperature in kelvin.

    Returns
    -------
    dict
        Membrane-side concentrations in mol/L.
    """
    membrane_concentrations = {}

    for ion, concentration in concentrations.items():
        factor = donnan_partition_factor(
            charge=charges[ion],
            delta_psi_V=delta_psi_V,
            temperature_K=temperature_K,
        )
        membrane_concentrations[ion] = concentration * factor

    return membrane_concentrations
