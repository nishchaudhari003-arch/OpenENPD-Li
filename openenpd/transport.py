"""
Membrane-scale transport helper utilities for OpenENPD-Li.

This module provides simple utilities needed to connect membrane-interface
partitioning results to the extended Nernst-Planck transport equation.
"""

from openenpd.units import nm_to_m, mol_L_to_mol_m3


def active_layer_thickness_m(case):
    """
    Extract active-layer thickness from a validation case and convert to meters.

    Parameters
    ----------
    case : dict
        Validation case definition.

    Returns
    -------
    float
        Active-layer thickness in meters.
    """
    thickness_nm = case["membrane_parameters"]["active_layer_thickness_nm"]
    return nm_to_m(thickness_nm)


def convert_concentrations_mol_L_to_mol_m3(concentrations_mol_L):
    """
    Convert a dictionary of concentrations from mol/L to mol/m^3.

    Parameters
    ----------
    concentrations_mol_L : dict
        Ion concentrations in mol/L.

    Returns
    -------
    dict
        Ion concentrations in mol/m^3.
    """
    return {
        ion: mol_L_to_mol_m3(concentration)
        for ion, concentration in concentrations_mol_L.items()
    }


def linear_concentration_gradient(
    upstream_concentration_mol_m3,
    downstream_concentration_mol_m3,
    thickness_m,
):
    """
    Compute a one-dimensional linear concentration gradient.

    The sign convention is:

        dc/dx = (c_downstream - c_upstream) / thickness

    Parameters
    ----------
    upstream_concentration_mol_m3 : float
        Concentration at membrane feed-side interface in mol/m^3.

    downstream_concentration_mol_m3 : float
        Concentration at membrane permeate-side interface in mol/m^3.

    thickness_m : float
        Membrane active-layer thickness in meters.

    Returns
    -------
    float
        Concentration gradient in mol/m^4.
    """
    if thickness_m == 0.0:
        raise ValueError("Membrane thickness must be nonzero.")

    return (
        downstream_concentration_mol_m3 - upstream_concentration_mol_m3
    ) / thickness_m


def linear_concentration_gradients(
    upstream_concentrations_mol_m3,
    downstream_concentrations_mol_m3,
    thickness_m,
):
    """
    Compute linear concentration gradients for multiple ions.

    Parameters
    ----------
    upstream_concentrations_mol_m3 : dict
        Feed-side membrane-interface concentrations in mol/m^3.

    downstream_concentrations_mol_m3 : dict
        Permeate-side membrane-interface concentrations in mol/m^3.

    thickness_m : float
        Membrane active-layer thickness in meters.

    Returns
    -------
    dict
        Concentration gradients in mol/m^4.
    """
    gradients = {}

    for ion, upstream_concentration in upstream_concentrations_mol_m3.items():
        gradients[ion] = linear_concentration_gradient(
            upstream_concentration_mol_m3=upstream_concentration,
            downstream_concentration_mol_m3=downstream_concentrations_mol_m3[ion],
            thickness_m=thickness_m,
        )

    return gradients
