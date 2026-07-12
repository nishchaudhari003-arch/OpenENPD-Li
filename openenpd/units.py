"""
Unit-conversion utilities for OpenENPD-Li.
"""


def lmh_to_m_s(value):
    """
    Convert water flux from LMH to m/s.

    LMH means liters per square meter per hour.

    Parameters
    ----------
    value : float
        Flux in L m^-2 h^-1.

    Returns
    -------
    float
        Flux in m s^-1.
    """
    return value * 1e-3 / 3600


def m_s_to_lmh(value):
    """
    Convert water flux from m/s to LMH.

    Parameters
    ----------
    value : float
        Flux in m s^-1.

    Returns
    -------
    float
        Flux in L m^-2 h^-1.
    """
    return value * 3600 / 1e-3


def nm_to_m(value):
    """
    Convert nanometers to meters.
    """
    return value * 1e-9
def mol_L_to_mol_m3(value):
    """
    Convert concentration from mol/L to mol/m^3.
    """
    return value * 1000.0


def mol_m3_to_mol_L(value):
    """
    Convert concentration from mol/m^3 to mol/L.
    """
    return value / 1000.0
