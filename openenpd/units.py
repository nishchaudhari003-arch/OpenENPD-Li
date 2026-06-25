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
