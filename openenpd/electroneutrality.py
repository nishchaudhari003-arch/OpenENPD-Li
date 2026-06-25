"""
Electroneutrality utilities for multicomponent brine systems.
"""


def charge_balance(concentrations, charges):
    """
    Compute the charge balance of an electrolyte solution.

    Parameters
    ----------
    concentrations : dict
        Dictionary of ion concentrations in mol/L.
        Example: {"Li+": 0.0490, "Mg2+": 0.0843, "Cl-": 0.2172}

    charges : dict
        Dictionary of ion charge numbers.
        Example: {"Li+": 1, "Mg2+": 2, "Cl-": -1}

    Returns
    -------
    float
        Sum of z_i * c_i in mol/L charge-equivalent units.
        A perfectly electroneutral solution gives zero.
    """
    balance = 0.0

    for ion, concentration in concentrations.items():
        balance += charges[ion] * concentration

    return balance


def is_electroneutral(concentrations, charges, tolerance=1e-3):
    """
    Check whether a solution is electroneutral within a given tolerance.
    """
    return abs(charge_balance(concentrations, charges)) <= tolerance
