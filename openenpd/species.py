"""
Ion/species definitions for OpenENPD-Li.

This module stores minimal ion identity information needed for
electroneutrality checks and transport-model setup.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ion:
    """
    Basic ion definition.

    Parameters
    ----------
    name : str
        Human-readable ion name.
    symbol : str
        Chemical symbol.
    charge : int
        Ionic charge number.
    """

    name: str
    symbol: str
    charge: int


LI = Ion(name="lithium", symbol="Li+", charge=1)
MG = Ion(name="magnesium", symbol="Mg2+", charge=2)
CL = Ion(name="chloride", symbol="Cl-", charge=-1)


COMMON_IONS = {
    "Li+": LI,
    "Mg2+": MG,
    "Cl-": CL,
}
