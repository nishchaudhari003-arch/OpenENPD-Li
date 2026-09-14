"""
Two-interface ENP-Donnan state assembler for OpenENPD-Li.

Membrane-based nanofiltration is a *two-interface* problem: ions partition
across a feed-side solution/membrane interface, are transported through the
active layer, and partition again across a permeate-side membrane/solution
interface. The existing simplified baseline (``openenpd.solver``) collapses
this to a single feed-side partitioning step plus a linear-profile transport
approximation. That baseline over-predicts rejection, so the roadmap moves
toward a full two-interface algebraic ENP-Donnan formulation.

This module implements the *first, deliberately non-solving* piece of that
formulation: an **immutable, optimizer-free state assembler**. Given a fully
specified set of two-interface inputs -- including the feed-side and
permeate-side Donnan potentials as *provided* values -- it evaluates every
interface quantity in closed form and returns them together with the
electroneutrality residual at each interface.

Scope and explicit assumptions
-------------------------------
- The assembler does **not** solve the nonlinear two-interface system. In
  particular it never calls ``scipy.optimize.root``/``brentq`` and never
  tunes any parameter. The two Donnan potentials and the permeate composition
  are treated as *inputs*; the interface charge residuals it returns are the
  quantities a later solver (Milestone 2/3) will drive to zero.
- Partitioning physics is reused verbatim from the existing modules so the
  assembler stays consistent with the baseline:

      K_i = K_Donnan,i(Δψ) * K_steric,i * K_dielectric,i

  (see ``openenpd.partitioning.total_partition_factor``). The steric and
  dielectric factors are potential-independent and therefore shared by both
  interfaces; only the Donnan factor differs between the feed side and the
  permeate side because each side has its own Donnan potential.
- Ideal activities are assumed (no activity coefficients), matching the rest
  of the current toolkit.
- **Hard-cutoff species** (``λ_i = r_i / r_p >= 1``) have a steric factor of
  exactly ``0.0``. Their membrane-interface concentration is therefore ``0.0``
  on both sides regardless of the Donnan potentials. This is physically the
  intended "fully size-excluded" behaviour, but numerically it can silently
  break a downstream solver (zero concentrations, undefined rejection, a
  singular Jacobian). The assembler therefore *marks* these species explicitly
  in :attr:`TwoInterfaceState.hard_cutoff_species` rather than letting them
  disappear unnoticed.

The state object is intentionally self-contained: it echoes its inputs
(concentrations, charges, potentials, fixed charge) alongside the assembled
outputs, so that the residual functions and solver added in later milestones
can read everything they need from one immutable object.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

from openenpd.dielectric import dielectric_partition_factor
from openenpd.donnan import donnan_partition_factor
from openenpd.steric import size_ratio, steric_partition_factor

__all__ = [
    "TwoInterfaceInputs",
    "TwoInterfaceState",
    "assemble_two_interface_state",
    "two_interface_inputs_from_case",
]


def _read_only(mapping):
    """
    Return an independent, read-only view of a mapping.

    A shallow copy is taken first so the returned view never aliases (and can
    never mutate) the caller's dictionary -- in particular a validation-case
    dictionary. The result is a ``MappingProxyType`` and therefore rejects
    item assignment.

    Parameters
    ----------
    mapping : Mapping
        Source mapping (e.g. a dict of ion -> value).

    Returns
    -------
    types.MappingProxyType
        Read-only view over a private copy of ``mapping``.
    """
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class TwoInterfaceInputs:
    """
    Immutable inputs for a two-interface ENP-Donnan state.

    All ion-keyed mappings are defensively copied into read-only views on
    construction, so passing a validation-case dictionary here never mutates
    it and the resulting object cannot be altered in place.

    Parameters
    ----------
    feed_bulk_concentrations_mol_L : Mapping[str, float]
        Bulk feed concentrations in mol/L.
    permeate_concentrations_mol_L : Mapping[str, float]
        Permeate (product) bulk concentrations in mol/L. In a full solve these
        are unknowns; here they are provided (e.g. placeholder values).
    charges : Mapping[str, float]
        Ion charge numbers z_i.
    ion_radii_nm : Mapping[str, float]
        Effective ion radii in nanometers.
    pore_radius_nm : float
        Membrane pore radius in nanometers.
    fixed_charge_mol_L : float
        Fixed membrane charge concentration X in mol/L (negative for a
        negatively charged membrane).
    feed_side_donnan_V : float
        Feed-side Donnan potential difference Δψ_D,feed in volts (provided,
        not solved).
    permeate_side_donnan_V : float
        Permeate-side Donnan potential difference Δψ_D,perm in volts (provided,
        not solved).
    pore_dielectric_constant : float
        Dielectric constant inside the membrane pore.
    temperature_K : float
        Temperature in kelvin.
    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Raises
    ------
    ValueError
        If the ion sets of the concentration, charge, and radius mappings are
        not identical, or if the temperature is not positive.
    """

    feed_bulk_concentrations_mol_L: Mapping[str, float]
    permeate_concentrations_mol_L: Mapping[str, float]
    charges: Mapping[str, float]
    ion_radii_nm: Mapping[str, float]
    pore_radius_nm: float
    fixed_charge_mol_L: float
    feed_side_donnan_V: float
    permeate_side_donnan_V: float
    pore_dielectric_constant: float
    temperature_K: float
    bulk_dielectric_constant: float = 80.1

    def __post_init__(self):
        # Freeze every ion-keyed mapping into an independent read-only view.
        # object.__setattr__ is required because the dataclass is frozen.
        object.__setattr__(
            self,
            "feed_bulk_concentrations_mol_L",
            _read_only(self.feed_bulk_concentrations_mol_L),
        )
        object.__setattr__(
            self,
            "permeate_concentrations_mol_L",
            _read_only(self.permeate_concentrations_mol_L),
        )
        object.__setattr__(self, "charges", _read_only(self.charges))
        object.__setattr__(self, "ion_radii_nm", _read_only(self.ion_radii_nm))

        self._validate()

    def _validate(self):
        ions = set(self.feed_bulk_concentrations_mol_L)

        for label, mapping in (
            ("permeate_concentrations_mol_L", self.permeate_concentrations_mol_L),
            ("charges", self.charges),
            ("ion_radii_nm", self.ion_radii_nm),
        ):
            if set(mapping) != ions:
                raise ValueError(
                    "Inconsistent ion sets: "
                    f"{label} keys {sorted(mapping)} do not match feed ions "
                    f"{sorted(ions)}."
                )

        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")

    @property
    def ions(self) -> Tuple[str, ...]:
        """Ordered ion labels, taken from the feed concentrations."""
        return tuple(self.feed_bulk_concentrations_mol_L.keys())


@dataclass(frozen=True)
class TwoInterfaceState:
    """
    Assembled, immutable two-interface ENP-Donnan state.

    This object bundles the assembled interface quantities together with an
    echo of the inputs, so downstream residual functions and solvers can read
    a complete description from a single place. It performs no solving.

    Attributes
    ----------
    ions : tuple of str
        Ordered ion labels.
    charges : Mapping[str, float]
        Echoed ion charge numbers.
    fixed_charge_mol_L : float
        Echoed fixed membrane charge concentration X in mol/L.
    feed_side_donnan_V, permeate_side_donnan_V : float
        Echoed Donnan potentials in volts.
    feed_bulk_concentrations_mol_L : Mapping[str, float]
        Echoed bulk feed concentrations in mol/L.
    permeate_concentrations_mol_L : Mapping[str, float]
        Echoed permeate concentrations in mol/L.
    size_ratios : Mapping[str, float]
        λ_i = r_i / r_p for each ion.
    steric_factors : Mapping[str, float]
        Potential-independent steric partition factors (0.0 for hard cutoff).
    dielectric_factors : Mapping[str, float]
        Potential-independent Born/dielectric partition factors.
    feed_side_donnan_factors, permeate_side_donnan_factors : Mapping[str, float]
        Donnan partition factors evaluated at each side's Donnan potential.
    feed_side_partition_factors, permeate_side_partition_factors : Mapping[str, float]
        Total partition factors K_i = K_Donnan * K_steric * K_dielectric for
        each interface.
    feed_membrane_concentrations_mol_L : Mapping[str, float]
        Feed-side membrane-interface concentrations: c_feed_bulk * K_feed.
    permeate_membrane_concentrations_mol_L : Mapping[str, float]
        Permeate-side membrane-interface concentrations: c_permeate * K_perm.
    feed_side_charge_residual_mol_L, permeate_side_charge_residual_mol_L : float
        Interface electroneutrality residuals Σ(z_i c_i,m) + X at each side.
        These are returned explicitly and are the targets a later solver drives
        to zero; they are generally nonzero for arbitrary provided potentials.
    hard_cutoff_species : tuple of str
        Ions with λ_i >= 1 (steric factor exactly 0.0). Their membrane
        concentrations are 0.0 on both sides. Flagged so downstream code
        handles them explicitly instead of silently propagating zeros.
    """

    ions: Tuple[str, ...]
    charges: Mapping[str, float]
    fixed_charge_mol_L: float
    feed_side_donnan_V: float
    permeate_side_donnan_V: float
    feed_bulk_concentrations_mol_L: Mapping[str, float]
    permeate_concentrations_mol_L: Mapping[str, float]
    size_ratios: Mapping[str, float]
    steric_factors: Mapping[str, float]
    dielectric_factors: Mapping[str, float]
    feed_side_donnan_factors: Mapping[str, float]
    permeate_side_donnan_factors: Mapping[str, float]
    feed_side_partition_factors: Mapping[str, float]
    permeate_side_partition_factors: Mapping[str, float]
    feed_membrane_concentrations_mol_L: Mapping[str, float]
    permeate_membrane_concentrations_mol_L: Mapping[str, float]
    feed_side_charge_residual_mol_L: float
    permeate_side_charge_residual_mol_L: float
    hard_cutoff_species: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_hard_cutoff(self) -> bool:
        """True if any ion is fully size-excluded (λ_i >= 1)."""
        return len(self.hard_cutoff_species) > 0


def _interface_charge_residual(concentrations, charges, fixed_charge_mol_L):
    """
    Compute the electroneutrality residual Σ(z_i c_i) + X at one interface.

    A value of zero means the mobile charge exactly balances the fixed
    membrane charge. This mirrors
    ``openenpd.partitioning.partitioning_charge_balance`` but takes
    already-partitioned membrane concentrations directly, without solving.

    Parameters
    ----------
    concentrations : Mapping[str, float]
        Membrane-interface concentrations in mol/L.
    charges : Mapping[str, float]
        Ion charge numbers.
    fixed_charge_mol_L : float
        Fixed membrane charge concentration X in mol/L.

    Returns
    -------
    float
        Charge-balance residual in mol/L charge-equivalent units.
    """
    mobile_charge = 0.0
    for ion, concentration in concentrations.items():
        mobile_charge += charges[ion] * concentration
    return mobile_charge + fixed_charge_mol_L


def assemble_two_interface_state(inputs: TwoInterfaceInputs) -> TwoInterfaceState:
    """
    Assemble a two-interface ENP-Donnan state from fully specified inputs.

    This is a pure, deterministic, optimizer-free evaluation. For each ion it
    computes the steric, dielectric, and (per-side) Donnan factors, combines
    them into per-interface total partition factors, applies them to the
    provided bulk concentrations to get membrane-interface concentrations, and
    reports the electroneutrality residual at each interface. Hard-cutoff
    species are identified and flagged.

    Parameters
    ----------
    inputs : TwoInterfaceInputs
        Fully specified two-interface inputs.

    Returns
    -------
    TwoInterfaceState
        The assembled, immutable state.

    Notes
    -----
    The partition-factor composition ``K = K_Donnan * K_steric * K_dielectric``
    is identical to ``openenpd.partitioning.total_partition_factor``; the
    factors are computed here from the same component functions so that the
    individual contributions can also be inspected.
    """
    ions = inputs.ions

    # Potential-independent factors (shared by both interfaces).
    size_ratios = {}
    steric_factors = {}
    dielectric_factors = {}
    for ion in ions:
        size_ratios[ion] = size_ratio(
            ion_radius_nm=inputs.ion_radii_nm[ion],
            pore_radius_nm=inputs.pore_radius_nm,
        )
        steric_factors[ion] = steric_partition_factor(
            ion_radius_nm=inputs.ion_radii_nm[ion],
            pore_radius_nm=inputs.pore_radius_nm,
        )
        dielectric_factors[ion] = dielectric_partition_factor(
            charge=inputs.charges[ion],
            ion_radius_nm=inputs.ion_radii_nm[ion],
            pore_dielectric_constant=inputs.pore_dielectric_constant,
            temperature_K=inputs.temperature_K,
            bulk_dielectric_constant=inputs.bulk_dielectric_constant,
        )

    # Per-side Donnan factors: only the Donnan potential differs between sides.
    feed_donnan_factors = {}
    permeate_donnan_factors = {}
    for ion in ions:
        feed_donnan_factors[ion] = donnan_partition_factor(
            charge=inputs.charges[ion],
            delta_psi_V=inputs.feed_side_donnan_V,
            temperature_K=inputs.temperature_K,
        )
        permeate_donnan_factors[ion] = donnan_partition_factor(
            charge=inputs.charges[ion],
            delta_psi_V=inputs.permeate_side_donnan_V,
            temperature_K=inputs.temperature_K,
        )

    # Total partition factors per interface.
    feed_partition_factors = {}
    permeate_partition_factors = {}
    for ion in ions:
        common = steric_factors[ion] * dielectric_factors[ion]
        feed_partition_factors[ion] = feed_donnan_factors[ion] * common
        permeate_partition_factors[ion] = permeate_donnan_factors[ion] * common

    # Membrane-interface concentrations on each side.
    feed_membrane = {}
    permeate_membrane = {}
    for ion in ions:
        feed_membrane[ion] = (
            inputs.feed_bulk_concentrations_mol_L[ion] * feed_partition_factors[ion]
        )
        permeate_membrane[ion] = (
            inputs.permeate_concentrations_mol_L[ion]
            * permeate_partition_factors[ion]
        )

    # Explicit interface electroneutrality residuals (targets for a later
    # solver; not driven to zero here).
    feed_residual = _interface_charge_residual(
        concentrations=feed_membrane,
        charges=inputs.charges,
        fixed_charge_mol_L=inputs.fixed_charge_mol_L,
    )
    permeate_residual = _interface_charge_residual(
        concentrations=permeate_membrane,
        charges=inputs.charges,
        fixed_charge_mol_L=inputs.fixed_charge_mol_L,
    )

    # Hard-cutoff species: fully size-excluded (steric factor collapses to 0).
    hard_cutoff_species = tuple(
        ion
        for ion in ions
        if size_ratios[ion] >= 1.0 or steric_factors[ion] == 0.0
    )

    return TwoInterfaceState(
        ions=ions,
        charges=_read_only(inputs.charges),
        fixed_charge_mol_L=inputs.fixed_charge_mol_L,
        feed_side_donnan_V=inputs.feed_side_donnan_V,
        permeate_side_donnan_V=inputs.permeate_side_donnan_V,
        feed_bulk_concentrations_mol_L=_read_only(
            inputs.feed_bulk_concentrations_mol_L
        ),
        permeate_concentrations_mol_L=_read_only(
            inputs.permeate_concentrations_mol_L
        ),
        size_ratios=_read_only(size_ratios),
        steric_factors=_read_only(steric_factors),
        dielectric_factors=_read_only(dielectric_factors),
        feed_side_donnan_factors=_read_only(feed_donnan_factors),
        permeate_side_donnan_factors=_read_only(permeate_donnan_factors),
        feed_side_partition_factors=_read_only(feed_partition_factors),
        permeate_side_partition_factors=_read_only(permeate_partition_factors),
        feed_membrane_concentrations_mol_L=_read_only(feed_membrane),
        permeate_membrane_concentrations_mol_L=_read_only(permeate_membrane),
        feed_side_charge_residual_mol_L=feed_residual,
        permeate_side_charge_residual_mol_L=permeate_residual,
        hard_cutoff_species=hard_cutoff_species,
    )


def two_interface_inputs_from_case(
    case,
    permeate_concentrations_mol_L,
    feed_side_donnan_V=0.0,
    permeate_side_donnan_V=0.0,
    bulk_dielectric_constant=80.1,
) -> TwoInterfaceInputs:
    """
    Build :class:`TwoInterfaceInputs` from a validation-case dictionary.

    This helper only *reads* from ``case`` (and copies what it reads), so the
    case dictionary is never mutated. The Donnan potentials default to ``0.0``
    V placeholders because this milestone does not solve for them; callers may
    pass any provided values.

    Parameters
    ----------
    case : dict
        Validation case definition, such as ``foo2023_lmc_ph7_case()``.
    permeate_concentrations_mol_L : Mapping[str, float]
        Provided permeate concentrations in mol/L (unknowns in a full solve).
    feed_side_donnan_V : float, optional
        Provided feed-side Donnan potential in volts.
    permeate_side_donnan_V : float, optional
        Provided permeate-side Donnan potential in volts.
    bulk_dielectric_constant : float, optional
        Dielectric constant of the bulk aqueous phase.

    Returns
    -------
    TwoInterfaceInputs
        Inputs ready for :func:`assemble_two_interface_state`.
    """
    params = case["membrane_parameters"]

    return TwoInterfaceInputs(
        feed_bulk_concentrations_mol_L=case["bulk_concentrations_mol_L"],
        permeate_concentrations_mol_L=permeate_concentrations_mol_L,
        charges=case["charges"],
        ion_radii_nm=case["ion_radii_nm"],
        pore_radius_nm=params["pore_radius_nm"],
        fixed_charge_mol_L=params["fixed_charge_mol_L"],
        feed_side_donnan_V=feed_side_donnan_V,
        permeate_side_donnan_V=permeate_side_donnan_V,
        pore_dielectric_constant=params["pore_dielectric_constant"],
        temperature_K=case["temperature_K"],
        bulk_dielectric_constant=bulk_dielectric_constant,
    )
