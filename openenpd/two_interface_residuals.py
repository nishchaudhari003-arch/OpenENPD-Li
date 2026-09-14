"""
Two-interface ENP-Donnan residual functions for OpenENPD-Li.

Milestone 2 of the two-interface roadmap. This module builds the **residual
equations** of the algebraic two-interface ENP-Donnan system on top of the
Milestone-1 state assembler (:mod:`openenpd.two_interface`). It constructs a
transparent, testable residual vector and its diagnostics.

It does **NOT** solve anything. There is no call to ``scipy.optimize.root`` or
``least_squares`` anywhere in this module, no parameter fitting, and no change
to the simplified baseline solver (:mod:`openenpd.solver`). The residuals are
the quantities that a later minimal nonlinear solver (Milestone 3) will drive
to zero; here they are only *evaluated* for provided inputs.

Residual concepts (following ``docs/two_interface_enp_design.md`` §5)
---------------------------------------------------------------------
For each ion ``i`` with charge ``z_i``, feed-side membrane concentration
``C_m0,i``, permeate-side membrane concentration ``C_mL,i``, bulk permeate
concentration ``C_p,i``, fixed membrane charge ``X`` and water flux ``J_v``:

1. Feed-side membrane electroneutrality:  ``r_EN,f  = Σ_i z_i C_m0,i + X``
2. Permeate-side membrane electroneutrality: ``r_EN,p = Σ_i z_i C_mL,i + X``
3. Permeate (free-solution) electroneutrality: ``r_EN,perm = Σ_i z_i C_p,i``
4. Zero-current condition:  ``r_I = Σ_i z_i J_i^ENP``
5. ENP flux closure per ion: ``r_J,i = J_i^ENP - C_p,i J_v``

The membrane ENP flux uses the design-document constant-gradient / mean-
concentration approximation (§5.4), evaluated between the two membrane-side
interface concentrations with a *provided* membrane potential drop ``Δψ_m``::

    J_i^ENP = -K_d,i D_i (C_mL,i - C_m0,i)/L
              - K_d,i D_i (z_i F / RT) C̄_m,i (Δψ_m / L)
              + K_c,i C̄_m,i J_v

where ``C̄_m,i = (C_m0,i + C_mL,i)/2``, ``L`` is the active-layer thickness,
``D_i`` the ion diffusivity, and ``K_d,i`` / ``K_c,i`` the existing diffusive
and convective hindrance factors. This reuses :func:`openenpd.enp.enp_flux`
(the existing ENP utility) and :func:`openenpd.transport.linear_concentration_gradient`;
the constant-gradient approximation is explicit and inherited from the current
baseline, not a hidden shortcut.

Explicit scope, units, and non-claims
-------------------------------------
- **Δψ_m is a provided input, not solved.** It is a genuine unknown of the
  full system; here the caller supplies it. The zero-current residual is
  therefore generally nonzero and is exactly the quantity a solver would use
  to determine Δψ_m.
- **Mixed units, raw residuals.** Electroneutrality residuals are in
  mol/L (charge-equivalent), matching the Milestone-1 assembler; the
  zero-current and flux-closure residuals are in mol m⁻² s⁻¹. The assembled
  vector is therefore *unscaled* and mixes units. Dimensionless residual
  scaling for a well-conditioned solve is a Milestone-3 concern (design doc
  §6.1) and is intentionally not done here. :attr:`TwoInterfaceResiduals.l2_norm`
  is a finiteness/magnitude diagnostic over the raw vector, not a physically
  scaled convergence metric.
- **Permeate electroneutrality is a dependent diagnostic.** For nonzero
  ``J_v`` it is algebraically implied by the flux-closure equations plus the
  zero-current condition (design doc §5.7). It is included in the reported
  residual vector for transparency, but a *square* ``root`` formulation in
  Milestone 3 should drop it to avoid over-determination; the active-ion set
  needed to build that square subset is exposed via
  :attr:`TwoInterfaceResiduals.active_ions`.
- **Hard-cutoff species are surfaced, not eliminated here.** A fully
  size-excluded species (steric factor 0, e.g. Mg²⁺ at the Foo-2023 pore
  radius) has zero membrane concentrations and a zero ENP flux, so its
  flux-closure residual reduces to ``-C_p,i J_v``. Residual construction does
  not crash on such species; they are reported in
  :attr:`TwoInterfaceResiduals.hard_cutoff_species` with a diagnostic message.
  Eliminating them from the active nonlinear vector is a Milestone-3 solver
  step (design doc §4).
- Nothing here claims the two-interface model is validated or that these
  residuals solve the problem. They are construction and diagnostics only.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Tuple

from openenpd.electroneutrality import charge_balance
from openenpd.enp import enp_flux
from openenpd.hindrance import hindrance_factors
from openenpd.transport import (
    active_layer_thickness_m,
    convert_concentrations_mol_L_to_mol_m3,
    linear_concentration_gradient,
)
from openenpd.two_interface import TwoInterfaceState

__all__ = [
    "TwoInterfaceTransportInputs",
    "TwoInterfaceResiduals",
    "two_interface_transport_inputs_from_case",
    "feed_side_electroneutrality_residual",
    "permeate_side_electroneutrality_residual",
    "permeate_electroneutrality_residual",
    "enp_ion_fluxes",
    "convective_permeate_fluxes",
    "flux_closure_residuals",
    "zero_current_residual",
    "residual_labels",
    "residual_vector",
    "residual_norm",
    "compute_two_interface_residuals",
]


def _read_only(mapping):
    """Return an independent, read-only ``MappingProxyType`` view of a mapping."""
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class TwoInterfaceTransportInputs:
    """
    Immutable transport-layer inputs needed to evaluate the ENP residuals.

    These are the quantities the membrane transport equation requires that the
    :class:`~openenpd.two_interface.TwoInterfaceState` does not itself carry.
    Ion-keyed mappings are defensively copied into read-only views, so passing
    a validation-case-derived dictionary never mutates it.

    Parameters
    ----------
    diffusivities_m2_s : Mapping[str, float]
        Ion diffusivities in m^2/s.
    diffusive_hindrance : Mapping[str, float]
        Diffusive hindrance factors K_d,i (dimensionless).
    convective_hindrance : Mapping[str, float]
        Convective hindrance factors K_c,i (dimensionless).
    thickness_m : float
        Membrane active-layer thickness L in meters.
    water_flux_m_s : float
        Water (volume) flux J_v in m/s.
    membrane_potential_drop_V : float
        Membrane potential drop Δψ_m = ψ_m(L) - ψ_m(0) in volts. Provided, not
        solved; the zero-current residual is the equation that would fix it.
    temperature_K : float
        Temperature in kelvin.

    Raises
    ------
    ValueError
        If the ion sets of the transport mappings are not identical, if the
        thickness is not positive, or if the temperature is not positive.
    """

    diffusivities_m2_s: Mapping[str, float]
    diffusive_hindrance: Mapping[str, float]
    convective_hindrance: Mapping[str, float]
    thickness_m: float
    water_flux_m_s: float
    membrane_potential_drop_V: float
    temperature_K: float

    def __post_init__(self):
        object.__setattr__(
            self, "diffusivities_m2_s", _read_only(self.diffusivities_m2_s)
        )
        object.__setattr__(
            self, "diffusive_hindrance", _read_only(self.diffusive_hindrance)
        )
        object.__setattr__(
            self, "convective_hindrance", _read_only(self.convective_hindrance)
        )
        self._validate()

    def _validate(self):
        ions = set(self.diffusivities_m2_s)
        for label, mapping in (
            ("diffusive_hindrance", self.diffusive_hindrance),
            ("convective_hindrance", self.convective_hindrance),
        ):
            if set(mapping) != ions:
                raise ValueError(
                    "Inconsistent ion sets: "
                    f"{label} keys {sorted(mapping)} do not match diffusivity "
                    f"ions {sorted(ions)}."
                )
        if self.thickness_m <= 0.0:
            raise ValueError("thickness_m must be positive.")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")


def two_interface_transport_inputs_from_case(
    case,
    water_flux_m_s,
    membrane_potential_drop_V,
    temperature_K=None,
) -> TwoInterfaceTransportInputs:
    """
    Build :class:`TwoInterfaceTransportInputs` from a validation-case dict.

    This helper only *reads* from ``case`` (and copies what it reads), so the
    case dictionary is never mutated. Diffusivities come straight from the
    case; hindrance factors are computed with the existing
    :func:`openenpd.hindrance.hindrance_factors`; the thickness comes from
    :func:`openenpd.transport.active_layer_thickness_m`.

    Parameters
    ----------
    case : dict
        Validation case definition, such as ``foo2023_lmc_ph7_case()``.
    water_flux_m_s : float
        Water flux J_v in m/s (e.g. one experimental value).
    membrane_potential_drop_V : float
        Provided membrane potential drop Δψ_m in volts.
    temperature_K : float, optional
        Temperature in kelvin. Defaults to ``case["temperature_K"]``.

    Returns
    -------
    TwoInterfaceTransportInputs
        Transport inputs ready for the residual functions.
    """
    params = case["membrane_parameters"]
    hindrance = hindrance_factors(
        ion_radii_nm=case["ion_radii_nm"],
        pore_radius_nm=params["pore_radius_nm"],
    )

    return TwoInterfaceTransportInputs(
        diffusivities_m2_s=case["diffusivities_m2_s"],
        diffusive_hindrance=hindrance["diffusive"],
        convective_hindrance=hindrance["convective"],
        thickness_m=active_layer_thickness_m(case),
        water_flux_m_s=water_flux_m_s,
        membrane_potential_drop_V=membrane_potential_drop_V,
        temperature_K=temperature_K if temperature_K is not None else case["temperature_K"],
    )


def _require_ion_coverage(state: TwoInterfaceState, transport: TwoInterfaceTransportInputs):
    """Raise a clear error if the transport inputs do not cover every state ion."""
    missing = [
        ion for ion in state.ions if ion not in transport.diffusivities_m2_s
    ]
    if missing:
        raise ValueError(
            "Transport inputs are missing entries for ions "
            f"{missing}; required by the two-interface state."
        )


# ---------------------------------------------------------------------------
# 1-3. Electroneutrality residuals (mol/L, charge-equivalent)
# ---------------------------------------------------------------------------

def feed_side_electroneutrality_residual(state: TwoInterfaceState) -> float:
    """
    Feed-side membrane electroneutrality residual ``Σ_i z_i C_m0,i + X``.

    Reuses :func:`openenpd.electroneutrality.charge_balance` on the feed-side
    membrane concentrations and adds the fixed membrane charge. Equal (by
    construction) to ``state.feed_side_charge_residual_mol_L``.

    Returns
    -------
    float
        Residual in mol/L charge-equivalent units. Zero means the mobile
        charge exactly balances the fixed membrane charge at the feed side.
    """
    return (
        charge_balance(state.feed_membrane_concentrations_mol_L, state.charges)
        + state.fixed_charge_mol_L
    )


def permeate_side_electroneutrality_residual(state: TwoInterfaceState) -> float:
    """
    Permeate-side membrane electroneutrality residual ``Σ_i z_i C_mL,i + X``.

    Returns
    -------
    float
        Residual in mol/L charge-equivalent units. Equal (by construction) to
        ``state.permeate_side_charge_residual_mol_L``.
    """
    return (
        charge_balance(state.permeate_membrane_concentrations_mol_L, state.charges)
        + state.fixed_charge_mol_L
    )


def permeate_electroneutrality_residual(state: TwoInterfaceState) -> float:
    """
    Permeate free-solution electroneutrality residual ``Σ_i z_i C_p,i``.

    There is no fixed charge in the free permeate solution. This residual is a
    *dependent diagnostic*: for nonzero J_v it is algebraically implied by the
    flux-closure and zero-current residuals (design doc §5.7).

    Returns
    -------
    float
        Residual in mol/L charge-equivalent units.
    """
    return charge_balance(state.permeate_concentrations_mol_L, state.charges)


# ---------------------------------------------------------------------------
# 4-5. Membrane ENP fluxes, flux closure, and zero current (mol m^-2 s^-1)
# ---------------------------------------------------------------------------

def enp_ion_fluxes(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> dict:
    """
    Membrane ENP fluxes for every ion, in mol m^-2 s^-1.

    Uses the constant-gradient / mean-concentration approximation between the
    two membrane-side interface concentrations, with the *provided* membrane
    potential drop (see module docstring). Concentrations are converted from
    mol/L (the assembler's unit) to mol/m^3 with the existing transport helper
    before entering :func:`openenpd.enp.enp_flux`.

    Hard-cutoff species have zero membrane concentrations and zero hindrance,
    so their ENP flux is 0.0; the calculation does not fail.

    Parameters
    ----------
    state : TwoInterfaceState
        Assembled two-interface state.
    transport : TwoInterfaceTransportInputs
        Transport-layer inputs.

    Returns
    -------
    dict
        Ion -> ENP molar flux in mol m^-2 s^-1.
    """
    _require_ion_coverage(state, transport)

    feed_membrane_mol_m3 = convert_concentrations_mol_L_to_mol_m3(
        state.feed_membrane_concentrations_mol_L
    )
    permeate_membrane_mol_m3 = convert_concentrations_mol_L_to_mol_m3(
        state.permeate_membrane_concentrations_mol_L
    )

    thickness_m = transport.thickness_m
    potential_gradient_V_m = transport.membrane_potential_drop_V / thickness_m

    fluxes = {}
    for ion in state.ions:
        upstream = feed_membrane_mol_m3[ion]
        downstream = permeate_membrane_mol_m3[ion]

        gradient = linear_concentration_gradient(
            upstream_concentration_mol_m3=upstream,
            downstream_concentration_mol_m3=downstream,
            thickness_m=thickness_m,
        )
        mean_concentration = 0.5 * (upstream + downstream)

        fluxes[ion] = enp_flux(
            diffusivity_m2_s=transport.diffusivities_m2_s[ion],
            charge=state.charges[ion],
            concentration_mol_m3=mean_concentration,
            concentration_gradient_mol_m4=gradient,
            potential_gradient_V_m=potential_gradient_V_m,
            water_flux_m_s=transport.water_flux_m_s,
            temperature_K=transport.temperature_K,
            diffusive_hindrance=transport.diffusive_hindrance[ion],
            convective_hindrance=transport.convective_hindrance[ion],
        )

    return fluxes


def convective_permeate_fluxes(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> dict:
    """
    Permeate convective removal flux ``C_p,i * J_v`` per ion, in mol m^-2 s^-1.

    Returns
    -------
    dict
        Ion -> convective flux in mol m^-2 s^-1.
    """
    permeate_mol_m3 = convert_concentrations_mol_L_to_mol_m3(
        state.permeate_concentrations_mol_L
    )
    return {
        ion: permeate_mol_m3[ion] * transport.water_flux_m_s
        for ion in state.ions
    }


def flux_closure_residuals(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> dict:
    """
    Species ENP flux-closure residuals ``J_i^ENP - C_p,i * J_v``.

    Returns
    -------
    dict
        Ion -> residual in mol m^-2 s^-1. Keys are exactly the state ions.
    """
    enp = enp_ion_fluxes(state, transport)
    convective = convective_permeate_fluxes(state, transport)
    return {ion: enp[ion] - convective[ion] for ion in state.ions}


def zero_current_residual(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> float:
    """
    Zero-current residual ``Σ_i z_i J_i^ENP``, in mol m^-2 s^-1.

    For an electrically open membrane this is zero at the physical solution;
    with a provided (non-solved) membrane potential drop it is generally
    nonzero and is the equation a solver uses to fix Δψ_m.

    Returns
    -------
    float
        Charge-weighted flux sum in mol m^-2 s^-1.
    """
    enp = enp_ion_fluxes(state, transport)
    return sum(state.charges[ion] * enp[ion] for ion in state.ions)


# ---------------------------------------------------------------------------
# 6. Residual vector assembly + 7. diagnostics
# ---------------------------------------------------------------------------

def residual_labels(state: TwoInterfaceState) -> Tuple[str, ...]:
    """
    Ordered labels aligned with :func:`residual_vector`.

    The order is: feed-side electroneutrality, permeate-side electroneutrality,
    permeate electroneutrality, zero current, then one flux-closure residual
    per ion in state order.

    Returns
    -------
    tuple of str
        Residual names, length ``4 + len(state.ions)``.
    """
    return (
        "feed_side_electroneutrality",
        "permeate_side_electroneutrality",
        "permeate_electroneutrality",
        "zero_current",
        *(f"flux_closure[{ion}]" for ion in state.ions),
    )


def residual_vector(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> Tuple[float, ...]:
    """
    Assemble the deterministic two-interface residual vector.

    The vector, in order, is::

        [ r_EN,f, r_EN,p, r_EN,perm, r_I, r_J,i for each ion ]

    Its length is ``4 + len(state.ions)`` and is fully determined by the
    inputs. This is a *raw, unscaled* vector that mixes units (mol/L for the
    electroneutrality entries, mol m^-2 s^-1 for zero current and flux
    closure) -- see the module docstring. No optimizer is called.

    Returns
    -------
    tuple of float
        The residual vector.
    """
    enp = enp_ion_fluxes(state, transport)
    convective = convective_permeate_fluxes(state, transport)

    feed_en = feed_side_electroneutrality_residual(state)
    permeate_side_en = permeate_side_electroneutrality_residual(state)
    permeate_en = permeate_electroneutrality_residual(state)
    zero_current = sum(state.charges[ion] * enp[ion] for ion in state.ions)
    flux_closure = [enp[ion] - convective[ion] for ion in state.ions]

    return (feed_en, permeate_side_en, permeate_en, zero_current, *flux_closure)


def residual_norm(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> float:
    """
    Euclidean (L2) norm of :func:`residual_vector`.

    A magnitude/finiteness diagnostic over the raw, mixed-unit vector -- not a
    physically scaled convergence metric (scaling is a Milestone-3 concern).

    Returns
    -------
    float
        L2 norm of the residual vector.
    """
    return math.sqrt(sum(value * value for value in residual_vector(state, transport)))


@dataclass(frozen=True)
class TwoInterfaceResiduals:
    """
    Assembled two-interface residuals and diagnostics (no solving performed).

    Attributes
    ----------
    ions : tuple of str
        Ordered ion labels.
    active_ions : tuple of str
        Ions that are not hard-cutoff. This is the set a Milestone-3 square
        ``root`` formulation would keep in the active nonlinear vector.
    feed_side_electroneutrality_mol_L : float
        ``Σ_i z_i C_m0,i + X``.
    permeate_side_electroneutrality_mol_L : float
        ``Σ_i z_i C_mL,i + X``.
    permeate_electroneutrality_mol_L : float
        ``Σ_i z_i C_p,i`` (dependent diagnostic).
    zero_current_mol_m2_s : float
        ``Σ_i z_i J_i^ENP``.
    flux_residuals_mol_m2_s : Mapping[str, float]
        Per-ion ``J_i^ENP - C_p,i J_v``.
    enp_fluxes_mol_m2_s : Mapping[str, float]
        Per-ion membrane ENP flux (diagnostic).
    convective_permeate_fluxes_mol_m2_s : Mapping[str, float]
        Per-ion ``C_p,i J_v`` (diagnostic).
    residual_vector : tuple of float
        Raw residual vector; see :func:`residual_vector`.
    residual_labels : tuple of str
        Names aligned with ``residual_vector``.
    l2_norm : float
        Euclidean norm of the raw vector (finiteness/magnitude diagnostic).
    max_abs_residual : float
        Largest absolute entry of the raw vector.
    is_finite : bool
        True iff every residual entry is finite.
    hard_cutoff_species : tuple of str
        Fully size-excluded ions (from the state), surfaced explicitly.
    messages : tuple of str
        Human-readable diagnostics/warnings (hard cutoff, non-finiteness,
        negative concentrations, dependent permeate-EN entry).
    """

    ions: Tuple[str, ...]
    active_ions: Tuple[str, ...]
    feed_side_electroneutrality_mol_L: float
    permeate_side_electroneutrality_mol_L: float
    permeate_electroneutrality_mol_L: float
    zero_current_mol_m2_s: float
    flux_residuals_mol_m2_s: Mapping[str, float]
    enp_fluxes_mol_m2_s: Mapping[str, float]
    convective_permeate_fluxes_mol_m2_s: Mapping[str, float]
    residual_vector: Tuple[float, ...]
    residual_labels: Tuple[str, ...]
    l2_norm: float
    max_abs_residual: float
    is_finite: bool
    hard_cutoff_species: Tuple[str, ...] = field(default_factory=tuple)
    messages: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def named_residuals(self) -> Mapping[str, float]:
        """Read-only mapping of scalar residual name -> value (flux entries included)."""
        named = {
            "feed_side_electroneutrality": self.feed_side_electroneutrality_mol_L,
            "permeate_side_electroneutrality": self.permeate_side_electroneutrality_mol_L,
            "permeate_electroneutrality": self.permeate_electroneutrality_mol_L,
            "zero_current": self.zero_current_mol_m2_s,
        }
        for ion, value in self.flux_residuals_mol_m2_s.items():
            named[f"flux_closure[{ion}]"] = value
        return MappingProxyType(named)


def _negative_concentration_ions(state: TwoInterfaceState) -> Tuple[str, ...]:
    """Return ions with a negative concentration in any of the state's compartments."""
    flagged = []
    compartments = (
        state.feed_bulk_concentrations_mol_L,
        state.permeate_concentrations_mol_L,
        state.feed_membrane_concentrations_mol_L,
        state.permeate_membrane_concentrations_mol_L,
    )
    for ion in state.ions:
        if any(compartment[ion] < 0.0 for compartment in compartments):
            flagged.append(ion)
    return tuple(flagged)


def compute_two_interface_residuals(
    state: TwoInterfaceState,
    transport: TwoInterfaceTransportInputs,
) -> TwoInterfaceResiduals:
    """
    Compute all two-interface residuals and diagnostics for provided inputs.

    Deterministic and optimizer-free. Neither ``state`` nor ``transport`` is
    mutated (both hold read-only mappings).

    Parameters
    ----------
    state : TwoInterfaceState
        Assembled two-interface state (Milestone 1).
    transport : TwoInterfaceTransportInputs
        Transport-layer inputs.

    Returns
    -------
    TwoInterfaceResiduals
        Named residuals, the raw residual vector and its labels, the L2 norm
        and max absolute residual, a finiteness flag, the hard-cutoff species,
        and diagnostic messages.
    """
    _require_ion_coverage(state, transport)

    enp = enp_ion_fluxes(state, transport)
    convective = convective_permeate_fluxes(state, transport)

    feed_en = feed_side_electroneutrality_residual(state)
    permeate_side_en = permeate_side_electroneutrality_residual(state)
    permeate_en = permeate_electroneutrality_residual(state)
    zero_current = sum(state.charges[ion] * enp[ion] for ion in state.ions)
    flux_residuals = {ion: enp[ion] - convective[ion] for ion in state.ions}

    vector = (
        feed_en,
        permeate_side_en,
        permeate_en,
        zero_current,
        *(flux_residuals[ion] for ion in state.ions),
    )
    labels = residual_labels(state)

    is_finite = all(math.isfinite(value) for value in vector)
    l2_norm = (
        math.sqrt(sum(value * value for value in vector)) if is_finite else math.inf
    )
    max_abs_residual = (
        max(abs(value) for value in vector) if is_finite else math.inf
    )

    active_ions = tuple(
        ion for ion in state.ions if ion not in state.hard_cutoff_species
    )

    messages = []
    if state.hard_cutoff_species:
        messages.append(
            "Hard-cutoff species present ("
            + ", ".join(state.hard_cutoff_species)
            + "): zero membrane concentration and zero ENP flux; flux-closure "
            "residual reduces to -C_p*J_v. A Milestone-3 solver should "
            "eliminate these from the active nonlinear vector."
        )
    negative_ions = _negative_concentration_ions(state)
    if negative_ions:
        messages.append(
            "Negative concentration(s) present for ("
            + ", ".join(negative_ions)
            + "): residuals were still computed but the physical state is "
            "unphysical; a solver must reject or transform these."
        )
    if not is_finite:
        messages.append(
            "Non-finite residual encountered; l2_norm and max_abs_residual set "
            "to inf."
        )
    messages.append(
        "permeate_electroneutrality is a dependent diagnostic (implied by flux "
        "closure + zero current for nonzero J_v); exclude it from a square "
        "root formulation."
    )

    return TwoInterfaceResiduals(
        ions=state.ions,
        active_ions=active_ions,
        feed_side_electroneutrality_mol_L=feed_en,
        permeate_side_electroneutrality_mol_L=permeate_side_en,
        permeate_electroneutrality_mol_L=permeate_en,
        zero_current_mol_m2_s=zero_current,
        flux_residuals_mol_m2_s=_read_only(flux_residuals),
        enp_fluxes_mol_m2_s=_read_only(enp),
        convective_permeate_fluxes_mol_m2_s=_read_only(convective),
        residual_vector=tuple(vector),
        residual_labels=labels,
        l2_norm=l2_norm,
        max_abs_residual=max_abs_residual,
        is_finite=is_finite,
        hard_cutoff_species=state.hard_cutoff_species,
        messages=tuple(messages),
    )
