"""
Minimal nonlinear two-interface ENP-Donnan solver for OpenENPD-Li.

Milestone 3 of the two-interface roadmap. This module wraps the Milestone-1
state assembler (:mod:`openenpd.two_interface`) and the Milestone-2 residual
functions (:mod:`openenpd.two_interface_residuals`) in a minimal nonlinear
solver that finds the **state variables** of the two-interface algebraic
system for one prescribed water flux.

Scope and non-claims
--------------------
- This solves **state variables**, not physical/model parameters. Pore radius,
  fixed charge, dielectric constant, ion radii, diffusivities, and hindrance
  factors are held at their case values; nothing here is fitted or tuned.
- It does **not** change the simplified baseline solver
  (:mod:`openenpd.solver`) and does not import it.
- Convergence on the Foo 2023 LM-C pH 7 case is **not** assumed. This milestone
  succeeds if the infrastructure works, controlled cases converge, failures are
  structured and honest, and the Foo wrappers run without crashing. No claim is
  made that the model is validated or reproduces negative Li rejection.

Unknown vector (one water flux J_v)
-----------------------------------
For the ``N`` *active* ions (all modeled ions minus fully size-excluded
hard-cutoff species, see below), the unknown vector is::

    y = [ C_p,i (mol/L) for each active ion,
          feed_side_donnan_V,
          permeate_side_donnan_V,
          membrane_potential_drop_V ]

Length ``N + 3``. Permeate concentrations are kept in mol/L. Positivity is
enforced by **direct bounds** on ``least_squares`` (lower bound
``concentration_lower_bound_mol_L``, upper bound
``concentration_upper_factor * C_f,i``) rather than a log transform: it is the
simplest option that guarantees non-negative concentrations, is natively
supported by ``least_squares``, and avoids Donnan-exponential overflow. The
three potentials are bounded to ``[-potential_bounds_V, +potential_bounds_V]``.

Residual subset solved (scaled)
-------------------------------
Following ``docs/two_interface_enp_design.md`` §5.8, the *core* solve vector is
the square-ish set (excluding the dependent permeate free-solution
electroneutrality diagnostic)::

    [ r_EN,f, r_EN,p, r_I, r_J,i for each active ion ]

Optionally, permeate free-solution electroneutrality can be appended with
``include_permeate_electroneutrality_in_solve=True`` (documented as a dependent
diagnostic; it over-determines the system for nonzero J_v).

Why ``least_squares`` and why scaling
-------------------------------------
``scipy.optimize.least_squares`` is used (not ``root``) because it supports
bounds (hence concentration positivity), handles non-square residual sets
(useful if the permeate-EN diagnostic is included), and is more robust for an
early controlled implementation. The residuals mix units (mol/L for
electroneutrality; mol m^-2 s^-1 for zero current and flux closure), so a raw
L2 norm is not a safe convergence target. An explicit scaling layer divides the
electroneutrality residuals by a concentration scale and the zero-current /
flux residuals by a molar-flux scale; **success is judged on the scaled
residuals**, never on the raw mixed-unit norm.

Hard-cutoff handling
--------------------
Fully size-excluded species (steric factor 0, e.g. Mg2+ at the Foo 2023 pore
radius) are eliminated from the unknown vector: their permeate concentration is
fixed to 0, so their membrane concentrations and ENP flux are 0 and their flux
closure is trivially satisfied. They are surfaced in the result. If *every*
ion is hard-cutoff there are no active unknowns and the solver returns a
structured failure rather than crashing or pretending success.
"""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from scipy.optimize import least_squares

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.hindrance import hindrance_factors
from openenpd.rejection import species_rejections
from openenpd.transport import active_layer_thickness_m
from openenpd.two_interface import (
    assemble_two_interface_state,
    two_interface_inputs_from_case,
)
from openenpd.two_interface_residuals import (
    TwoInterfaceTransportInputs,
    compute_two_interface_residuals,
    feed_side_electroneutrality_residual,
    flux_closure_residuals,
    permeate_electroneutrality_residual,
    permeate_side_electroneutrality_residual,
    zero_current_residual,
)

__all__ = [
    "TwoInterfaceSolverOptions",
    "TwoInterfaceSolveResult",
    "ResidualScales",
    "unknown_labels",
    "pack_unknowns",
    "unpack_unknowns",
    "compute_residual_scales",
    "solve_two_interface_for_flux",
    "solve_foo2023_lmc_ph7_flux",
    "solve_foo2023_lmc_ph7_all_fluxes",
]


def _read_only(mapping):
    """Return an independent, read-only ``MappingProxyType`` view of a mapping."""
    return MappingProxyType(dict(mapping))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoInterfaceSolverOptions:
    """
    Options for the minimal two-interface solver.

    Parameters
    ----------
    max_nfev : int
        Maximum number of residual evaluations passed to ``least_squares``.
    tolerance : float
        Success threshold on the maximum absolute *scaled* residual. A solve is
        reported successful only if ``least_squares`` reports success AND the
        max absolute scaled residual is at or below this value.
    potential_bounds_V : float
        Symmetric bound (in volts) for each of the three potential unknowns:
        each is constrained to ``[-potential_bounds_V, +potential_bounds_V]``.
    concentration_lower_bound_mol_L : float
        Lower bound for every permeate concentration unknown (mol/L). Must be
        non-negative; keeps concentrations strictly bounded away from negatives.
    concentration_upper_factor : float
        Upper bound for each permeate concentration unknown, as a multiple of
        that ion's feed concentration. Must exceed 1.
    include_permeate_electroneutrality_in_solve : bool
        If True, append the (dependent) permeate free-solution electroneutrality
        residual to the scaled solve vector. Off by default; see module doc.
    flux_scale_floor_mol_m2_s : float
        Floor for the molar-flux scale, so flux residuals stay well-scaled even
        when ``J_v`` is small or zero. Must be positive.
    verbose : bool
        If True, request ``least_squares`` verbosity (diagnostic only).

    Raises
    ------
    ValueError
        If any option is outside its valid range.
    """

    max_nfev: int = 2000
    tolerance: float = 1e-6
    potential_bounds_V: float = 0.2
    concentration_lower_bound_mol_L: float = 1e-9
    concentration_upper_factor: float = 10.0
    include_permeate_electroneutrality_in_solve: bool = False
    flux_scale_floor_mol_m2_s: float = 1e-12
    verbose: bool = False

    def __post_init__(self):
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer.")
        if not (self.tolerance > 0.0):
            raise ValueError("tolerance must be positive.")
        if not (self.potential_bounds_V > 0.0):
            raise ValueError("potential_bounds_V must be positive.")
        if self.concentration_lower_bound_mol_L < 0.0:
            raise ValueError("concentration_lower_bound_mol_L must be non-negative.")
        if not (self.concentration_upper_factor > 1.0):
            raise ValueError("concentration_upper_factor must exceed 1.")
        if not (self.flux_scale_floor_mol_m2_s > 0.0):
            raise ValueError("flux_scale_floor_mol_m2_s must be positive.")


# ---------------------------------------------------------------------------
# Residual scales
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidualScales:
    """
    Representative scales used to non-dimensionalize the solve residuals.

    Attributes
    ----------
    concentration_scale_mol_L : float
        Scale for electroneutrality residuals (mol/L).
    flux_scale_mol_m2_s : float
        Scale for zero-current and flux-closure residuals (mol m^-2 s^-1).
    """

    concentration_scale_mol_L: float
    flux_scale_mol_m2_s: float


def compute_residual_scales(
    feed_concentrations_mol_L: Mapping[str, float],
    water_flux_m_s: float,
    options: TwoInterfaceSolverOptions,
) -> ResidualScales:
    """
    Compute representative concentration and molar-flux scales.

    The concentration scale is the total feed ionic concentration
    ``Σ_i C_f,i`` (mol/L). The molar-flux scale is
    ``max(C_ref[mol/m^3] * |J_v|, flux_scale_floor)``, i.e. a convective flux
    scale with a positive floor so it is never zero.

    Returns
    -------
    ResidualScales
        The computed scales.
    """
    concentration_scale_mol_L = sum(feed_concentrations_mol_L.values())
    if not (concentration_scale_mol_L > 0.0):
        raise ValueError("Total feed concentration must be positive for scaling.")

    concentration_scale_mol_m3 = concentration_scale_mol_L * 1000.0
    flux_scale = max(
        concentration_scale_mol_m3 * abs(water_flux_m_s),
        options.flux_scale_floor_mol_m2_s,
    )
    return ResidualScales(
        concentration_scale_mol_L=concentration_scale_mol_L,
        flux_scale_mol_m2_s=flux_scale,
    )


# ---------------------------------------------------------------------------
# Unknown-vector packing / unpacking
# ---------------------------------------------------------------------------

def unknown_labels(active_ions: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Ordered labels for the unknown vector.

    Returns
    -------
    tuple of str
        ``[C_p[ion] for each active ion, feed_donnan_V, permeate_donnan_V,
        membrane_potential_drop_V]``.
    """
    return (
        *(f"C_p[{ion}]" for ion in active_ions),
        "feed_donnan_V",
        "permeate_donnan_V",
        "membrane_potential_drop_V",
    )


def pack_unknowns(
    permeate_active_mol_L: Mapping[str, float],
    feed_donnan_V: float,
    permeate_donnan_V: float,
    membrane_potential_drop_V: float,
    active_ions: Tuple[str, ...],
) -> list:
    """
    Pack state variables into the flat unknown vector (see :func:`unknown_labels`).

    Returns
    -------
    list of float
        The unknown vector, length ``len(active_ions) + 3``.
    """
    return [permeate_active_mol_L[ion] for ion in active_ions] + [
        feed_donnan_V,
        permeate_donnan_V,
        membrane_potential_drop_V,
    ]


def unpack_unknowns(x, active_ions: Tuple[str, ...]):
    """
    Unpack the flat unknown vector into named state variables.

    Returns
    -------
    tuple
        ``(permeate_active_mol_L, feed_donnan_V, permeate_donnan_V,
        membrane_potential_drop_V)`` where the first item is a dict.
    """
    n = len(active_ions)
    if len(x) != n + 3:
        raise ValueError(
            f"Unknown vector has length {len(x)}, expected {n + 3} for "
            f"{n} active ions."
        )
    permeate = {ion: float(x[i]) for i, ion in enumerate(active_ions)}
    return permeate, float(x[n]), float(x[n + 1]), float(x[n + 2])


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoInterfaceSolveResult:
    """
    Structured result of a single-flux two-interface solve.

    Attributes
    ----------
    success : bool
        True only if the optimizer reported success AND the maximum absolute
        scaled residual is at or below the tolerance.
    message : str
        Human-readable status/diagnostic message.
    water_flux_m_s : float
        The water flux this result is for.
    active_ions : tuple of str
        Ions carried as unknowns (all modeled ions minus hard cutoff).
    hard_cutoff_species : tuple of str
        Fully size-excluded ions (permeate concentration fixed to 0).
    permeate_concentrations_mol_L : Mapping[str, float]
        Predicted permeate concentrations for every modeled ion (hard-cutoff
        ions are 0).
    feed_donnan_potential_V, permeate_donnan_potential_V : float
        Solved (or attempted) Donnan potentials in volts.
    membrane_potential_drop_V : float
        Solved (or attempted) membrane potential drop in volts.
    predicted_rejections : Mapping[str, float]
        ``R_i = 1 - C_p,i / C_f,i`` for every modeled ion.
    residual_norm_scaled : float
        L2 norm of the scaled solve residual vector.
    residual_max_abs_scaled : float
        Maximum absolute scaled residual (the success metric).
    residual_norm_raw : float
        L2 norm of the full raw residual vector (all residuals, mol-mixed).
    raw_residuals : Mapping[str, float]
        Named full raw residuals at the solution (includes the permeate-EN
        diagnostic), for auditing.
    scaled_residual_vector : tuple of float
        The scaled solve residual vector actually minimized.
    scaled_residual_labels : tuple of str
        Labels aligned with ``scaled_residual_vector``.
    optimizer_status : int or None
        ``least_squares`` status code (None if the optimizer was not run).
    optimizer_nfev : int or None
        Number of residual evaluations (None if the optimizer was not run).
    """

    success: bool
    message: str
    water_flux_m_s: float
    active_ions: Tuple[str, ...]
    hard_cutoff_species: Tuple[str, ...]
    permeate_concentrations_mol_L: Mapping[str, float]
    feed_donnan_potential_V: float
    permeate_donnan_potential_V: float
    membrane_potential_drop_V: float
    predicted_rejections: Mapping[str, float]
    residual_norm_scaled: float
    residual_max_abs_scaled: float
    residual_norm_raw: float
    raw_residuals: Mapping[str, float]
    scaled_residual_vector: Tuple[float, ...]
    scaled_residual_labels: Tuple[str, ...]
    optimizer_status: Optional[int] = None
    optimizer_nfev: Optional[int] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _determine_ions_and_cutoff(case):
    """
    Probe the assembler (zero potentials, feed as placeholder permeate) to get
    the ordered ion set and the hard-cutoff species. Hard-cutoff depends only
    on radii/pore, so this probe is representative regardless of composition.
    """
    probe_inputs = two_interface_inputs_from_case(
        case,
        permeate_concentrations_mol_L=dict(case["bulk_concentrations_mol_L"]),
        feed_side_donnan_V=0.0,
        permeate_side_donnan_V=0.0,
    )
    probe_state = assemble_two_interface_state(probe_inputs)
    return probe_state.ions, probe_state.hard_cutoff_species


def _build_state_and_transport(
    case,
    permeate_full_mol_L,
    feed_donnan_V,
    permeate_donnan_V,
    membrane_potential_drop_V,
    water_flux_m_s,
    diffusivities,
    diffusive_hindrance,
    convective_hindrance,
    thickness_m,
    temperature_K,
    variant=None,
):
    """
    Assemble a state and matching transport inputs for one trial point.

    When ``variant`` is ``None`` this is the exact hard-baseline path (the
    original behaviour). When a :class:`~openenpd.constitutive_variants.ConstitutiveVariantConfig`
    is given, the state is assembled with the variant steric/radius treatment;
    the transport hindrance factors are passed in already computed for the
    variant by the caller.
    """
    if variant is None:
        inputs = two_interface_inputs_from_case(
            case,
            permeate_concentrations_mol_L=permeate_full_mol_L,
            feed_side_donnan_V=feed_donnan_V,
            permeate_side_donnan_V=permeate_donnan_V,
        )
        state = assemble_two_interface_state(inputs)
    else:
        # Lazy import to avoid an import cycle with constitutive_variants.
        from openenpd.constitutive_variants import build_two_interface_state_variant

        state = build_two_interface_state_variant(
            case,
            permeate_concentrations_mol_L=permeate_full_mol_L,
            feed_side_donnan_V=feed_donnan_V,
            permeate_side_donnan_V=permeate_donnan_V,
            config=variant,
        )
    transport = TwoInterfaceTransportInputs(
        diffusivities_m2_s=diffusivities,
        diffusive_hindrance=diffusive_hindrance,
        convective_hindrance=convective_hindrance,
        thickness_m=thickness_m,
        water_flux_m_s=water_flux_m_s,
        membrane_potential_drop_V=membrane_potential_drop_V,
        temperature_K=temperature_K,
    )
    return state, transport


def _scaled_solve_vector(state, transport, active_ions, scales, options):
    """
    Build the scaled solve residual vector and its labels.

    Order: feed-side EN, permeate-side EN, zero current, active-ion flux
    closures, then (optionally) permeate free-solution EN.
    """
    c_scale = scales.concentration_scale_mol_L
    j_scale = scales.flux_scale_mol_m2_s

    feed_en = feed_side_electroneutrality_residual(state)
    permeate_side_en = permeate_side_electroneutrality_residual(state)
    zero_current = zero_current_residual(state, transport)
    flux = flux_closure_residuals(state, transport)

    labels = ["feed_side_electroneutrality", "permeate_side_electroneutrality", "zero_current"]
    vector = [feed_en / c_scale, permeate_side_en / c_scale, zero_current / j_scale]

    for ion in active_ions:
        labels.append(f"flux_closure[{ion}]")
        vector.append(flux[ion] / j_scale)

    if options.include_permeate_electroneutrality_in_solve:
        labels.append("permeate_electroneutrality")
        vector.append(permeate_electroneutrality_residual(state) / c_scale)

    return vector, tuple(labels)


def _assemble_result(
    *,
    success,
    message,
    water_flux_m_s,
    active_ions,
    hard_cutoff_species,
    permeate_full_mol_L,
    feed_donnan_V,
    permeate_donnan_V,
    membrane_potential_drop_V,
    feed_concentrations_mol_L,
    scaled_vector,
    scaled_labels,
    raw_named,
    raw_norm,
    optimizer_status,
    optimizer_nfev,
):
    """Bundle a fully-populated, immutable solve result."""
    if scaled_vector:
        residual_norm_scaled = math.sqrt(sum(v * v for v in scaled_vector))
        residual_max_abs_scaled = max(abs(v) for v in scaled_vector)
    else:
        residual_norm_scaled = math.inf
        residual_max_abs_scaled = math.inf

    predicted_rejections = species_rejections(
        feed_concentrations=feed_concentrations_mol_L,
        permeate_concentrations=permeate_full_mol_L,
    )

    return TwoInterfaceSolveResult(
        success=success,
        message=message,
        water_flux_m_s=water_flux_m_s,
        active_ions=tuple(active_ions),
        hard_cutoff_species=tuple(hard_cutoff_species),
        permeate_concentrations_mol_L=_read_only(permeate_full_mol_L),
        feed_donnan_potential_V=feed_donnan_V,
        permeate_donnan_potential_V=permeate_donnan_V,
        membrane_potential_drop_V=membrane_potential_drop_V,
        predicted_rejections=_read_only(predicted_rejections),
        residual_norm_scaled=residual_norm_scaled,
        residual_max_abs_scaled=residual_max_abs_scaled,
        residual_norm_raw=raw_norm,
        raw_residuals=_read_only(raw_named),
        scaled_residual_vector=tuple(scaled_vector),
        scaled_residual_labels=tuple(scaled_labels),
        optimizer_status=optimizer_status,
        optimizer_nfev=optimizer_nfev,
    )


# ---------------------------------------------------------------------------
# Public solve entry points
# ---------------------------------------------------------------------------

def solve_two_interface_for_flux(
    case,
    water_flux_m_s,
    options: Optional[TwoInterfaceSolverOptions] = None,
    variant=None,
) -> TwoInterfaceSolveResult:
    """
    Solve the two-interface state variables for one water flux.

    Solves for the active-ion permeate concentrations and the three potentials
    that drive the scaled core residuals (feed/permeate-side membrane
    electroneutrality, zero current, active-ion flux closure) to zero. Does not
    fit any physical parameter and never mutates ``case``.

    Parameters
    ----------
    case : dict
        Validation-case definition (feed concentrations, charges, radii,
        diffusivities, temperature, membrane parameters).
    water_flux_m_s : float
        Prescribed water flux J_v in m/s.
    options : TwoInterfaceSolverOptions, optional
        Solver options. Defaults to ``TwoInterfaceSolverOptions()``.
    variant : ConstitutiveVariantConfig, optional
        Constitutive steric/hindrance variant. When ``None`` (the default) the
        solver uses the hard-baseline physics exactly as before. When given, the
        variant steric/hindrance/radius treatment is used to build the state and
        transport (diagnostic; no parameter fitting).

    Returns
    -------
    TwoInterfaceSolveResult
        Structured result. ``success`` is False (never an exception) for a
        non-converged solve, an all-hard-cutoff system, or a non-finite
        residual evaluation.
    """
    if options is None:
        options = TwoInterfaceSolverOptions()

    feed_concentrations_mol_L = dict(case["bulk_concentrations_mol_L"])
    params = case["membrane_parameters"]

    if variant is None:
        ions, hard_cutoff_species = _determine_ions_and_cutoff(case)
        hindrance = hindrance_factors(
            ion_radii_nm=case["ion_radii_nm"],
            pore_radius_nm=params["pore_radius_nm"],
        )
    else:
        # Lazy import to avoid an import cycle with constitutive_variants.
        from openenpd.constitutive_variants import (
            variant_hindrance_factors,
            variant_ions_and_cutoff,
        )

        ions, hard_cutoff_species = variant_ions_and_cutoff(case, variant)
        hindrance = variant_hindrance_factors(case, variant)

    active_ions = tuple(ion for ion in ions if ion not in hard_cutoff_species)

    # Precompute the (composition-independent) transport pieces once.
    diffusivities = dict(case["diffusivities_m2_s"])
    diffusive_hindrance = dict(hindrance["diffusive"])
    convective_hindrance = dict(hindrance["convective"])
    thickness_m = active_layer_thickness_m(case)
    temperature_K = case["temperature_K"]

    scales = compute_residual_scales(feed_concentrations_mol_L, water_flux_m_s, options)

    def _permeate_full(permeate_active):
        """Full permeate dict over all ions; hard-cutoff ions fixed to 0."""
        return {
            ion: (permeate_active[ion] if ion in permeate_active else 0.0)
            for ion in ions
        }

    def _finalize(success, message, permeate_full, feed_donnan, perm_donnan, dpsi_m,
                  scaled_vector, scaled_labels, status, nfev):
        state, transport = _build_state_and_transport(
            case, permeate_full, feed_donnan, perm_donnan, dpsi_m, water_flux_m_s,
            diffusivities, diffusive_hindrance, convective_hindrance,
            thickness_m, temperature_K, variant=variant,
        )
        raw = compute_two_interface_residuals(state, transport)
        return _assemble_result(
            success=success,
            message=message,
            water_flux_m_s=water_flux_m_s,
            active_ions=active_ions,
            hard_cutoff_species=hard_cutoff_species,
            permeate_full_mol_L=permeate_full,
            feed_donnan_V=feed_donnan,
            permeate_donnan_V=perm_donnan,
            membrane_potential_drop_V=dpsi_m,
            feed_concentrations_mol_L=feed_concentrations_mol_L,
            scaled_vector=scaled_vector,
            scaled_labels=scaled_labels,
            raw_named=dict(raw.named_residuals),
            raw_norm=raw.l2_norm,
            optimizer_status=status,
            optimizer_nfev=nfev,
        )

    # Structured failure: nothing to solve.
    if not active_ions:
        permeate_full = {ion: 0.0 for ion in ions}
        return _finalize(
            success=False,
            message=(
                "All modeled ions are hard-cutoff ("
                + ", ".join(hard_cutoff_species)
                + "); no active unknowns to solve. Returning zeros without "
                "claiming convergence."
            ),
            permeate_full=permeate_full,
            feed_donnan=0.0,
            perm_donnan=0.0,
            dpsi_m=0.0,
            scaled_vector=[],
            scaled_labels=(),
            status=None,
            nfev=None,
        )

    # Initial guess: feed-like permeate, zero potentials (deterministic, no use
    # of experimental rejection). Documented in the module docstring.
    initial_permeate = {ion: feed_concentrations_mol_L[ion] for ion in active_ions}
    x0 = pack_unknowns(initial_permeate, 0.0, 0.0, 0.0, active_ions)

    # Bounds: bounded concentrations (positivity) + bounded potentials.
    lower = []
    upper = []
    for ion in active_ions:
        lo = options.concentration_lower_bound_mol_L
        hi = options.concentration_upper_factor * feed_concentrations_mol_L[ion]
        if not (hi > lo):
            hi = lo + max(lo, 1e-9)
        lower.append(lo)
        upper.append(hi)
        # Clamp the initial guess strictly inside the bounds.
    for k, ion in enumerate(active_ions):
        x0[k] = min(max(x0[k], lower[k]), upper[k])
    for _ in range(3):
        lower.append(-options.potential_bounds_V)
        upper.append(options.potential_bounds_V)

    def residual_fun(x):
        permeate_active, feed_donnan, perm_donnan, dpsi_m = unpack_unknowns(x, active_ions)
        permeate_full = _permeate_full(permeate_active)
        state, transport = _build_state_and_transport(
            case, permeate_full, feed_donnan, perm_donnan, dpsi_m, water_flux_m_s,
            diffusivities, diffusive_hindrance, convective_hindrance,
            thickness_m, temperature_K, variant=variant,
        )
        vector, _ = _scaled_solve_vector(state, transport, active_ions, scales, options)
        return vector

    # Run the optimizer, converting any evaluation failure into a structured
    # result rather than an uninformative crash.
    try:
        solution = least_squares(
            residual_fun,
            x0,
            bounds=(lower, upper),
            max_nfev=options.max_nfev,
            verbose=2 if options.verbose else 0,
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        permeate_full = _permeate_full(initial_permeate)
        return _finalize(
            success=False,
            message=f"Optimizer failed with an exception: {exc!r}",
            permeate_full=permeate_full,
            feed_donnan=0.0,
            perm_donnan=0.0,
            dpsi_m=0.0,
            scaled_vector=[],
            scaled_labels=(),
            status=None,
            nfev=None,
        )

    permeate_active, feed_donnan, perm_donnan, dpsi_m = unpack_unknowns(
        solution.x, active_ions
    )
    permeate_full = _permeate_full(permeate_active)

    # Recompute the scaled vector and labels at the solution for reporting.
    state, transport = _build_state_and_transport(
        case, permeate_full, feed_donnan, perm_donnan, dpsi_m, water_flux_m_s,
        diffusivities, diffusive_hindrance, convective_hindrance,
        thickness_m, temperature_K, variant=variant,
    )
    scaled_vector, scaled_labels = _scaled_solve_vector(
        state, transport, active_ions, scales, options
    )

    max_abs_scaled = max(abs(v) for v in scaled_vector) if scaled_vector else math.inf
    converged = bool(solution.success) and math.isfinite(max_abs_scaled) and (
        max_abs_scaled <= options.tolerance
    )
    if converged:
        message = (
            f"Converged: max abs scaled residual {max_abs_scaled:.3e} <= "
            f"tolerance {options.tolerance:.3e} (optimizer: {solution.message})."
        )
    else:
        message = (
            f"Not converged: max abs scaled residual "
            f"{max_abs_scaled:.3e} vs tolerance {options.tolerance:.3e} "
            f"(optimizer status {solution.status}: {solution.message})."
        )

    return _finalize(
        success=converged,
        message=message,
        permeate_full=permeate_full,
        feed_donnan=feed_donnan,
        perm_donnan=perm_donnan,
        dpsi_m=dpsi_m,
        scaled_vector=scaled_vector,
        scaled_labels=scaled_labels,
        status=int(solution.status),
        nfev=int(solution.nfev),
    )


def solve_foo2023_lmc_ph7_flux(
    water_flux_m_s,
    options: Optional[TwoInterfaceSolverOptions] = None,
    variant=None,
) -> TwoInterfaceSolveResult:
    """
    Solve the two-interface state for one Foo 2023 LM-C pH 7 water flux.

    Convenience wrapper over :func:`solve_two_interface_for_flux` with the
    ``foo2023_lmc_ph7_case()``. Convergence is not assumed; the result is
    structured and honest whether or not the solve succeeds. ``variant`` is
    forwarded (``None`` => hard baseline).
    """
    return solve_two_interface_for_flux(
        foo2023_lmc_ph7_case(), water_flux_m_s, options, variant=variant
    )


def solve_foo2023_lmc_ph7_all_fluxes(
    options: Optional[TwoInterfaceSolverOptions] = None,
    variant=None,
) -> Tuple[TwoInterfaceSolveResult, ...]:
    """
    Solve the two-interface state for all four Foo 2023 LM-C pH 7 water fluxes.

    ``variant`` is forwarded to every flux (``None`` => hard baseline).

    Returns
    -------
    tuple of TwoInterfaceSolveResult
        One structured result per experimental water flux, in the case's order.
        Failed solves remain explicit (``success=False``), never dropped.
    """
    case = foo2023_lmc_ph7_case()
    water_fluxes_m_s = [
        value * 1e-6 for value in case["experimental_data"]["Jw_um_s"]
    ]
    return tuple(
        solve_two_interface_for_flux(case, water_flux_m_s, options, variant=variant)
        for water_flux_m_s in water_fluxes_m_s
    )
