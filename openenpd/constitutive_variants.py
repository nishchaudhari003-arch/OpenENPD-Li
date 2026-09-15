"""
Configurable constitutive model variants for steric/hindrance treatment.

Milestone 5 of the two-interface roadmap. The Foo 2023 LM-C pH 7 failure is
strongly linked to the **hard steric/hindrance cutoff**: a hydrated ion at or
above the pore radius (λ = r_i / r_p >= 1) is given a steric partition factor
of exactly 0 and zero hindrance, which forces Mg2+ out of the transport problem
entirely. This module adds *configurable, explicitly documented* constitutive
variants that relax that hard cutoff in controlled diagnostic ways.

These variants are **diagnostic tools, not validated physical laws**. Nothing
here is tuned to force agreement with experiment, and the default configuration
reproduces the existing hard-baseline behaviour exactly.

Variant rules (as functions of the effective size ratio λ)
----------------------------------------------------------
The base steric/hindrance term is the existing ``s(λ) = (max(1 - λ, 0))**2``
(0 for λ >= 1). Radius scaling changes λ itself:
``λ_eff = (ion_radius_scale * r_i) / r_p_eff`` where ``r_p_eff`` is
``pore_radius_nm_override`` if given else ``pore_radius_scale * r_p``.

- ``"hard"``  : the existing behaviour, ``s(λ_eff)`` (exactly 0 for λ_eff >= 1).
- ``"floor"`` : ``max(s(λ_eff), floor)`` — a minimum positive factor so a
  size-excluded ion is not identically zeroed. Diagnostic only.
- ``"soft"``  : a smooth, everywhere-positive replacement of the discontinuous
  hard clamp built from a softplus,
  ``s_soft(λ_eff) = (w * ln(1 + exp((1 - λ_eff)/w)))**2`` with
  ``w = soft_cutoff_width``. As ``w -> 0`` this recovers the hard term; for
  finite ``w`` it removes the abrupt zero near λ = 1. A ``floor`` is also
  applied so the result is never below ``steric_floor`` / ``hindrance_floor``
  (both default 0, so no effect unless set). Diagnostic only.

The same rules are used for the steric partition factor and for the
diffusive/convective hindrance factors (which share the ``(1 - λ)**2`` form in
this toolkit).

Effective ion radius also feeds the Born/dielectric penalty, so
``ion_radius_scale`` consistently rescales the dielectric factor as well.

Nothing in this module claims validation, a physics fix, or reproduction of
Foo 2023.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.dielectric import dielectric_partition_factor
from openenpd.donnan import donnan_partition_factor
from openenpd.steric import size_ratio
from openenpd.two_interface import TwoInterfaceState

__all__ = [
    "ConstitutiveVariantConfig",
    "STERIC_MODELS",
    "HINDRANCE_MODELS",
    "steric_partition_factor_variant",
    "hindrance_factor_variant",
    "effective_ion_radius_nm",
    "effective_pore_radius_nm",
    "effective_size_ratio",
    "build_two_interface_state_variant",
    "variant_ions_and_cutoff",
    "variant_hindrance_factors",
    "default_variants",
    "variant_foo2023_lmc_ph7_comparison_rows",
    "run_foo2023_lmc_ph7_variant_comparison",
    "variant_foo2023_lmc_ph7_metrics",
    "scan_ion_radius_scale",
    "scan_pore_radius_scale",
    "scan_steric_floor",
    "scan_hindrance_floor",
    "DEFAULT_ION_RADIUS_SCALE_SCAN",
    "DEFAULT_PORE_RADIUS_SCALE_SCAN",
    "DEFAULT_STERIC_FLOOR_SCAN",
    "DEFAULT_HINDRANCE_FLOOR_SCAN",
]

STERIC_MODELS = ("hard", "floor", "soft")
HINDRANCE_MODELS = ("hard", "floor", "soft")

DEFAULT_ION_RADIUS_SCALE_SCAN = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
DEFAULT_PORE_RADIUS_SCALE_SCAN = (1.00, 1.25, 1.50, 1.75, 2.00)
DEFAULT_STERIC_FLOOR_SCAN = (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
DEFAULT_HINDRANCE_FLOOR_SCAN = (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)


def _read_only(mapping):
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class ConstitutiveVariantConfig:
    """
    Configuration for a constitutive steric/hindrance variant.

    The default is the hard baseline and reproduces the existing behaviour
    exactly (``steric_model="hard"``, ``hindrance_model="hard"``, unit scales,
    zero floors).

    Parameters
    ----------
    steric_model : str
        One of ``"hard"``, ``"floor"``, ``"soft"``.
    hindrance_model : str
        One of ``"hard"``, ``"floor"``, ``"soft"``.
    ion_radius_scale : float
        Multiplier on hydrated ion radii (must be > 0).
    pore_radius_scale : float
        Multiplier on the pore radius (must be > 0). Ignored if
        ``pore_radius_nm_override`` is set.
    pore_radius_nm_override : float or None
        Explicit effective pore radius in nm (must be > 0 if given). Takes
        precedence over ``pore_radius_scale``.
    steric_floor : float
        Minimum steric partition factor for the ``"floor"``/``"soft"`` models
        (0 <= value <= 1).
    hindrance_floor : float
        Minimum hindrance factor for the ``"floor"``/``"soft"`` models
        (0 <= value <= 1).
    soft_cutoff_width : float
        Width ``w`` of the softplus transition for the ``"soft"`` model
        (must be > 0). Unused by ``"hard"``/``"floor"``.
    label : str
        Human-readable label carried into diagnostics.

    Raises
    ------
    ValueError
        On an invalid model name, non-positive scale/width/override, or a floor
        outside ``[0, 1]``.
    """

    steric_model: str = "hard"
    hindrance_model: str = "hard"
    ion_radius_scale: float = 1.0
    pore_radius_scale: float = 1.0
    pore_radius_nm_override: Optional[float] = None
    steric_floor: float = 0.0
    hindrance_floor: float = 0.0
    soft_cutoff_width: float = 0.1
    label: str = "hard_baseline"

    def __post_init__(self):
        if self.steric_model not in STERIC_MODELS:
            raise ValueError(
                f"steric_model must be one of {STERIC_MODELS}, got "
                f"{self.steric_model!r}."
            )
        if self.hindrance_model not in HINDRANCE_MODELS:
            raise ValueError(
                f"hindrance_model must be one of {HINDRANCE_MODELS}, got "
                f"{self.hindrance_model!r}."
            )
        if not (self.ion_radius_scale > 0.0):
            raise ValueError("ion_radius_scale must be positive.")
        if not (self.pore_radius_scale > 0.0):
            raise ValueError("pore_radius_scale must be positive.")
        if self.pore_radius_nm_override is not None and not (
            self.pore_radius_nm_override > 0.0
        ):
            raise ValueError("pore_radius_nm_override must be positive if given.")
        if not (0.0 <= self.steric_floor <= 1.0):
            raise ValueError("steric_floor must be in [0, 1].")
        if not (0.0 <= self.hindrance_floor <= 1.0):
            raise ValueError("hindrance_floor must be in [0, 1].")
        if not (self.soft_cutoff_width > 0.0):
            raise ValueError("soft_cutoff_width must be positive.")


# ---------------------------------------------------------------------------
# Factor functions
# ---------------------------------------------------------------------------

def _hard_term(lambda_ratio):
    """Base steric/hindrance term: (1 - lambda)^2 for lambda < 1, else 0.0."""
    if lambda_ratio >= 1.0:
        return 0.0
    return (1.0 - lambda_ratio) ** 2


def _softplus(x):
    """Numerically stable softplus ln(1 + exp(x))."""
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _soft_term(lambda_ratio, width):
    """Smooth, everywhere-positive replacement of the hard clamp."""
    smoothed = width * _softplus((1.0 - lambda_ratio) / width)
    return smoothed * smoothed


def steric_partition_factor_variant(lambda_ratio, config: ConstitutiveVariantConfig):
    """
    Steric partition factor under the configured variant.

    Returns
    -------
    float
        Dimensionless steric partition factor.
    """
    if config.steric_model == "hard":
        return _hard_term(lambda_ratio)
    if config.steric_model == "floor":
        return max(_hard_term(lambda_ratio), config.steric_floor)
    # "soft"
    return max(_soft_term(lambda_ratio, config.soft_cutoff_width), config.steric_floor)


def hindrance_factor_variant(lambda_ratio, config: ConstitutiveVariantConfig):
    """
    Diffusive/convective hindrance factor under the configured variant.

    The toolkit uses the same ``(1 - λ)^2`` form for both hindrance factors, so
    a single function serves both.

    Returns
    -------
    float
        Dimensionless hindrance factor.
    """
    if config.hindrance_model == "hard":
        return _hard_term(lambda_ratio)
    if config.hindrance_model == "floor":
        return max(_hard_term(lambda_ratio), config.hindrance_floor)
    # "soft"
    return max(
        _soft_term(lambda_ratio, config.soft_cutoff_width), config.hindrance_floor
    )


def effective_ion_radius_nm(ion_radius_nm, config: ConstitutiveVariantConfig):
    """Return the scaled effective ion radius in nm."""
    return config.ion_radius_scale * ion_radius_nm


def effective_pore_radius_nm(pore_radius_nm, config: ConstitutiveVariantConfig):
    """Return the effective pore radius in nm (override wins over scale)."""
    if config.pore_radius_nm_override is not None:
        return config.pore_radius_nm_override
    return config.pore_radius_scale * pore_radius_nm


def effective_size_ratio(ion_radius_nm, pore_radius_nm, config: ConstitutiveVariantConfig):
    """Return the effective size ratio λ_eff = r_i_eff / r_p_eff."""
    return size_ratio(
        ion_radius_nm=effective_ion_radius_nm(ion_radius_nm, config),
        pore_radius_nm=effective_pore_radius_nm(pore_radius_nm, config),
    )


# ---------------------------------------------------------------------------
# Variant state assembly (mirrors openenpd.two_interface for the hard default)
# ---------------------------------------------------------------------------

def build_two_interface_state_variant(
    case,
    permeate_concentrations_mol_L,
    feed_side_donnan_V,
    permeate_side_donnan_V,
    config: ConstitutiveVariantConfig,
) -> TwoInterfaceState:
    """
    Assemble a :class:`~openenpd.two_interface.TwoInterfaceState` under a variant.

    This mirrors :func:`openenpd.two_interface.assemble_two_interface_state`
    (same arithmetic and field layout) but uses the variant steric factor,
    effective (scaled) radii, and a hard-cutoff criterion of "effective steric
    factor is exactly 0". For the default hard config it reproduces the baseline
    assembler exactly. It reads ``case`` without mutating it.

    Parameters
    ----------
    case : dict
        Validation-case definition.
    permeate_concentrations_mol_L : Mapping[str, float]
        Provided permeate concentrations in mol/L.
    feed_side_donnan_V, permeate_side_donnan_V : float
        Provided Donnan potentials in volts.
    config : ConstitutiveVariantConfig
        The variant configuration.

    Returns
    -------
    TwoInterfaceState
        The assembled, immutable state.
    """
    feed_bulk = dict(case["bulk_concentrations_mol_L"])
    charges = dict(case["charges"])
    ion_radii_nm = dict(case["ion_radii_nm"])
    params = case["membrane_parameters"]
    pore_radius_nm = params["pore_radius_nm"]
    fixed_charge_mol_L = params["fixed_charge_mol_L"]
    pore_dielectric_constant = params["pore_dielectric_constant"]
    temperature_K = case["temperature_K"]

    ions = tuple(feed_bulk.keys())

    size_ratios = {}
    steric_factors = {}
    dielectric_factors = {}
    for ion in ions:
        r_eff = effective_ion_radius_nm(ion_radii_nm[ion], config)
        r_p_eff = effective_pore_radius_nm(pore_radius_nm, config)
        lambda_eff = size_ratio(ion_radius_nm=r_eff, pore_radius_nm=r_p_eff)
        size_ratios[ion] = lambda_eff
        steric_factors[ion] = steric_partition_factor_variant(lambda_eff, config)
        # Effective ion radius feeds the Born penalty too.
        dielectric_factors[ion] = dielectric_partition_factor(
            charge=charges[ion],
            ion_radius_nm=r_eff,
            pore_dielectric_constant=pore_dielectric_constant,
            temperature_K=temperature_K,
        )

    feed_donnan_factors = {}
    permeate_donnan_factors = {}
    for ion in ions:
        feed_donnan_factors[ion] = donnan_partition_factor(
            charge=charges[ion],
            delta_psi_V=feed_side_donnan_V,
            temperature_K=temperature_K,
        )
        permeate_donnan_factors[ion] = donnan_partition_factor(
            charge=charges[ion],
            delta_psi_V=permeate_side_donnan_V,
            temperature_K=temperature_K,
        )

    feed_partition_factors = {}
    permeate_partition_factors = {}
    for ion in ions:
        common = steric_factors[ion] * dielectric_factors[ion]
        feed_partition_factors[ion] = feed_donnan_factors[ion] * common
        permeate_partition_factors[ion] = permeate_donnan_factors[ion] * common

    feed_membrane = {}
    permeate_membrane = {}
    for ion in ions:
        feed_membrane[ion] = feed_bulk[ion] * feed_partition_factors[ion]
        permeate_membrane[ion] = (
            permeate_concentrations_mol_L[ion] * permeate_partition_factors[ion]
        )

    feed_residual = (
        sum(charges[ion] * feed_membrane[ion] for ion in ions) + fixed_charge_mol_L
    )
    permeate_residual = (
        sum(charges[ion] * permeate_membrane[ion] for ion in ions)
        + fixed_charge_mol_L
    )

    # Hard-cutoff = effective steric factor collapses to exactly zero. For the
    # hard model this matches lambda >= 1; for floor/soft it is never zero.
    hard_cutoff_species = tuple(
        ion for ion in ions if steric_factors[ion] == 0.0
    )

    return TwoInterfaceState(
        ions=ions,
        charges=_read_only(charges),
        fixed_charge_mol_L=fixed_charge_mol_L,
        feed_side_donnan_V=feed_side_donnan_V,
        permeate_side_donnan_V=permeate_side_donnan_V,
        feed_bulk_concentrations_mol_L=_read_only(feed_bulk),
        permeate_concentrations_mol_L=_read_only(dict(permeate_concentrations_mol_L)),
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


def variant_ions_and_cutoff(
    case, config: ConstitutiveVariantConfig
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Return ``(ions, hard_cutoff_species)`` for a case under a variant.

    Hard-cutoff depends only on effective radii/pore, so a zero-potential probe
    with feed as placeholder permeate is representative.
    """
    probe = build_two_interface_state_variant(
        case,
        permeate_concentrations_mol_L=dict(case["bulk_concentrations_mol_L"]),
        feed_side_donnan_V=0.0,
        permeate_side_donnan_V=0.0,
        config=config,
    )
    return probe.ions, probe.hard_cutoff_species


def variant_hindrance_factors(
    case, config: ConstitutiveVariantConfig
) -> Mapping[str, Mapping[str, float]]:
    """
    Return diffusive/convective hindrance factors for a case under a variant.

    Mirrors :func:`openenpd.hindrance.hindrance_factors` structure but applies
    the effective size ratio and the configured hindrance model.

    Returns
    -------
    dict
        ``{"diffusive": {ion: K_d}, "convective": {ion: K_c}}``.
    """
    ion_radii_nm = case["ion_radii_nm"]
    pore_radius_nm = case["membrane_parameters"]["pore_radius_nm"]

    diffusive = {}
    convective = {}
    for ion, radius_nm in ion_radii_nm.items():
        lambda_eff = effective_size_ratio(radius_nm, pore_radius_nm, config)
        factor = hindrance_factor_variant(lambda_eff, config)
        diffusive[ion] = factor
        convective[ion] = factor
    return {"diffusive": diffusive, "convective": convective}


# ---------------------------------------------------------------------------
# Predefined variants
# ---------------------------------------------------------------------------

def default_variants() -> Tuple[ConstitutiveVariantConfig, ...]:
    """
    Return a deterministic tuple of diagnostic variant configurations.

    Includes the hard baseline, steric-floor-only, hindrance-floor-only,
    ion-radius scaling, pore-radius scaling, and a combined diagnostic variant.
    None of these are tuned or claimed to be physically correct.
    """
    return (
        ConstitutiveVariantConfig(label="hard_baseline"),
        ConstitutiveVariantConfig(
            steric_model="floor", steric_floor=0.01, label="steric_floor_0p01"
        ),
        ConstitutiveVariantConfig(
            hindrance_model="floor",
            hindrance_floor=0.01,
            label="hindrance_floor_0p01",
        ),
        ConstitutiveVariantConfig(
            ion_radius_scale=0.85, label="ion_radius_scale_0p85"
        ),
        ConstitutiveVariantConfig(
            pore_radius_scale=1.25, label="pore_radius_scale_1p25"
        ),
        ConstitutiveVariantConfig(
            steric_model="floor",
            hindrance_model="floor",
            steric_floor=0.01,
            hindrance_floor=0.01,
            ion_radius_scale=0.90,
            pore_radius_scale=1.10,
            label="combined_diagnostic",
        ),
    )


# ---------------------------------------------------------------------------
# Diagnostics over the Foo 2023 LM-C pH 7 case
# ---------------------------------------------------------------------------

def _config_row_fields(config: ConstitutiveVariantConfig):
    """Config columns shared by comparison rows."""
    return {
        "variant_label": config.label,
        "steric_model": config.steric_model,
        "hindrance_model": config.hindrance_model,
        "ion_radius_scale": config.ion_radius_scale,
        "pore_radius_scale": config.pore_radius_scale,
        "pore_radius_nm_override": config.pore_radius_nm_override,
        "steric_floor": config.steric_floor,
        "hindrance_floor": config.hindrance_floor,
    }


def variant_foo2023_lmc_ph7_comparison_rows(config, options=None):
    """
    Build one comparison row per Foo 2023 LM-C pH 7 water flux for a variant.

    Each row carries the variant label and configuration, the experimental
    rejections, the two-interface best-effort predictions, and solver
    diagnostics. Failed solves remain explicit (``solve_success=False``).

    Parameters
    ----------
    config : ConstitutiveVariantConfig
        The variant configuration to run.
    options : TwoInterfaceSolverOptions, optional
        Options forwarded to the two-interface solver.

    Returns
    -------
    list of dict
        Four rows (one per experimental flux).
    """
    # Lazy import to avoid an import cycle (the solver imports this module's
    # state builder lazily as well).
    from openenpd.two_interface_solver import solve_two_interface_for_flux

    case = foo2023_lmc_ph7_case()
    experimental = case["experimental_data"]
    water_fluxes_m_s = [value * 1e-6 for value in experimental["Jw_um_s"]]
    config_fields = _config_row_fields(config)

    rows = []
    for index, water_flux_m_s in enumerate(water_fluxes_m_s):
        result = solve_two_interface_for_flux(
            case, water_flux_m_s, options=options, variant=config
        )
        rejections = dict(result.predicted_rejections)

        row = {
            **config_fields,
            "pressure_bar": experimental["pressure_bar"][index],
            "Jw_LMH": experimental["Jw_LMH"][index],
            "Jw_um_s": experimental["Jw_um_s"][index],
            "experimental_R_Li": experimental["R_Li"][index],
            "experimental_R_Mg": experimental["R_Mg"][index],
            "predicted_R_Li": rejections.get("Li+"),
            "predicted_R_Mg": rejections.get("Mg2+"),
            "solve_success": result.success,
            "scaled_residual_max_abs": result.residual_max_abs_scaled,
            "scaled_residual_norm": result.residual_norm_scaled,
            "hard_cutoff_species": tuple(result.hard_cutoff_species),
        }
        rows.append(row)

    return rows


def run_foo2023_lmc_ph7_variant_comparison(configs=None, options=None):
    """
    Build comparison rows across several variants (default: :func:`default_variants`).

    Returns
    -------
    list of dict
        ``4 * len(configs)`` rows, each labelled by ``variant_label``.
    """
    if configs is None:
        configs = default_variants()

    rows = []
    for config in configs:
        rows.extend(variant_foo2023_lmc_ph7_comparison_rows(config, options=options))
    return rows


def _rmse(errors):
    if not errors:
        return None
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def variant_foo2023_lmc_ph7_metrics(config, options=None):
    """
    Compute diagnostic metrics for one variant on Foo 2023 LM-C pH 7.

    RMSE numbers are computed over the two-interface **best-effort** predictions
    (all fluxes, regardless of convergence) and are labelled diagnostic. They
    are not a validation result unless ``number_successful_solves`` is positive
    and the residuals are small.

    Returns
    -------
    dict
        Variant label/config plus RMSE, solve counts, and the Mg-hard-cutoff /
        negative-Li diagnostics.
    """
    rows = variant_foo2023_lmc_ph7_comparison_rows(config, options=options)

    li_errors = [
        r["predicted_R_Li"] - r["experimental_R_Li"]
        for r in rows
        if r["predicted_R_Li"] is not None
    ]
    mg_errors = [
        r["predicted_R_Mg"] - r["experimental_R_Mg"]
        for r in rows
        if r["predicted_R_Mg"] is not None
    ]

    n_success = sum(1 for r in rows if r["solve_success"])
    mg_hard_cutoff_present = any(
        "Mg2+" in r["hard_cutoff_species"] for r in rows
    )
    reproduces_negative_li = any(
        r["predicted_R_Li"] is not None and r["predicted_R_Li"] < 0.0 for r in rows
    )
    reproduces_negative_li_successful = any(
        r["predicted_R_Li"] is not None
        and r["predicted_R_Li"] < 0.0
        and r["solve_success"]
        for r in rows
    )

    return {
        "variant_label": config.label,
        "steric_model": config.steric_model,
        "hindrance_model": config.hindrance_model,
        "ion_radius_scale": config.ion_radius_scale,
        "pore_radius_scale": config.pore_radius_scale,
        "pore_radius_nm_override": config.pore_radius_nm_override,
        "steric_floor": config.steric_floor,
        "hindrance_floor": config.hindrance_floor,
        "R_Li_rmse": _rmse(li_errors),
        "R_Mg_rmse": _rmse(mg_errors),
        "total_rmse": _rmse(li_errors + mg_errors),
        "number_successful_solves": n_success,
        "number_failed_solves": len(rows) - n_success,
        "reproduces_negative_Li_rejection": reproduces_negative_li,
        "reproduces_negative_Li_rejection_in_successful_solve": (
            reproduces_negative_li_successful
        ),
        "Mg_hard_cutoff_present": mg_hard_cutoff_present,
        "rmse_is_best_effort_diagnostic": n_success < len(rows),
    }


# ---------------------------------------------------------------------------
# Sensitivity scans
# ---------------------------------------------------------------------------

def _scan(values, config_builder, options=None):
    """Run variant metrics for each scan value; return a list of metric dicts."""
    return [
        variant_foo2023_lmc_ph7_metrics(config_builder(value), options=options)
        for value in values
    ]


def scan_ion_radius_scale(values=None, options=None):
    """Scan ``ion_radius_scale`` and return one metrics dict per value."""
    if values is None:
        values = DEFAULT_ION_RADIUS_SCALE_SCAN
    return _scan(
        values,
        lambda value: ConstitutiveVariantConfig(
            ion_radius_scale=value, label=f"ion_radius_scale_{value:g}"
        ),
        options=options,
    )


def scan_pore_radius_scale(values=None, options=None):
    """Scan ``pore_radius_scale`` and return one metrics dict per value."""
    if values is None:
        values = DEFAULT_PORE_RADIUS_SCALE_SCAN
    return _scan(
        values,
        lambda value: ConstitutiveVariantConfig(
            pore_radius_scale=value, label=f"pore_radius_scale_{value:g}"
        ),
        options=options,
    )


def scan_steric_floor(values=None, options=None):
    """Scan ``steric_floor`` (floor model) and return one metrics dict per value."""
    if values is None:
        values = DEFAULT_STERIC_FLOOR_SCAN
    return _scan(
        values,
        lambda value: ConstitutiveVariantConfig(
            steric_model="floor", steric_floor=value, label=f"steric_floor_{value:g}"
        ),
        options=options,
    )


def scan_hindrance_floor(values=None, options=None):
    """Scan ``hindrance_floor`` (floor model) and return one metrics dict per value."""
    if values is None:
        values = DEFAULT_HINDRANCE_FLOOR_SCAN
    return _scan(
        values,
        lambda value: ConstitutiveVariantConfig(
            hindrance_model="floor",
            hindrance_floor=value,
            label=f"hindrance_floor_{value:g}",
        ),
        options=options,
    )
