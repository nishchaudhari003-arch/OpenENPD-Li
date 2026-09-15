"""
Bounded parameter estimation and identifiability diagnostics for OpenENPD-Li.

Milestone 6 of the two-interface roadmap. This module adds a small, bounded
parameter-estimation layer over the Foo 2023 LM-C pH 7 case, together with
identifiability diagnostics. It asks a narrow, honest question: can one or two
*effective* parameters improve agreement with the experimental Li/Mg rejection,
and are those parameters identifiable from only four flux points?

This is **not** a validation-success milestone and **not** parameter tuning for
pretty plots. The parameters fitted here are *effective diagnostic parameters*,
not unique physical estimates. Nothing is claimed validated.

Data and objective
-------------------
The target is the Foo 2023 LM-C pH 7 experimental Li and Mg rejection at the
four water fluxes. For each flux and fitted species the residual is
``R_i,predicted - R_i,experimental``. With Li and Mg over four fluxes there are
at most ``2 * 4 = 8`` rejection residuals, so **only one- or two-parameter fits
are run by default** — more parameters than that cannot be constrained by eight
points, and the code emits an identifiability warning if asked.

Effective parameters (each with explicit bounds)
------------------------------------------------
Mapped either onto a :class:`~openenpd.constitutive_variants.ConstitutiveVariantConfig`
field or onto a copy of the case's membrane parameters:

- ``ion_radius_scale``     (variant; 0.40-1.20)
- ``pore_radius_scale``    (variant; 0.80-2.50)
- ``steric_floor``         (variant, forces ``steric_model="floor"``; 0.0-0.30)
- ``hindrance_floor``      (variant, forces ``hindrance_model="floor"``; 0.0-0.30)
- ``fixed_charge_scale``   (case: scales ``fixed_charge_mol_L``; 0.25-2.00)
- ``dielectric_scale``     (case: scales ``pore_dielectric_constant``; 0.50-1.50)

Failed solves are never hidden: the objective augments the rejection residuals
with a per-flux penalty for non-converged solves, RMSE is reported as
best-effort with explicit success/failure counts, and a fit whose model solves
all fail is not reported as successful.
"""

import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.constitutive_variants import ConstitutiveVariantConfig
from openenpd.two_interface_solver import (
    TwoInterfaceSolverOptions,
    solve_two_interface_for_flux,
)

__all__ = [
    "PARAMETER_REGISTRY",
    "SUPPORTED_PARAMETERS",
    "ParameterSpec",
    "parameter_spec",
    "EstimationConfig",
    "EstimationResult",
    "build_foo2023_estimation_objective",
    "fit_foo2023_parameters",
    "run_foo2023_single_parameter_fits",
    "run_foo2023_two_parameter_fits",
    "foo2023_parameter_estimation_summary",
    "foo2023_parameter_estimation_summary_dataframe",
    "objective_scan_1d",
    "objective_scan_2d",
    "DEFAULT_SINGLE_PARAMETERS",
    "DEFAULT_PARAMETER_PAIRS",
    "CONDITION_NUMBER_WARNING_THRESHOLD",
]

# name -> (target, default_initial, default_lower, default_upper, description)
PARAMETER_REGISTRY = {
    "ion_radius_scale": (
        "variant",
        1.0,
        0.40,
        1.20,
        "Multiplier on hydrated ion radii (effective in-pore radius).",
    ),
    "pore_radius_scale": (
        "variant",
        1.0,
        0.80,
        2.50,
        "Multiplier on the membrane pore radius (effective transport radius).",
    ),
    "steric_floor": (
        "variant",
        0.01,
        0.0,
        0.30,
        "Minimum steric partition factor (forces steric_model='floor').",
    ),
    "hindrance_floor": (
        "variant",
        0.01,
        0.0,
        0.30,
        "Minimum hindrance factor (forces hindrance_model='floor').",
    ),
    "fixed_charge_scale": (
        "case",
        1.0,
        0.25,
        2.00,
        "Multiplier on the fixed membrane charge concentration.",
    ),
    "dielectric_scale": (
        "case",
        1.0,
        0.50,
        1.50,
        "Multiplier on the membrane pore dielectric constant.",
    ),
}

SUPPORTED_PARAMETERS = tuple(PARAMETER_REGISTRY.keys())

DEFAULT_SINGLE_PARAMETERS = (
    "ion_radius_scale",
    "pore_radius_scale",
    "steric_floor",
    "hindrance_floor",
    "fixed_charge_scale",
)

DEFAULT_PARAMETER_PAIRS = (
    ("ion_radius_scale", "pore_radius_scale"),
    ("ion_radius_scale", "hindrance_floor"),
    ("pore_radius_scale", "hindrance_floor"),
    ("fixed_charge_scale", "ion_radius_scale"),
)

CONDITION_NUMBER_WARNING_THRESHOLD = 1.0e6

# Inner two-interface solver options used when the caller passes none. The
# max_nfev cap only bounds churn on non-converging solves; converging solves
# finish in far fewer evaluations, so this does not change which solves succeed.
_DEFAULT_ESTIMATION_SOLVER_OPTIONS = TwoInterfaceSolverOptions(max_nfev=250)


def _read_only(mapping):
    return MappingProxyType(dict(mapping))


# ---------------------------------------------------------------------------
# Parameter specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterSpec:
    """
    Specification of a single fitted effective parameter.

    Parameters
    ----------
    name : str
        A supported parameter name (see :data:`SUPPORTED_PARAMETERS`).
    initial_value : float
        Initial guess; must lie within ``[lower_bound, upper_bound]``.
    lower_bound, upper_bound : float
        Bounds (``lower_bound < upper_bound``). Physical scales must be
        non-negative.
    description : str
        Human-readable description.

    Raises
    ------
    ValueError
        On an unsupported name, ``lower_bound >= upper_bound``, a negative
        bound, or an initial value outside the bounds.
    """

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    description: str = ""

    def __post_init__(self):
        if self.name not in PARAMETER_REGISTRY:
            raise ValueError(
                f"Unsupported parameter name {self.name!r}; supported: "
                f"{SUPPORTED_PARAMETERS}."
            )
        if not (self.lower_bound < self.upper_bound):
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be < upper_bound "
                f"({self.upper_bound}) for {self.name!r}."
            )
        if self.lower_bound < 0.0:
            raise ValueError(
                f"lower_bound must be non-negative for {self.name!r} "
                "(physical scales/floors cannot be negative)."
            )
        if not (self.lower_bound <= self.initial_value <= self.upper_bound):
            raise ValueError(
                f"initial_value ({self.initial_value}) must lie within bounds "
                f"[{self.lower_bound}, {self.upper_bound}] for {self.name!r}."
            )


def parameter_spec(
    name,
    initial_value=None,
    lower_bound=None,
    upper_bound=None,
    description=None,
) -> ParameterSpec:
    """
    Build a :class:`ParameterSpec`, filling defaults from :data:`PARAMETER_REGISTRY`.

    Raises
    ------
    ValueError
        If ``name`` is unsupported.
    """
    if name not in PARAMETER_REGISTRY:
        raise ValueError(
            f"Unsupported parameter name {name!r}; supported: {SUPPORTED_PARAMETERS}."
        )
    _target, default_initial, default_lower, default_upper, default_desc = (
        PARAMETER_REGISTRY[name]
    )
    return ParameterSpec(
        name=name,
        initial_value=default_initial if initial_value is None else initial_value,
        lower_bound=default_lower if lower_bound is None else lower_bound,
        upper_bound=default_upper if upper_bound is None else upper_bound,
        description=default_desc if description is None else description,
    )


# ---------------------------------------------------------------------------
# Estimation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimationConfig:
    """
    Configuration for a Foo 2023 parameter-estimation fit.

    Parameters
    ----------
    parameter_specs : sequence of ParameterSpec
        The parameters to fit (non-empty). Names must be unique.
    base_variant_config : ConstitutiveVariantConfig, optional
        The constitutive variant the fit builds on (defaults to the hard
        baseline). Fitted variant parameters override the corresponding fields.
    species_to_fit : sequence of str, optional
        Species whose rejection is fitted (default ``("Li+", "Mg2+")``).
    include_failed_solves : bool
        If True (default), best-effort rejection residuals from non-converged
        solves are used (plus a per-flux penalty). If False, a failed flux's
        rejection residuals are replaced by the penalty value.
    failed_solve_penalty : float
        Non-negative penalty added per non-converged flux, so the optimizer is
        pushed away from failing regions without hiding the failures.
    max_nfev : int
        Maximum residual evaluations for the outer least_squares.
    tolerance : float
        ``ftol``/``xtol``/``gtol`` passed to least_squares.
    solver_options : TwoInterfaceSolverOptions, optional
        Inner two-interface solver options.
    seed_grid_points : int
        Per-parameter density of the deterministic coarse grid used to seed the
        local optimizer. Because the hard-cutoff release makes the objective
        discontinuous, a local optimizer started at the identity point can get
        stuck in a failing region; the grid seed (which the per-flux penalty
        pushes toward converging points) makes the bounded fit robust and
        deterministic. Set to 1 to disable seeding and start from
        ``initial_value``.

    Raises
    ------
    ValueError
        On empty or duplicate specs, unknown species, a negative penalty, or an
        invalid grid density.
    """

    parameter_specs: Tuple[ParameterSpec, ...]
    base_variant_config: ConstitutiveVariantConfig = field(
        default_factory=ConstitutiveVariantConfig
    )
    species_to_fit: Tuple[str, ...] = ("Li+", "Mg2+")
    include_failed_solves: bool = True
    failed_solve_penalty: float = 1.0
    max_nfev: int = 200
    tolerance: float = 1.0e-8
    solver_options: Optional[TwoInterfaceSolverOptions] = None
    seed_grid_points: int = 4

    def __post_init__(self):
        object.__setattr__(self, "parameter_specs", tuple(self.parameter_specs))
        object.__setattr__(self, "species_to_fit", tuple(self.species_to_fit))

        if not self.parameter_specs:
            raise ValueError("parameter_specs must be non-empty.")
        names = [spec.name for spec in self.parameter_specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names in specs: {names}.")
        allowed_species = set(foo2023_lmc_ph7_case()["charges"].keys())
        for species in self.species_to_fit:
            if species not in allowed_species:
                raise ValueError(
                    f"species_to_fit contains {species!r} not in the case "
                    f"({sorted(allowed_species)})."
                )
        if not self.species_to_fit:
            raise ValueError("species_to_fit must be non-empty.")
        if self.failed_solve_penalty < 0.0:
            raise ValueError("failed_solve_penalty must be non-negative.")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer.")
        if self.seed_grid_points < 1:
            raise ValueError("seed_grid_points must be a positive integer.")

    @property
    def number_of_parameters(self) -> int:
        return len(self.parameter_specs)

    def identifiability_warnings(self):
        """Return static identifiability warnings implied by the config shape."""
        warnings = []
        n_params = self.number_of_parameters
        n_data = len(self.species_to_fit) * 4
        if n_params > 2:
            warnings.append(
                f"Fitting {n_params} parameters to only {n_data} rejection data "
                "points: strongly under-constrained; treat as diagnostic only."
            )
        if n_params >= n_data:
            warnings.append(
                f"Degrees of freedom <= 0 ({n_data} data - {n_params} params); "
                "the fit cannot be constrained."
            )
        return warnings


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimationResult:
    """Structured result of a bounded Foo 2023 parameter-estimation fit."""

    success: bool
    message: str
    parameter_values: Mapping[str, float]
    initial_parameter_values: Mapping[str, float]
    bounds: Mapping[str, Tuple[float, float]]
    residual_vector: Tuple[float, ...]
    rmse_by_species: Mapping[str, float]
    total_rmse: float
    number_of_data_points: int
    number_of_parameters: int
    degrees_of_freedom: int
    number_of_successful_solves: int
    number_of_failed_solves: int
    reproduces_negative_Li_rejection: bool
    reproduces_negative_Li_rejection_in_successful_solve: bool
    rmse_is_best_effort: bool
    parameters_at_bounds: Tuple[str, ...]
    optimizer_status: Optional[int]
    optimizer_nfev: Optional[int]
    identifiability_diagnostics: Mapping[str, object]
    predicted_R_Li_by_flux: Tuple[float, ...] = field(default_factory=tuple)
    predicted_R_Mg_by_flux: Tuple[float, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Parameter application + model evaluation
# ---------------------------------------------------------------------------

def _apply_parameters(base_case, base_variant_config, specs, values):
    """
    Apply a parameter vector to a (case, variant) pair without mutating inputs.

    Returns a deep-copied, modified case and a new variant config. Variant-target
    parameters override variant fields (floors also switch the model to
    ``"floor"``); case-target parameters scale the copied membrane parameters.
    """
    case = copy.deepcopy(base_case)
    variant_overrides = {}
    labels = [base_variant_config.label]

    for spec, value in zip(specs, values):
        value = float(value)
        target = PARAMETER_REGISTRY[spec.name][0]
        if target == "variant":
            if spec.name == "steric_floor":
                variant_overrides["steric_floor"] = value
                variant_overrides["steric_model"] = "floor"
            elif spec.name == "hindrance_floor":
                variant_overrides["hindrance_floor"] = value
                variant_overrides["hindrance_model"] = "floor"
            else:
                variant_overrides[spec.name] = value
        else:  # case
            params = case["membrane_parameters"]
            if spec.name == "fixed_charge_scale":
                params["fixed_charge_mol_L"] = (
                    base_case["membrane_parameters"]["fixed_charge_mol_L"] * value
                )
                if "fixed_charge_mol_m3" in params:
                    params["fixed_charge_mol_m3"] = (
                        base_case["membrane_parameters"]["fixed_charge_mol_m3"] * value
                    )
            elif spec.name == "dielectric_scale":
                params["pore_dielectric_constant"] = (
                    base_case["membrane_parameters"]["pore_dielectric_constant"]
                    * value
                )
        labels.append(f"{spec.name}={value:g}")

    variant = replace(
        base_variant_config, label="|".join(labels), **variant_overrides
    )
    return case, variant


def _evaluate_foo2023(case, variant, solver_options):
    """
    Run the two-interface solver over all four Foo fluxes for a (case, variant).

    Returns a dict of per-flux predictions/success and aggregate diagnostics.
    Experimental values come from the canonical case (unchanged by fitting).
    """
    canonical = foo2023_lmc_ph7_case()
    experimental = canonical["experimental_data"]
    water_fluxes_m_s = [value * 1e-6 for value in experimental["Jw_um_s"]]
    options = (
        solver_options
        if solver_options is not None
        else _DEFAULT_ESTIMATION_SOLVER_OPTIONS
    )

    per_flux = []
    for index, water_flux_m_s in enumerate(water_fluxes_m_s):
        result = solve_two_interface_for_flux(
            case, water_flux_m_s, options=options, variant=variant
        )
        rejections = dict(result.predicted_rejections)
        per_flux.append(
            {
                "success": result.success,
                "R_Li": rejections.get("Li+"),
                "R_Mg": rejections.get("Mg2+"),
                "hard_cutoff_species": tuple(result.hard_cutoff_species),
                "scaled_residual_max_abs": result.residual_max_abs_scaled,
            }
        )

    n_success = sum(1 for row in per_flux if row["success"])
    mg_hard_cutoff_present = any(
        "Mg2+" in row["hard_cutoff_species"] for row in per_flux
    )
    negative_li_any = any(
        row["R_Li"] is not None and row["R_Li"] < 0.0 for row in per_flux
    )
    negative_li_success = any(
        row["R_Li"] is not None and row["R_Li"] < 0.0 and row["success"]
        for row in per_flux
    )

    return {
        "experimental": experimental,
        "per_flux": per_flux,
        "number_successful_solves": n_success,
        "number_failed_solves": len(per_flux) - n_success,
        "Mg_hard_cutoff_present": mg_hard_cutoff_present,
        "reproduces_negative_Li_rejection": negative_li_any,
        "reproduces_negative_Li_rejection_in_successful_solve": negative_li_success,
    }


def _rejection_residuals(evaluation, species_to_fit):
    """Return (labels, residuals) for R_pred - R_exp over species x fluxes."""
    experimental = evaluation["experimental"]
    per_flux = evaluation["per_flux"]
    species_key = {"Li+": "R_Li", "Mg2+": "R_Mg"}

    labels = []
    residuals = []
    for species in species_to_fit:
        exp_key = "R_Li" if species == "Li+" else "R_Mg"
        pred_key = species_key[species]
        for index, row in enumerate(per_flux):
            predicted = row[pred_key]
            experimental_value = experimental[exp_key][index]
            if predicted is None or not math.isfinite(predicted):
                residual = None
            else:
                residual = predicted - experimental_value
            labels.append(f"{species}@flux{index}")
            residuals.append(residual)
    return labels, residuals


def build_foo2023_estimation_objective(config: EstimationConfig):
    """
    Build the least_squares objective callable for a fit configuration.

    Returns
    -------
    tuple
        ``(objective, metadata)`` where ``objective(x)`` returns a fixed-length
        residual vector: the ``species x flux`` rejection residuals followed by
        one penalty slot per flux (``failed_solve_penalty`` when that flux's
        solve did not converge, else 0). ``metadata`` records the base case,
        specs, and species.
    """
    base_case = foo2023_lmc_ph7_case()
    specs = config.parameter_specs
    solver_options = config.solver_options

    def objective(x):
        case, variant = _apply_parameters(
            base_case, config.base_variant_config, specs, x
        )
        evaluation = _evaluate_foo2023(case, variant, solver_options)
        _labels, residuals = _rejection_residuals(evaluation, config.species_to_fit)

        vector = []
        # Rejection residuals (best-effort or penalty-replaced per config).
        species_count = len(config.species_to_fit)
        per_flux = evaluation["per_flux"]
        for position, residual in enumerate(residuals):
            flux_index = position % 4
            success = per_flux[flux_index]["success"]
            if residual is None:
                vector.append(config.failed_solve_penalty)
            elif success or config.include_failed_solves:
                vector.append(residual)
            else:
                vector.append(config.failed_solve_penalty)
        # Per-flux penalty slots (fixed length, honest failure signal).
        for row in per_flux:
            vector.append(0.0 if row["success"] else config.failed_solve_penalty)

        return vector

    metadata = {
        "base_case": base_case,
        "specs": specs,
        "species_to_fit": config.species_to_fit,
        "number_of_data_points": len(config.species_to_fit) * 4,
    }
    return objective, metadata


def _rmse(values):
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def _grid_seed_initial_guess(objective, specs, seed_grid_points):
    """
    Return a deterministic seed for the local optimizer.

    Grids each parameter over its bounds, evaluates the augmented objective's
    sum of squares at every grid point (plus the specs' own initial point), and
    returns the argmin. The per-flux penalty makes converging regions win. With
    ``seed_grid_points < 2`` this just returns the specs' initial values.
    """
    initial = [spec.initial_value for spec in specs]
    if seed_grid_points < 2:
        return initial

    axes = []
    for spec in specs:
        span = spec.upper_bound - spec.lower_bound
        axes.append(
            [
                spec.lower_bound + span * k / (seed_grid_points - 1)
                for k in range(seed_grid_points)
            ]
        )

    def _sum_sq(point):
        vector = objective(list(point))
        return sum(v * v for v in vector)

    best_point = initial
    best_value = _sum_sq(initial)
    for combo in itertools.product(*axes):
        value = _sum_sq(combo)
        if value < best_value:
            best_value = value
            best_point = list(combo)
    return best_point


def _identifiability_diagnostics(config, solution, n_data):
    """Assemble identifiability diagnostics from the config and optimizer state."""
    n_params = config.number_of_parameters
    dof = n_data - n_params
    warnings = list(config.identifiability_warnings())
    if dof <= 0:
        warnings.append("Non-positive degrees of freedom; fit is not constrained.")

    condition_number = None
    parameter_correlation = None
    if solution is not None and getattr(solution, "jac", None) is not None:
        jac = np.asarray(solution.jac, dtype=float)
        if jac.ndim == 2 and jac.shape[1] >= 2 and jac.size and np.all(np.isfinite(jac)):
            singular_values = np.linalg.svd(jac, compute_uv=False)
            smallest = singular_values.min()
            condition_number = (
                float(singular_values.max() / smallest)
                if smallest > 0.0
                else math.inf
            )
            if not math.isfinite(condition_number) or (
                condition_number > CONDITION_NUMBER_WARNING_THRESHOLD
            ):
                warnings.append(
                    "Jacobian is ill-conditioned (condition number "
                    f"{condition_number:.3e}); parameters are weakly identifiable."
                )
            try:
                gram = jac.T @ jac
                covariance = np.linalg.inv(gram)
                deviations = np.sqrt(np.diag(covariance))
                outer = np.outer(deviations, deviations)
                with np.errstate(invalid="ignore", divide="ignore"):
                    correlation = covariance / outer
                parameter_correlation = correlation.tolist()
            except np.linalg.LinAlgError:
                parameter_correlation = None

    return {
        "number_of_data_points": n_data,
        "number_of_parameters": n_params,
        "degrees_of_freedom": dof,
        "jacobian_condition_number": condition_number,
        "parameter_correlation": parameter_correlation,
        "warnings": tuple(warnings),
    }


def fit_foo2023_parameters(config: EstimationConfig) -> EstimationResult:
    """
    Run a bounded least_squares fit of the configured effective parameters.

    Does not mutate ``config`` or the case. A fit is reported successful only if
    the optimizer converged AND at least one model solve converged; a fit whose
    model solves all fail is reported ``success=False``.

    Returns
    -------
    EstimationResult
        Fitted values, RMSE (best-effort, labelled), solve counts, bound-hitting,
        the negative-Li diagnostic, and identifiability diagnostics.
    """
    specs = config.parameter_specs
    objective, metadata = build_foo2023_estimation_objective(config)

    lower = [spec.lower_bound for spec in specs]
    upper = [spec.upper_bound for spec in specs]
    n_data = metadata["number_of_data_points"]

    # Deterministic grid seed to make the local, bounded fit robust to the
    # hard-cutoff discontinuity (documented; not target tuning).
    x0 = _grid_seed_initial_guess(objective, specs, config.seed_grid_points)

    solution = None
    try:
        solution = least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            max_nfev=config.max_nfev,
            ftol=config.tolerance,
            xtol=config.tolerance,
            gtol=config.tolerance,
        )
        fitted_x = [float(v) for v in solution.x]
        optimizer_status = int(solution.status)
        optimizer_nfev = int(solution.nfev)
        optimizer_ok = bool(solution.success)
        optimizer_message = solution.message
    except (ValueError, FloatingPointError, OverflowError) as exc:
        fitted_x = list(x0)
        optimizer_status = None
        optimizer_nfev = None
        optimizer_ok = False
        optimizer_message = f"Optimizer raised: {exc!r}"

    # Evaluate the model at the fitted point for reporting.
    base_case = foo2023_lmc_ph7_case()
    fitted_case, fitted_variant = _apply_parameters(
        base_case, config.base_variant_config, specs, fitted_x
    )
    evaluation = _evaluate_foo2023(fitted_case, fitted_variant, config.solver_options)
    _labels, residuals = _rejection_residuals(evaluation, config.species_to_fit)

    # Best-effort RMSE per species from finite residuals.
    rmse_by_species = {}
    all_finite = []
    for offset, species in enumerate(config.species_to_fit):
        species_residuals = [
            residuals[offset * 4 + i]
            for i in range(4)
            if residuals[offset * 4 + i] is not None
        ]
        rmse_by_species[species] = _rmse(species_residuals)
        all_finite.extend(species_residuals)
    total_rmse = _rmse(all_finite)

    n_success = evaluation["number_successful_solves"]
    n_failed = evaluation["number_failed_solves"]

    parameters_at_bounds = tuple(
        spec.name
        for spec, value in zip(specs, fitted_x)
        if math.isclose(value, spec.lower_bound, abs_tol=1e-9)
        or math.isclose(value, spec.upper_bound, abs_tol=1e-9)
    )

    identifiability = _identifiability_diagnostics(config, solution, n_data)
    if parameters_at_bounds:
        identifiability = dict(identifiability)
        identifiability["warnings"] = identifiability["warnings"] + (
            "Parameter(s) at bounds ("
            + ", ".join(parameters_at_bounds)
            + "); the optimum may lie outside the allowed range.",
        )

    success = optimizer_ok and n_success > 0 and total_rmse is not None
    if success:
        message = (
            f"Fit converged with {n_success}/4 model solves converged; "
            f"total best-effort RMSE {total_rmse:.4f}."
        )
    elif n_success == 0:
        message = (
            "Fit not usable: no model solve converged at the fitted point; "
            "RMSE is best-effort from non-converged solves only."
        )
    else:
        message = (
            f"Fit did not fully converge (optimizer: {optimizer_message}); "
            f"{n_success}/4 model solves converged."
        )

    return EstimationResult(
        success=success,
        message=message,
        parameter_values=_read_only(
            {spec.name: value for spec, value in zip(specs, fitted_x)}
        ),
        initial_parameter_values=_read_only(
            {spec.name: spec.initial_value for spec in specs}
        ),
        bounds=_read_only(
            {spec.name: (spec.lower_bound, spec.upper_bound) for spec in specs}
        ),
        residual_vector=tuple(
            (float("nan") if r is None else float(r)) for r in residuals
        ),
        rmse_by_species=_read_only(rmse_by_species),
        total_rmse=total_rmse,
        number_of_data_points=n_data,
        number_of_parameters=config.number_of_parameters,
        degrees_of_freedom=n_data - config.number_of_parameters,
        number_of_successful_solves=n_success,
        number_of_failed_solves=n_failed,
        reproduces_negative_Li_rejection=evaluation[
            "reproduces_negative_Li_rejection"
        ],
        reproduces_negative_Li_rejection_in_successful_solve=evaluation[
            "reproduces_negative_Li_rejection_in_successful_solve"
        ],
        rmse_is_best_effort=n_success < 4,
        parameters_at_bounds=parameters_at_bounds,
        optimizer_status=optimizer_status,
        optimizer_nfev=optimizer_nfev,
        identifiability_diagnostics=identifiability,
        predicted_R_Li_by_flux=tuple(
            row["R_Li"] for row in evaluation["per_flux"]
        ),
        predicted_R_Mg_by_flux=tuple(
            row["R_Mg"] for row in evaluation["per_flux"]
        ),
    )


# ---------------------------------------------------------------------------
# Batch fits
# ---------------------------------------------------------------------------

def run_foo2023_single_parameter_fits(
    names=None,
    base_variant_config=None,
    solver_options=None,
    max_nfev=200,
) -> Tuple[EstimationResult, ...]:
    """Run a one-parameter fit for each named parameter (default: the standard set)."""
    if names is None:
        names = DEFAULT_SINGLE_PARAMETERS
    if base_variant_config is None:
        base_variant_config = ConstitutiveVariantConfig()

    results = []
    for name in names:
        config = EstimationConfig(
            parameter_specs=(parameter_spec(name),),
            base_variant_config=base_variant_config,
            solver_options=solver_options,
            max_nfev=max_nfev,
        )
        results.append(fit_foo2023_parameters(config))
    return tuple(results)


def run_foo2023_two_parameter_fits(
    pairs=None,
    base_variant_config=None,
    solver_options=None,
    max_nfev=200,
) -> Tuple[EstimationResult, ...]:
    """Run a two-parameter fit for each allowed pair (default: the standard pairs)."""
    if pairs is None:
        pairs = DEFAULT_PARAMETER_PAIRS
    if base_variant_config is None:
        base_variant_config = ConstitutiveVariantConfig()

    results = []
    for name_a, name_b in pairs:
        config = EstimationConfig(
            parameter_specs=(parameter_spec(name_a), parameter_spec(name_b)),
            base_variant_config=base_variant_config,
            solver_options=solver_options,
            max_nfev=max_nfev,
        )
        results.append(fit_foo2023_parameters(config))
    return tuple(results)


def _best_usable(results):
    """Return the usable fit with the smallest finite total RMSE, or None."""
    usable = [
        r for r in results if r.total_rmse is not None and math.isfinite(r.total_rmse)
    ]
    if not usable:
        return None
    return min(usable, key=lambda r: r.total_rmse)


def foo2023_parameter_estimation_summary(
    single_names=None,
    pairs=None,
    base_variant_config=None,
    solver_options=None,
    max_nfev=200,
):
    """
    Run one- and two-parameter fits and return a conservative summary dict.

    The summary states the best one- and two-parameter diagnostic fits, whether
    either reproduces negative Li rejection, whether any fitted parameter sits on
    a bound, whether identifiability is weak, and that the results are effective
    diagnostic parameters only (not validation success).
    """
    single_results = run_foo2023_single_parameter_fits(
        names=single_names,
        base_variant_config=base_variant_config,
        solver_options=solver_options,
        max_nfev=max_nfev,
    )
    pair_results = run_foo2023_two_parameter_fits(
        pairs=pairs,
        base_variant_config=base_variant_config,
        solver_options=solver_options,
        max_nfev=max_nfev,
    )

    best_single = _best_usable(single_results)
    best_pair = _best_usable(pair_results)

    def _describe(result):
        if result is None:
            return None
        return {
            "parameter_values": dict(result.parameter_values),
            "total_rmse": result.total_rmse,
            "rmse_by_species": dict(result.rmse_by_species),
            "success": result.success,
            "rmse_is_best_effort": result.rmse_is_best_effort,
            "number_of_successful_solves": result.number_of_successful_solves,
            "reproduces_negative_Li_rejection": result.reproduces_negative_Li_rejection,
            "parameters_at_bounds": result.parameters_at_bounds,
            "identifiability_warnings": result.identifiability_diagnostics["warnings"],
        }

    any_negative_li = any(
        r.reproduces_negative_Li_rejection
        for r in (single_results + pair_results)
    )
    any_on_bounds = any(
        r.parameters_at_bounds for r in (single_results + pair_results)
    )
    weak_identifiability = any(
        r.identifiability_diagnostics["warnings"]
        for r in (single_results + pair_results)
    )

    return {
        "best_one_parameter_fit": _describe(best_single),
        "best_two_parameter_fit": _describe(best_pair),
        "any_fit_reproduces_negative_Li_rejection": any_negative_li,
        "any_fitted_parameter_on_bounds": any_on_bounds,
        "identifiability_is_weak": weak_identifiability,
        "interpretation": (
            "Fitted values are effective diagnostic parameters only, not unique "
            "physical estimates and not validation success. With only eight "
            "rejection points, one- and two-parameter fits are weakly "
            "identifiable; a lower RMSE that does not reproduce the experimental "
            "negative Li rejection is not evidence of correct physics."
        ),
    }


def foo2023_parameter_estimation_summary_dataframe(
    single_names=None,
    pairs=None,
    base_variant_config=None,
    solver_options=None,
    max_nfev=200,
) -> pd.DataFrame:
    """Return one- and two-parameter fit results as a deterministic DataFrame."""
    single_results = run_foo2023_single_parameter_fits(
        names=single_names,
        base_variant_config=base_variant_config,
        solver_options=solver_options,
        max_nfev=max_nfev,
    )
    pair_results = run_foo2023_two_parameter_fits(
        pairs=pairs,
        base_variant_config=base_variant_config,
        solver_options=solver_options,
        max_nfev=max_nfev,
    )

    rows = []
    for result in single_results + pair_results:
        rows.append(
            {
                "parameters": "+".join(result.parameter_values.keys()),
                "fitted_values": tuple(round(v, 6) for v in result.parameter_values.values()),
                "number_of_parameters": result.number_of_parameters,
                "degrees_of_freedom": result.degrees_of_freedom,
                "total_rmse": result.total_rmse,
                "rmse_is_best_effort": result.rmse_is_best_effort,
                "number_successful_solves": result.number_of_successful_solves,
                "number_failed_solves": result.number_of_failed_solves,
                "reproduces_negative_Li_rejection": result.reproduces_negative_Li_rejection,
                "parameters_at_bounds": result.parameters_at_bounds,
                "success": result.success,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Objective scans
# ---------------------------------------------------------------------------

DEFAULT_SCAN_VALUES = {
    "ion_radius_scale": (0.6, 0.7, 0.8, 0.9, 1.0),
    "pore_radius_scale": (1.0, 1.25, 1.5, 1.75, 2.0),
    "hindrance_floor": (0.0, 0.01, 0.05, 0.1, 0.2),
    "steric_floor": (0.0, 0.01, 0.05, 0.1, 0.2),
    "fixed_charge_scale": (0.5, 0.75, 1.0, 1.5, 2.0),
    "dielectric_scale": (0.6, 0.8, 1.0, 1.2, 1.4),
}


def _scan_metrics_row(name_values, base_variant_config, solver_options):
    """Evaluate the model for a dict of {param_name: value} and return a metrics row."""
    base_case = foo2023_lmc_ph7_case()
    specs = [parameter_spec(name) for name in name_values]
    values = [name_values[spec.name] for spec in specs]
    case, variant = _apply_parameters(base_case, base_variant_config, specs, values)
    evaluation = _evaluate_foo2023(case, variant, solver_options)

    labels, residuals = _rejection_residuals(evaluation, ("Li+", "Mg2+"))
    li_res = [residuals[i] for i in range(4) if residuals[i] is not None]
    mg_res = [residuals[4 + i] for i in range(4) if residuals[4 + i] is not None]

    row = dict(name_values)
    row.update(
        {
            "total_rmse": _rmse(li_res + mg_res),
            "R_Li_rmse": _rmse(li_res),
            "R_Mg_rmse": _rmse(mg_res),
            "number_successful_solves": evaluation["number_successful_solves"],
            "number_failed_solves": evaluation["number_failed_solves"],
            "reproduces_negative_Li_rejection": evaluation[
                "reproduces_negative_Li_rejection"
            ],
            "Mg_hard_cutoff_present": evaluation["Mg_hard_cutoff_present"],
        }
    )
    return row


def objective_scan_1d(
    name,
    values=None,
    base_variant_config=None,
    solver_options=None,
) -> pd.DataFrame:
    """
    Deterministic 1D objective scan over one parameter.

    Returns
    -------
    pandas.DataFrame
        One row per value with the parameter, total/species RMSE, solve counts,
        negative-Li flag, and Mg-hard-cutoff status.
    """
    if name not in PARAMETER_REGISTRY:
        raise ValueError(
            f"Unsupported parameter name {name!r}; supported: {SUPPORTED_PARAMETERS}."
        )
    if values is None:
        values = DEFAULT_SCAN_VALUES[name]
    if base_variant_config is None:
        base_variant_config = ConstitutiveVariantConfig()

    rows = [
        _scan_metrics_row({name: float(value)}, base_variant_config, solver_options)
        for value in values
    ]
    columns = [
        name,
        "total_rmse",
        "R_Li_rmse",
        "R_Mg_rmse",
        "number_successful_solves",
        "number_failed_solves",
        "reproduces_negative_Li_rejection",
        "Mg_hard_cutoff_present",
    ]
    return pd.DataFrame(rows, columns=columns)


def objective_scan_2d(
    name_x,
    name_y,
    values_x=None,
    values_y=None,
    base_variant_config=None,
    solver_options=None,
) -> pd.DataFrame:
    """
    Deterministic 2D objective scan over a parameter pair (full grid).

    Returns
    -------
    pandas.DataFrame
        One row per (x, y) grid point with total/species RMSE, solve counts,
        negative-Li flag, and Mg-hard-cutoff status.
    """
    for name in (name_x, name_y):
        if name not in PARAMETER_REGISTRY:
            raise ValueError(
                f"Unsupported parameter name {name!r}; supported: "
                f"{SUPPORTED_PARAMETERS}."
            )
    if name_x == name_y:
        raise ValueError("objective_scan_2d requires two distinct parameters.")
    if values_x is None:
        values_x = DEFAULT_SCAN_VALUES[name_x]
    if values_y is None:
        values_y = DEFAULT_SCAN_VALUES[name_y]
    if base_variant_config is None:
        base_variant_config = ConstitutiveVariantConfig()

    rows = []
    for value_x in values_x:
        for value_y in values_y:
            rows.append(
                _scan_metrics_row(
                    {name_x: float(value_x), name_y: float(value_y)},
                    base_variant_config,
                    solver_options,
                )
            )
    columns = [
        name_x,
        name_y,
        "total_rmse",
        "R_Li_rmse",
        "R_Mg_rmse",
        "number_successful_solves",
        "number_failed_solves",
        "reproduces_negative_Li_rejection",
        "Mg_hard_cutoff_present",
    ]
    return pd.DataFrame(rows, columns=columns)
