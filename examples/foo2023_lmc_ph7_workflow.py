"""
End-to-end diagnostic workflow for the Foo 2023 LM-C pH 7 case.

This example runs a small, fast, deterministic tour of the toolkit and prints
concise summaries:

1. baseline (single-interface) ENP-Donnan validation metrics,
2. two-interface solver validation diagnostics (solve counts),
3. a small constitutive-variant comparison (hard baseline vs a diagnostic
   pore-radius-scaled variant),
4. a small parameter-estimation summary (one one-parameter fit and one
   two-parameter fit).

It uses only data already in the repository, needs no plotting display, and is
written to run quickly. Nothing here tunes parameters for agreement or claims
validation success; the two-interface results are diagnostic. To keep the run
fast, the two-interface/variant/estimation steps cap the inner solver's
iteration budget -- this bounds work on non-converging solves only and does not
change which solves converge or the baseline validation numbers.

Run directly::

    python examples/foo2023_lmc_ph7_workflow.py

or import and call :func:`main`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenpd.constitutive_variants import (  # noqa: E402
    ConstitutiveVariantConfig,
    variant_foo2023_lmc_ph7_metrics,
)
from openenpd.parameter_estimation import (  # noqa: E402
    EstimationConfig,
    fit_foo2023_parameters,
    parameter_spec,
)
from openenpd.two_interface_solver import TwoInterfaceSolverOptions  # noqa: E402
from openenpd.two_interface_validation import (  # noqa: E402
    two_interface_foo2023_lmc_ph7_solver_diagnostics,
)
from openenpd.validation import foo2023_lmc_ph7_validation_metrics  # noqa: E402

# A modest inner-solver budget keeps this example quick without altering which
# solves converge (converging solves finish in far fewer iterations).
_FAST_SOLVER_OPTIONS = TwoInterfaceSolverOptions(max_nfev=250)


def _format(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main(verbose=True):
    """
    Run the small end-to-end diagnostic workflow and return a summary dict.

    Parameters
    ----------
    verbose : bool
        If True (default), print concise summaries to stdout.

    Returns
    -------
    dict
        Baseline metrics, two-interface diagnostics, variant comparison, and a
        small parameter-estimation summary.
    """
    lines = []

    # 1. Baseline (single-interface) validation metrics.
    baseline_metrics = foo2023_lmc_ph7_validation_metrics()
    lines.append("Baseline ENP-Donnan validation (single interface):")
    lines.append(
        f"  Li RMSE = {_format(baseline_metrics['R_Li_rmse'])}, "
        f"Mg RMSE = {_format(baseline_metrics['R_Mg_rmse'])} "
        "(baseline over-rejects both ions)."
    )

    # 2. Two-interface solver validation diagnostics.
    ti_diagnostics = two_interface_foo2023_lmc_ph7_solver_diagnostics(
        options=_FAST_SOLVER_OPTIONS
    )
    lines.append("")
    lines.append("Two-interface solver diagnostics (hard baseline physics):")
    lines.append(
        f"  successful solves = {ti_diagnostics['number_of_successful_solves']}/"
        f"{ti_diagnostics['number_of_fluxes']}, "
        f"negative Li reproduced = {ti_diagnostics['negative_li_rejection_reproduced']}."
    )

    # 3. Small constitutive-variant comparison.
    hard_config = ConstitutiveVariantConfig(label="hard_baseline")
    pore_config = ConstitutiveVariantConfig(
        pore_radius_scale=1.25, label="pore_radius_scale_1p25"
    )
    variant_metrics = {
        config.label: variant_foo2023_lmc_ph7_metrics(
            config, options=_FAST_SOLVER_OPTIONS
        )
        for config in (hard_config, pore_config)
    }
    lines.append("")
    lines.append("Constitutive variant comparison (diagnostic):")
    for label, metrics in variant_metrics.items():
        lines.append(
            f"  {label}: total RMSE = {_format(metrics['total_rmse'])}, "
            f"successful solves = {metrics['number_successful_solves']}/4, "
            f"Mg hard cutoff = {metrics['Mg_hard_cutoff_present']}, "
            f"negative Li = {metrics['reproduces_negative_Li_rejection']}."
        )

    # 4. Small parameter-estimation summary (kept fast on purpose).
    single_result = fit_foo2023_parameters(
        EstimationConfig(
            parameter_specs=(parameter_spec("pore_radius_scale"),),
            seed_grid_points=3,
            max_nfev=20,
        )
    )
    pair_result = fit_foo2023_parameters(
        EstimationConfig(
            parameter_specs=(
                parameter_spec("ion_radius_scale"),
                parameter_spec("pore_radius_scale"),
            ),
            seed_grid_points=3,
            max_nfev=20,
        )
    )
    lines.append("")
    lines.append("Parameter estimation (diagnostic, effective parameters only):")
    lines.append(
        "  one-parameter: "
        + ", ".join(
            f"{name}={value:.3f}"
            for name, value in single_result.parameter_values.items()
        )
        + f" -> total RMSE = {_format(single_result.total_rmse)}, "
        f"negative Li = {single_result.reproduces_negative_Li_rejection}."
    )
    lines.append(
        "  two-parameter: "
        + ", ".join(
            f"{name}={value:.3f}"
            for name, value in pair_result.parameter_values.items()
        )
        + f" -> total RMSE = {_format(pair_result.total_rmse)}, "
        f"weak identifiability warnings = "
        f"{len(pair_result.identifiability_diagnostics['warnings'])}."
    )
    lines.append("")
    lines.append(
        "Note: results are diagnostic. No current model reproduces the "
        "experimental negative Li rejection; this is not validation success."
    )

    if verbose:
        print("\n".join(lines))

    return {
        "baseline_metrics": baseline_metrics,
        "two_interface_diagnostics": ti_diagnostics,
        "variant_metrics": variant_metrics,
        "parameter_estimation": {
            "one_parameter": single_result,
            "two_parameter": pair_result,
        },
        "report_lines": lines,
    }


if __name__ == "__main__":
    main()
