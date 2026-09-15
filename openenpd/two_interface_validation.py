"""
Two-interface vs. baseline vs. experiment validation diagnostics for OpenENPD-Li.

Milestone 4 of the two-interface roadmap. This module builds a transparent
comparison and diagnostic workflow for the Foo 2023 LM-C pH 7 case that places
three things side by side, per experimental water flux:

1. experimental Foo 2023 LM-C pH 7 rejection data,
2. the existing simplified baseline predictions
   (:func:`openenpd.validation.foo2023_lmc_ph7_comparison_rows`),
3. the minimal two-interface solver predictions and solver diagnostics
   (:func:`openenpd.two_interface_solver.solve_foo2023_lmc_ph7_all_fluxes`).

This is validation **reporting and diagnostics only**. It does not modify any
solver physics, does not tune or fit parameters, and does not add new cases.

Honesty guarantees
------------------
- Failed two-interface solves are **never hidden**: every row carries
  ``two_interface_success`` and the solver message. Best-effort predicted
  rejections from a failed solve are reported but always accompanied by
  ``success=False``.
- "Best-effort" RMSE (over all fluxes regardless of convergence) is a
  *diagnostic* number, clearly named as such. "Successful-only" RMSE is
  reported separately and is ``None`` when there are zero successful solves.
- The solver diagnostic summary explicitly records whether any two-interface
  prediction reproduces the experimental negative Li rejection.

Nothing here claims the two-interface model is validated or that it reproduces
Foo 2023.
"""

import math

import pandas as pd

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.two_interface_solver import (
    TwoInterfaceSolverOptions,
    solve_foo2023_lmc_ph7_all_fluxes,
)
from openenpd.validation import (
    foo2023_lmc_ph7_comparison_rows as _baseline_comparison_rows,
)

__all__ = [
    "COMPARISON_COLUMNS",
    "two_interface_foo2023_lmc_ph7_comparison_rows",
    "two_interface_foo2023_lmc_ph7_comparison_dataframe",
    "two_interface_foo2023_lmc_ph7_metrics",
    "two_interface_foo2023_lmc_ph7_solver_diagnostics",
]


# Deterministic column order for the comparison DataFrame.
COMPARISON_COLUMNS = (
    "pressure_bar",
    "Jw_LMH",
    "Jw_um_s",
    "experimental_R_Li",
    "experimental_R_Mg",
    "baseline_R_Li",
    "baseline_R_Mg",
    "two_interface_success",
    "two_interface_message",
    "two_interface_R_Li",
    "two_interface_R_Mg",
    "two_interface_scaled_residual_norm",
    "two_interface_scaled_residual_max_abs",
    "two_interface_raw_residual_norm",
    "hard_cutoff_species",
)


def _rmse(errors):
    """
    Root-mean-square of a list of errors, or ``None`` for an empty list.

    Parameters
    ----------
    errors : list of float
        Prediction-minus-experiment errors.

    Returns
    -------
    float or None
        RMSE, or ``None`` if ``errors`` is empty.
    """
    if not errors:
        return None
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def two_interface_foo2023_lmc_ph7_comparison_rows(options=None):
    """
    Build one comparison row per Foo 2023 LM-C pH 7 experimental water flux.

    Each row carries the experimental rejections, the simplified baseline
    predictions, and the two-interface solver's best-effort predictions plus
    solver diagnostics. Failed solves remain explicit (``two_interface_success``
    is ``False`` with the solver message and best-effort rejections).

    Parameters
    ----------
    options : TwoInterfaceSolverOptions, optional
        Options forwarded to the two-interface solver. Defaults to
        ``TwoInterfaceSolverOptions()``.

    Returns
    -------
    list of dict
        One row per water flux, with the keys in :data:`COMPARISON_COLUMNS`.
    """
    if options is None:
        options = TwoInterfaceSolverOptions()

    case = foo2023_lmc_ph7_case()
    experimental = case["experimental_data"]

    baseline_rows = _baseline_comparison_rows()
    two_interface_results = solve_foo2023_lmc_ph7_all_fluxes(options)

    n_fluxes = len(experimental["Jw_um_s"])
    rows = []

    for index in range(n_fluxes):
        baseline_row = baseline_rows[index]
        result = two_interface_results[index]
        rejections = dict(result.predicted_rejections)

        rows.append(
            {
                "pressure_bar": experimental["pressure_bar"][index],
                "Jw_LMH": experimental["Jw_LMH"][index],
                "Jw_um_s": experimental["Jw_um_s"][index],
                "experimental_R_Li": experimental["R_Li"][index],
                "experimental_R_Mg": experimental["R_Mg"][index],
                "baseline_R_Li": baseline_row["R_Li_pred"],
                "baseline_R_Mg": baseline_row["R_Mg_pred"],
                "two_interface_success": result.success,
                "two_interface_message": result.message,
                "two_interface_R_Li": rejections.get("Li+"),
                "two_interface_R_Mg": rejections.get("Mg2+"),
                "two_interface_scaled_residual_norm": result.residual_norm_scaled,
                "two_interface_scaled_residual_max_abs": result.residual_max_abs_scaled,
                "two_interface_raw_residual_norm": result.residual_norm_raw,
                "hard_cutoff_species": tuple(result.hard_cutoff_species),
            }
        )

    return rows


def two_interface_foo2023_lmc_ph7_comparison_dataframe(options=None):
    """
    Return the comparison rows as a deterministic pandas DataFrame.

    Columns are ordered as in :data:`COMPARISON_COLUMNS`.

    Parameters
    ----------
    options : TwoInterfaceSolverOptions, optional
        Options forwarded to the two-interface solver.

    Returns
    -------
    pandas.DataFrame
        One row per water flux.
    """
    rows = two_interface_foo2023_lmc_ph7_comparison_rows(options)
    return pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))


def two_interface_foo2023_lmc_ph7_metrics(options=None):
    """
    Compute comparison metrics for baseline and two-interface predictions.

    Three families of RMSE are reported against experiment:

    - ``baseline_*``: simplified baseline vs experiment (always available).
    - ``two_interface_best_effort_*``: two-interface predictions over *all*
      fluxes regardless of convergence. This is a **diagnostic** number, not a
      validation result, because most/all Foo solves may be non-converged.
    - ``two_interface_successful_*``: two-interface predictions over converged
      solves only. ``None`` when there are zero successful solves.

    Parameters
    ----------
    options : TwoInterfaceSolverOptions, optional
        Options forwarded to the two-interface solver.

    Returns
    -------
    dict
        RMSE metrics, best-effort/successful splits, and solve counts.
    """
    rows = two_interface_foo2023_lmc_ph7_comparison_rows(options)

    baseline_li_errors = [r["baseline_R_Li"] - r["experimental_R_Li"] for r in rows]
    baseline_mg_errors = [r["baseline_R_Mg"] - r["experimental_R_Mg"] for r in rows]

    best_effort_li_errors = [
        r["two_interface_R_Li"] - r["experimental_R_Li"]
        for r in rows
        if r["two_interface_R_Li"] is not None
    ]
    best_effort_mg_errors = [
        r["two_interface_R_Mg"] - r["experimental_R_Mg"]
        for r in rows
        if r["two_interface_R_Mg"] is not None
    ]

    successful_rows = [r for r in rows if r["two_interface_success"]]
    successful_li_errors = [
        r["two_interface_R_Li"] - r["experimental_R_Li"]
        for r in successful_rows
        if r["two_interface_R_Li"] is not None
    ]
    successful_mg_errors = [
        r["two_interface_R_Mg"] - r["experimental_R_Mg"]
        for r in successful_rows
        if r["two_interface_R_Mg"] is not None
    ]

    number_successful = len(successful_rows)

    return {
        "case_id": "foo2023_lmc_ph7",
        "baseline_R_Li_rmse": _rmse(baseline_li_errors),
        "baseline_R_Mg_rmse": _rmse(baseline_mg_errors),
        "baseline_total_rmse": _rmse(baseline_li_errors + baseline_mg_errors),
        "two_interface_best_effort_R_Li_rmse": _rmse(best_effort_li_errors),
        "two_interface_best_effort_R_Mg_rmse": _rmse(best_effort_mg_errors),
        "two_interface_best_effort_total_rmse": _rmse(
            best_effort_li_errors + best_effort_mg_errors
        ),
        "two_interface_successful_R_Li_rmse": _rmse(successful_li_errors),
        "two_interface_successful_R_Mg_rmse": _rmse(successful_mg_errors),
        "two_interface_successful_total_rmse": _rmse(
            successful_li_errors + successful_mg_errors
        )
        if number_successful > 0
        else None,
        "number_of_fluxes": len(rows),
        "number_of_successful_two_interface_solves": number_successful,
        "number_of_failed_two_interface_solves": len(rows) - number_successful,
        "best_effort_is_diagnostic_only": True,
    }


def two_interface_foo2023_lmc_ph7_solver_diagnostics(options=None):
    """
    Summarize the two-interface solver behaviour across all Foo 2023 fluxes.

    The summary is deliberately conservative and honest: it reports solve
    counts, residual ranges, per-flux residual norms and hard-cutoff species,
    and whether the experimental negative Li rejection was reproduced by any
    two-interface prediction (best-effort) and by any *successful* solve.

    Parameters
    ----------
    options : TwoInterfaceSolverOptions, optional
        Options forwarded to the two-interface solver.

    Returns
    -------
    dict
        Solver diagnostic summary.
    """
    rows = two_interface_foo2023_lmc_ph7_comparison_rows(options)

    successful = [r for r in rows if r["two_interface_success"]]
    failed = [r for r in rows if not r["two_interface_success"]]

    scaled_max_by_flux = [
        r["two_interface_scaled_residual_max_abs"] for r in rows
    ]
    finite_scaled_max = [v for v in scaled_max_by_flux if math.isfinite(v)]

    # Experimental Li rejection is negative for this case; did any two-interface
    # prediction actually produce a negative Li rejection?
    negative_li_any = any(
        r["two_interface_R_Li"] is not None and r["two_interface_R_Li"] < 0.0
        for r in rows
    )
    negative_li_successful = any(
        r["two_interface_R_Li"] is not None and r["two_interface_R_Li"] < 0.0
        for r in successful
    )

    if successful:
        success_text = (
            f"{len(successful)} of {len(rows)} two-interface solves converged "
            "to the scaled tolerance."
        )
    else:
        success_text = (
            "No two-interface solves converged to the scaled tolerance for this "
            "case; all reported two-interface rejections are best-effort "
            "diagnostics from non-converged solves, not validated predictions."
        )

    negative_text = (
        "At least one two-interface prediction reproduced negative Li rejection."
        if negative_li_any
        else "No two-interface prediction reproduced the experimental negative "
        "Li rejection."
    )

    return {
        "case_id": "foo2023_lmc_ph7",
        "number_of_fluxes": len(rows),
        "number_of_successful_solves": len(successful),
        "number_of_failed_solves": len(failed),
        "scaled_residual_max_abs_by_flux": tuple(scaled_max_by_flux),
        "scaled_residual_norm_by_flux": tuple(
            r["two_interface_scaled_residual_norm"] for r in rows
        ),
        "raw_residual_norm_by_flux": tuple(
            r["two_interface_raw_residual_norm"] for r in rows
        ),
        "scaled_residual_max_abs_range": (
            (min(finite_scaled_max), max(finite_scaled_max))
            if finite_scaled_max
            else (None, None)
        ),
        "hard_cutoff_species_by_flux": tuple(
            tuple(r["hard_cutoff_species"]) for r in rows
        ),
        "negative_li_rejection_reproduced": negative_li_any,
        "negative_li_rejection_in_successful_solve": negative_li_successful,
        "summary_text": " ".join([success_text, negative_text]),
    }
