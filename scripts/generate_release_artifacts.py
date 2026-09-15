"""
Generate a small set of reproducible diagnostic artifacts for OpenENPD-Li.

This script rolls up the toolkit's diagnostic outputs for the Foo 2023 LM-C
pH 7 case into a single output directory:

- ``baseline_comparison.csv`` / ``baseline_metrics.json`` -- single-interface
  baseline validation table and RMSE metrics,
- ``two_interface_comparison.csv`` -- two-interface best-effort predictions and
  solver diagnostics per flux,
- ``constitutive_variant_metrics.csv`` -- diagnostic RMSE / solve counts /
  hard-cutoff / negative-Li flags per variant,
- ``parameter_estimation_summary.json`` -- a small one- and two-parameter fit
  summary,
- ``rejection_comparison.png`` / ``two_interface_rejection_comparison.png`` --
  backend-safe diagnostic plots,
- ``SUMMARY.md`` -- a conservative human-readable roll-up.

It uses only repository data, needs no plotting display or network access,
never mutates the source case, and is safe to run repeatedly (it overwrites the
same files). Nothing here tunes parameters or claims validation. To keep the
run fast, the two-interface / variant / estimation steps cap the inner solver's
iteration budget, which bounds work on non-converging solves only.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenpd.constitutive_variants import (  # noqa: E402
    default_variants,
    variant_foo2023_lmc_ph7_metrics,
)
from openenpd.parameter_estimation import (  # noqa: E402
    foo2023_parameter_estimation_summary,
)
from openenpd.plots import (  # noqa: E402
    plot_foo2023_lmc_ph7_rejection_comparison,
    plot_two_interface_foo2023_lmc_ph7_rejection_comparison,
)
from openenpd.two_interface_solver import TwoInterfaceSolverOptions  # noqa: E402
from openenpd.two_interface_validation import (  # noqa: E402
    two_interface_foo2023_lmc_ph7_comparison_dataframe,
)
from openenpd.validation import (  # noqa: E402
    foo2023_lmc_ph7_comparison_dataframe,
    foo2023_lmc_ph7_validation_metrics,
)

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

# Cap the inner solver budget so failing solves do not churn; converging solves
# finish well within this and their outcomes are unchanged.
_FAST_SOLVER_OPTIONS = TwoInterfaceSolverOptions(max_nfev=250)


def _generate_plots(output_directory, options):
    rejection_path = output_directory / "rejection_comparison.png"
    two_interface_path = output_directory / "two_interface_rejection_comparison.png"

    baseline_figure, _ = plot_foo2023_lmc_ph7_rejection_comparison(
        output_path=rejection_path
    )
    two_interface_figure, _ = plot_two_interface_foo2023_lmc_ph7_rejection_comparison(
        options=options, output_path=two_interface_path
    )
    plt.close(baseline_figure)
    plt.close(two_interface_figure)
    return {
        "rejection_comparison": rejection_path,
        "two_interface_rejection_comparison": two_interface_path,
    }


def main(output_directory=None, include_plots=True):
    """
    Generate the diagnostic artifacts and return a mapping of artifact paths.

    Parameters
    ----------
    output_directory : str or pathlib.Path, optional
        Destination directory (created if needed). Defaults to ``outputs/``.
    include_plots : bool
        If True (default), also render the backend-safe diagnostic plots.

    Returns
    -------
    dict
        Artifact name -> written path.
    """
    output_directory = Path(
        DEFAULT_OUTPUT_DIRECTORY if output_directory is None else output_directory
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Baseline single-interface validation.
    baseline_comparison = foo2023_lmc_ph7_comparison_dataframe()
    paths["baseline_comparison"] = output_directory / "baseline_comparison.csv"
    baseline_comparison.to_csv(paths["baseline_comparison"], index=False)

    baseline_metrics = foo2023_lmc_ph7_validation_metrics()
    paths["baseline_metrics"] = output_directory / "baseline_metrics.json"
    with paths["baseline_metrics"].open("w", encoding="utf-8") as handle:
        json.dump(baseline_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # 2. Two-interface validation diagnostics.
    two_interface_comparison = two_interface_foo2023_lmc_ph7_comparison_dataframe(
        options=_FAST_SOLVER_OPTIONS
    )
    paths["two_interface_comparison"] = (
        output_directory / "two_interface_comparison.csv"
    )
    two_interface_comparison.to_csv(paths["two_interface_comparison"], index=False)

    # 3. Constitutive-variant diagnostic metrics.
    variant_metrics = [
        variant_foo2023_lmc_ph7_metrics(config, options=_FAST_SOLVER_OPTIONS)
        for config in default_variants()
    ]
    import pandas as pd  # local import keeps module import light

    variant_frame = pd.DataFrame(variant_metrics)
    paths["constitutive_variant_metrics"] = (
        output_directory / "constitutive_variant_metrics.csv"
    )
    variant_frame.to_csv(paths["constitutive_variant_metrics"], index=False)

    # 4. Small parameter-estimation summary (kept fast).
    estimation_summary = foo2023_parameter_estimation_summary(
        single_names=["pore_radius_scale"],
        pairs=[("ion_radius_scale", "pore_radius_scale")],
        max_nfev=20,
    )
    paths["parameter_estimation_summary"] = (
        output_directory / "parameter_estimation_summary.json"
    )
    with paths["parameter_estimation_summary"].open("w", encoding="utf-8") as handle:
        json.dump(estimation_summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    # 5. Optional plots.
    if include_plots:
        paths.update(_generate_plots(output_directory, _FAST_SOLVER_OPTIONS))

    # 6. Human-readable roll-up.
    best_single = estimation_summary["best_one_parameter_fit"]
    summary_lines = [
        "# OpenENPD-Li diagnostic artifacts (Foo 2023 LM-C pH 7)",
        "",
        "These outputs are diagnostic. No current model reproduces the "
        "experimental negative Li rejection; this is not validation success.",
        "",
        "## Baseline (single interface)",
        f"- Li rejection RMSE: {baseline_metrics['R_Li_rmse']:.4f}",
        f"- Mg rejection RMSE: {baseline_metrics['R_Mg_rmse']:.4f}",
        "- The baseline over-rejects both ions.",
        "",
        "## Two-interface diagnostics",
        f"- Rows written: {len(two_interface_comparison)} (one per flux).",
        "- Under hard steric/hindrance assumptions the solves do not converge; "
        "failed solves remain explicit in the CSV.",
        "",
        "## Constitutive variants",
        f"- Variants compared: {len(variant_frame)}.",
        "- Radius/pore-scaling variants relax the Mg2+ hard cutoff and lower "
        "diagnostic RMSE, but none reproduce negative Li rejection.",
        "",
        "## Parameter estimation",
    ]
    if best_single is not None:
        summary_lines.append(
            "- Best one-parameter diagnostic fit: "
            + ", ".join(
                f"{name}={value:.3f}"
                for name, value in best_single["parameter_values"].items()
            )
            + f" (total RMSE {best_single['total_rmse']:.4f}); effective "
            "diagnostic parameter only, not a unique physical estimate."
        )
    summary_lines.append("")
    paths["summary"] = output_directory / "SUMMARY.md"
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")

    return paths


if __name__ == "__main__":
    generated = main()
    for name, path in generated.items():
        print(f"{name}: {path}")
