"""Plotting utilities for OpenENPD-Li validation workflows."""

import matplotlib.pyplot as plt

from openenpd.validation import foo2023_lmc_ph7_comparison_dataframe
from openenpd.two_interface_validation import (
    two_interface_foo2023_lmc_ph7_comparison_dataframe,
)


def plot_foo2023_lmc_ph7_rejection_comparison(output_path=None):
    """Plot experimental and predicted Li/Mg rejection against water flux."""
    dataframe = foo2023_lmc_ph7_comparison_dataframe()
    figure, axes = plt.subplots(figsize=(7, 4.5))

    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Li_exp"],
        "o-",
        label="Li experimental",
    )
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Li_pred"],
        "o--",
        label="Li predicted",
    )
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Mg_exp"],
        "s-",
        label="Mg experimental",
    )
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Mg_pred"],
        "s--",
        label="Mg predicted",
    )
    axes.set_xlabel("Water flux, $J_w$ (LMH)")
    axes.set_ylabel("Rejection")
    axes.set_title("Foo 2023 LM-C pH ~7 rejection comparison")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_foo2023_lmc_ph7_residuals(output_path=None):
    """Plot Li/Mg prediction residuals against water flux."""
    dataframe = foo2023_lmc_ph7_comparison_dataframe()
    figure, axes = plt.subplots(figsize=(7, 4.5))

    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Li_residual"],
        "o-",
        label="Li residual",
    )
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["R_Mg_residual"],
        "s-",
        label="Mg residual",
    )
    axes.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes.set_xlabel("Water flux, $J_w$ (LMH)")
    axes.set_ylabel("Predicted - experimental rejection")
    axes.set_title("Foo 2023 LM-C pH ~7 rejection residuals")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_two_interface_foo2023_lmc_ph7_rejection_comparison(
    options=None,
    output_path=None,
):
    """
    Plot experiment vs baseline vs two-interface best-effort rejection.

    The two-interface series is best-effort (includes non-converged solves) and
    is labelled as such; no validation claim is made in the title.
    """
    dataframe = two_interface_foo2023_lmc_ph7_comparison_dataframe(options)
    figure, axes = plt.subplots(figsize=(7.5, 4.5))

    axes.plot(dataframe["Jw_LMH"], dataframe["experimental_R_Li"], "o-", label="Li experiment")
    axes.plot(dataframe["Jw_LMH"], dataframe["baseline_R_Li"], "o--", label="Li baseline")
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["two_interface_R_Li"],
        "o:",
        label="Li two-interface (best-effort)",
    )
    axes.plot(dataframe["Jw_LMH"], dataframe["experimental_R_Mg"], "s-", label="Mg experiment")
    axes.plot(dataframe["Jw_LMH"], dataframe["baseline_R_Mg"], "s--", label="Mg baseline")
    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["two_interface_R_Mg"],
        "s:",
        label="Mg two-interface (best-effort)",
    )
    axes.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes.set_xlabel("Water flux, $J_w$ (LMH)")
    axes.set_ylabel("Rejection")
    axes.set_title("Foo 2023 LM-C pH ~7: experiment vs baseline vs two-interface (diagnostic)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=8, ncol=2)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_two_interface_foo2023_lmc_ph7_scaled_residual(
    options=None,
    output_path=None,
):
    """Plot the two-interface maximum absolute scaled residual against water flux."""
    dataframe = two_interface_foo2023_lmc_ph7_comparison_dataframe(options)
    figure, axes = plt.subplots(figsize=(7, 4.5))

    axes.plot(
        dataframe["Jw_LMH"],
        dataframe["two_interface_scaled_residual_max_abs"],
        "o-",
        label="max abs scaled residual",
    )
    axes.set_xlabel("Water flux, $J_w$ (LMH)")
    axes.set_ylabel("Max absolute scaled residual")
    axes.set_title("Foo 2023 LM-C pH ~7: two-interface scaled residual (diagnostic)")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_two_interface_foo2023_lmc_ph7_solve_success(
    options=None,
    output_path=None,
):
    """Plot two-interface solve success (1) / failure (0) against water flux."""
    dataframe = two_interface_foo2023_lmc_ph7_comparison_dataframe(options)
    figure, axes = plt.subplots(figsize=(7, 3.5))

    success_numeric = [1 if value else 0 for value in dataframe["two_interface_success"]]
    axes.scatter(dataframe["Jw_LMH"], success_numeric, s=80)
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["failed", "converged"])
    axes.set_ylim(-0.5, 1.5)
    axes.set_xlabel("Water flux, $J_w$ (LMH)")
    axes.set_title("Foo 2023 LM-C pH ~7: two-interface solve success by flux (diagnostic)")
    axes.grid(alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def _nan_if_none(value):
    return float("nan") if value is None else value


def plot_constitutive_variant_total_rmse(metrics, output_path=None):
    """
    Bar chart of total (Li+Mg) rejection RMSE by constitutive variant.

    ``metrics`` is a sequence of dicts from
    ``constitutive_variants.variant_foo2023_lmc_ph7_metrics``. RMSE is
    best-effort/diagnostic for variants whose solves did not all converge.
    """
    labels = [m["variant_label"] for m in metrics]
    totals = [_nan_if_none(m["total_rmse"]) for m in metrics]

    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.bar(range(len(labels)), totals)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel("Total rejection RMSE (diagnostic)")
    axes.set_title("Foo 2023 LM-C pH ~7: total RMSE by constitutive variant (diagnostic)")
    axes.grid(alpha=0.3, axis="y")
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_constitutive_variant_li_mg_rmse(metrics, output_path=None):
    """Grouped bar chart of Li and Mg rejection RMSE by constitutive variant."""
    labels = [m["variant_label"] for m in metrics]
    li = [_nan_if_none(m["R_Li_rmse"]) for m in metrics]
    mg = [_nan_if_none(m["R_Mg_rmse"]) for m in metrics]

    positions = range(len(labels))
    width = 0.4

    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.bar([p - width / 2 for p in positions], li, width=width, label="Li RMSE")
    axes.bar([p + width / 2 for p in positions], mg, width=width, label="Mg RMSE")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel("Rejection RMSE (diagnostic)")
    axes.set_title("Foo 2023 LM-C pH ~7: Li/Mg RMSE by constitutive variant (diagnostic)")
    axes.grid(alpha=0.3, axis="y")
    axes.legend()
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_constitutive_variant_mg_hard_cutoff(metrics, output_path=None):
    """Bar chart of Mg2+ hard-cutoff status (1 present / 0 relaxed) by variant."""
    labels = [m["variant_label"] for m in metrics]
    status = [1 if m["Mg_hard_cutoff_present"] else 0 for m in metrics]

    figure, axes = plt.subplots(figsize=(8, 4.0))
    axes.bar(range(len(labels)), status)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["relaxed", "hard cutoff"])
    axes.set_ylim(-0.5, 1.5)
    axes.set_title("Foo 2023 LM-C pH ~7: Mg2+ hard-cutoff status by variant (diagnostic)")
    axes.grid(alpha=0.3, axis="y")
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    return figure, axes
