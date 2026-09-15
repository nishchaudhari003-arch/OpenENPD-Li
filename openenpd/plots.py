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
