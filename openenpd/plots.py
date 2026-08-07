"""Plotting utilities for OpenENPD-Li validation workflows."""

import matplotlib.pyplot as plt

from openenpd.validation import foo2023_lmc_ph7_comparison_dataframe


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
