import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from openenpd.plots import (
    plot_foo2023_lmc_ph7_rejection_comparison,
    plot_foo2023_lmc_ph7_residuals,
)


@pytest.mark.parametrize(
    "plot_function",
    [
        plot_foo2023_lmc_ph7_rejection_comparison,
        plot_foo2023_lmc_ph7_residuals,
    ],
)
def test_foo2023_lmc_ph7_plot_returns_figure_and_axes(plot_function):
    figure, axes = plot_function()

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)

    plt.close(figure)


@pytest.mark.parametrize(
    ("plot_function", "filename"),
    [
        (plot_foo2023_lmc_ph7_rejection_comparison, "comparison.png"),
        (plot_foo2023_lmc_ph7_residuals, "residuals.png"),
    ],
)
def test_foo2023_lmc_ph7_plot_saves_nonempty_file(
    plot_function,
    filename,
    tmp_path,
):
    output_path = tmp_path / filename

    figure, _ = plot_function(output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(figure)
