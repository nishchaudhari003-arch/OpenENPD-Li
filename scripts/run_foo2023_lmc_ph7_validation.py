"""Generate reproducible Foo 2023 LM-C pH ~7 validation artifacts."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openenpd.plots import (  # noqa: E402
    plot_foo2023_lmc_ph7_rejection_comparison,
    plot_foo2023_lmc_ph7_residuals,
)
from openenpd.validation import (  # noqa: E402
    foo2023_lmc_ph7_comparison_dataframe,
    foo2023_lmc_ph7_validation_metrics,
)

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "foo2023_lmc_ph7"


def generate_foo2023_lmc_ph7_validation_outputs(
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
):
    """Generate tabular, metric, and plot artifacts for the validation case."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "comparison": output_directory / "comparison.csv",
        "metrics": output_directory / "metrics.json",
        "rejection_comparison": output_directory / "rejection_comparison.png",
        "residuals": output_directory / "residuals.png",
    }

    dataframe = foo2023_lmc_ph7_comparison_dataframe()
    dataframe.to_csv(output_paths["comparison"], index=False)

    metrics = foo2023_lmc_ph7_validation_metrics()
    with output_paths["metrics"].open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    comparison_figure, _ = plot_foo2023_lmc_ph7_rejection_comparison(
        output_paths["rejection_comparison"]
    )
    residuals_figure, _ = plot_foo2023_lmc_ph7_residuals(
        output_paths["residuals"]
    )
    plt.close(comparison_figure)
    plt.close(residuals_figure)

    return output_paths


if __name__ == "__main__":
    generated_paths = generate_foo2023_lmc_ph7_validation_outputs()
    for artifact_name, artifact_path in generated_paths.items():
        print(f"{artifact_name}: {artifact_path}")
