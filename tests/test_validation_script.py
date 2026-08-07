import json

import pandas as pd

from scripts.run_foo2023_lmc_ph7_validation import (
    generate_foo2023_lmc_ph7_validation_outputs,
)


def test_generate_foo2023_lmc_ph7_validation_outputs(tmp_path):
    output_paths = generate_foo2023_lmc_ph7_validation_outputs(tmp_path)

    assert set(output_paths) == {
        "comparison",
        "metrics",
        "rejection_comparison",
        "residuals",
    }
    for output_path in output_paths.values():
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    dataframe = pd.read_csv(output_paths["comparison"])
    assert len(dataframe) == 4

    with output_paths["metrics"].open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    assert {"case_id", "R_Li_rmse", "R_Mg_rmse"}.issubset(metrics)
