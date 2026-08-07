import pandas as pd

from openenpd.diagnostics import (
    foo2023_lmc_ph7_ablation_summary,
    foo2023_lmc_ph7_hindrance_floor_sensitivity,
    foo2023_lmc_ph7_partitioning_diagnostics,
    foo2023_lmc_ph7_radius_sensitivity,
)


def test_partitioning_diagnostics_contains_case_ions():
    diagnostics = foo2023_lmc_ph7_partitioning_diagnostics()

    assert {"Li+", "Mg2+", "Cl-"} == set(diagnostics.index)


def test_partitioning_diagnostics_exposes_required_components():
    diagnostics = foo2023_lmc_ph7_partitioning_diagnostics()
    required_columns = {
        "charge",
        "ion_radius_nm",
        "pore_radius_nm",
        "size_ratio_lambda",
        "steric_partition_factor",
        "dielectric_partition_factor_delta_psi_0",
        "solved_combined_donnan_potential_V",
        "donnan_partition_factor_at_solved_potential",
        "total_partition_factor",
        "bulk_concentration_mol_L",
        "membrane_interface_concentration_mol_L",
        "diffusive_hindrance",
        "convective_hindrance",
    }

    assert required_columns.issubset(diagnostics.columns)


def test_partitioning_diagnostics_hard_cutoff_behavior():
    diagnostics = foo2023_lmc_ph7_partitioning_diagnostics()

    assert diagnostics.loc["Mg2+", "steric_partition_factor"] == 0.0
    assert 0.0 < diagnostics.loc["Li+", "steric_partition_factor"] < 0.01


def test_partitioning_diagnostics_total_factors_are_nonnegative():
    diagnostics = foo2023_lmc_ph7_partitioning_diagnostics()

    assert (diagnostics["total_partition_factor"] >= 0.0).all()


def test_ablation_summary_contains_requested_variants():
    summary = foo2023_lmc_ph7_ablation_summary()
    expected_variants = {
        "Donnan only",
        "Donnan + steric",
        "Donnan + dielectric",
        "Donnan + steric + dielectric",
    }

    assert expected_variants == set(summary.index.get_level_values("variant"))


def test_radius_sensitivity_default_output():
    sensitivity = foo2023_lmc_ph7_radius_sensitivity()
    required_columns = {
        "radius_scale",
        "Li_size_ratio",
        "Mg_size_ratio",
        "Cl_size_ratio",
        "Li_hindrance",
        "Mg_hindrance",
        "Cl_hindrance",
        "R_Li_rmse",
        "R_Mg_rmse",
        "total_rmse",
    }

    assert isinstance(sensitivity, pd.DataFrame)
    assert len(sensitivity) > 1
    assert required_columns.issubset(sensitivity.columns)
    assert (sensitivity["total_rmse"] >= 0.0).all()


def test_smaller_radius_scale_increases_li_and_mg_hindrance():
    sensitivity = foo2023_lmc_ph7_radius_sensitivity([0.8, 1.0]).set_index(
        "radius_scale"
    )

    assert sensitivity.loc[0.8, "Li_hindrance"] > sensitivity.loc[
        1.0, "Li_hindrance"
    ]
    assert sensitivity.loc[0.8, "Mg_hindrance"] > sensitivity.loc[
        1.0, "Mg_hindrance"
    ]


def test_hindrance_floor_sensitivity_default_output():
    sensitivity = foo2023_lmc_ph7_hindrance_floor_sensitivity()
    expected_floors = {0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2}
    required_columns = {
        "hindrance_floor",
        "Li_hindrance_effective",
        "Mg_hindrance_effective",
        "Cl_hindrance_effective",
        "R_Li_rmse",
        "R_Mg_rmse",
        "total_rmse",
    }

    assert isinstance(sensitivity, pd.DataFrame)
    assert required_columns.issubset(sensitivity.columns)
    assert expected_floors == set(sensitivity["hindrance_floor"])


def test_positive_hindrance_floor_increases_mg_effective_hindrance():
    sensitivity = foo2023_lmc_ph7_hindrance_floor_sensitivity(
        [0.0, 0.05]
    ).set_index("hindrance_floor")

    assert sensitivity.loc[0.05, "Mg_hindrance_effective"] > sensitivity.loc[
        0.0, "Mg_hindrance_effective"
    ]
