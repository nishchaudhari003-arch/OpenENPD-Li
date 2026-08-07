from openenpd.diagnostics import (
    foo2023_lmc_ph7_ablation_summary,
    foo2023_lmc_ph7_partitioning_diagnostics,
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
