"""Diagnostic ablations for published OpenENPD-Li validation cases."""

import pandas as pd
from scipy.optimize import brentq

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.dielectric import dielectric_partition_factor
from openenpd.donnan import donnan_partition_factor
from openenpd.hindrance import hindrance_factors
from openenpd.interface import compute_interface_state
from openenpd.steric import size_ratio, steric_partition_factor


def foo2023_lmc_ph7_partitioning_diagnostics():
    """Return per-ion component diagnostics for the combined partition model."""
    case = foo2023_lmc_ph7_case()
    parameters = case["membrane_parameters"]
    pore_radius_nm = parameters["pore_radius_nm"]
    temperature_K = case["temperature_K"]
    interface_state = compute_interface_state(case)
    solved_potential_V = interface_state["delta_psi_V"]
    hindrance = hindrance_factors(
        ion_radii_nm=case["ion_radii_nm"],
        pore_radius_nm=pore_radius_nm,
    )

    rows = []
    for ion, charge in case["charges"].items():
        ion_radius_nm = case["ion_radii_nm"][ion]
        rows.append(
            {
                "ion": ion,
                "charge": charge,
                "ion_radius_nm": ion_radius_nm,
                "pore_radius_nm": pore_radius_nm,
                "size_ratio_lambda": size_ratio(
                    ion_radius_nm=ion_radius_nm,
                    pore_radius_nm=pore_radius_nm,
                ),
                "steric_partition_factor": steric_partition_factor(
                    ion_radius_nm=ion_radius_nm,
                    pore_radius_nm=pore_radius_nm,
                ),
                "dielectric_partition_factor_delta_psi_0": (
                    dielectric_partition_factor(
                        charge=charge,
                        ion_radius_nm=ion_radius_nm,
                        pore_dielectric_constant=parameters[
                            "pore_dielectric_constant"
                        ],
                        temperature_K=temperature_K,
                    )
                ),
                "solved_combined_donnan_potential_V": solved_potential_V,
                "donnan_partition_factor_at_solved_potential": (
                    donnan_partition_factor(
                        charge=charge,
                        delta_psi_V=solved_potential_V,
                        temperature_K=temperature_K,
                    )
                ),
                "total_partition_factor": interface_state["partition_factors"][
                    ion
                ],
                "bulk_concentration_mol_L": case[
                    "bulk_concentrations_mol_L"
                ][ion],
                "membrane_interface_concentration_mol_L": interface_state[
                    "membrane_concentrations_mol_L"
                ][ion],
                "diffusive_hindrance": hindrance["diffusive"][ion],
                "convective_hindrance": hindrance["convective"][ion],
            }
        )

    return pd.DataFrame(rows).set_index("ion")


def foo2023_lmc_ph7_ablation_summary():
    """Compare self-consistent partitioning across four component ablations."""
    case = foo2023_lmc_ph7_case()
    parameters = case["membrane_parameters"]
    temperature_K = case["temperature_K"]
    diagnostics = foo2023_lmc_ph7_partitioning_diagnostics()
    variants = (
        ("Donnan only", False, False),
        ("Donnan + steric", True, False),
        ("Donnan + dielectric", False, True),
        ("Donnan + steric + dielectric", True, True),
    )

    rows = []
    for variant, include_steric, include_dielectric in variants:
        solved_potential_V = brentq(
            lambda delta_psi_V: sum(
                case["charges"][ion]
                * case["bulk_concentrations_mol_L"][ion]
                * donnan_partition_factor(
                    charge=case["charges"][ion],
                    delta_psi_V=delta_psi_V,
                    temperature_K=temperature_K,
                )
                * (
                    diagnostics.loc[ion, "steric_partition_factor"]
                    if include_steric
                    else 1.0
                )
                * (
                    diagnostics.loc[
                        ion,
                        "dielectric_partition_factor_delta_psi_0",
                    ]
                    if include_dielectric
                    else 1.0
                )
                for ion in case["charges"]
            )
            + parameters["fixed_charge_mol_L"],
            -0.5,
            0.5,
        )

        for ion, charge in case["charges"].items():
            donnan_factor = donnan_partition_factor(
                charge=charge,
                delta_psi_V=solved_potential_V,
                temperature_K=temperature_K,
            )
            steric_factor = (
                diagnostics.loc[ion, "steric_partition_factor"]
                if include_steric
                else 1.0
            )
            dielectric_factor = (
                diagnostics.loc[
                    ion,
                    "dielectric_partition_factor_delta_psi_0",
                ]
                if include_dielectric
                else 1.0
            )
            total_factor = donnan_factor * steric_factor * dielectric_factor
            bulk_concentration = case["bulk_concentrations_mol_L"][ion]

            rows.append(
                {
                    "variant": variant,
                    "ion": ion,
                    "donnan_potential_V": solved_potential_V,
                    "donnan_partition_factor": donnan_factor,
                    "steric_partition_factor": steric_factor,
                    "dielectric_partition_factor": dielectric_factor,
                    "total_partition_factor": total_factor,
                    "bulk_concentration_mol_L": bulk_concentration,
                    "membrane_interface_concentration_mol_L": (
                        bulk_concentration * total_factor
                    ),
                }
            )

    return pd.DataFrame(rows).set_index(["variant", "ion"])
