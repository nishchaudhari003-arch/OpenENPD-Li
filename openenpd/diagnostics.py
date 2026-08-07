"""Diagnostic ablations for published OpenENPD-Li validation cases."""

from copy import deepcopy

import pandas as pd
from scipy.optimize import brentq

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.dielectric import dielectric_partition_factor
from openenpd.donnan import donnan_partition_factor
from openenpd.hindrance import hindrance_factors
from openenpd.interface import compute_interface_state
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejections_for_fluxes
from openenpd.steric import size_ratio, steric_partition_factor
from openenpd.validation import rejection_rmse


def _foo2023_lmc_ph7_rmse_for_model_inputs(case, model_inputs):
    """Return Li, Mg, and combined rejection RMSE for prepared model inputs."""
    predictions = predict_rejections_for_fluxes(
        model_inputs=model_inputs,
        water_fluxes_m_s=model_inputs["water_fluxes_m_s"],
        temperature_K=case["temperature_K"],
    )
    experimental = case["experimental_data"]
    rows = []

    for index, prediction in enumerate(predictions):
        rows.append(
            {
                "R_Li_exp": experimental["R_Li"][index],
                "R_Li_pred": prediction["rejections"]["Li+"],
                "R_Mg_exp": experimental["R_Mg"][index],
                "R_Mg_pred": prediction["rejections"]["Mg2+"],
            }
        )

    li_rmse = rejection_rmse(rows, "R_Li_exp", "R_Li_pred")
    mg_rmse = rejection_rmse(rows, "R_Mg_exp", "R_Mg_pred")
    total_rmse = ((li_rmse**2 + mg_rmse**2) / 2.0) ** 0.5

    return li_rmse, mg_rmse, total_rmse


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


def foo2023_lmc_ph7_radius_sensitivity(radius_scale_values=None):
    """Evaluate validation RMSE while uniformly scaling effective ion radii."""
    if radius_scale_values is None:
        radius_scale_values = [
            round(0.40 + 0.05 * index, 2) for index in range(13)
        ]

    baseline_case = foo2023_lmc_ph7_case()
    baseline_radii = baseline_case["ion_radii_nm"]
    pore_radius_nm = baseline_case["membrane_parameters"]["pore_radius_nm"]
    rows = []

    for radius_scale in radius_scale_values:
        radius_scale = float(radius_scale)
        case = deepcopy(baseline_case)
        case["ion_radii_nm"] = {
            ion: radius_nm * radius_scale
            for ion, radius_nm in baseline_radii.items()
        }
        model_inputs = prepare_case_transport_inputs(case)
        li_rmse, mg_rmse, total_rmse = (
            _foo2023_lmc_ph7_rmse_for_model_inputs(case, model_inputs)
        )

        rows.append(
            {
                "radius_scale": radius_scale,
                "Li_size_ratio": size_ratio(
                    case["ion_radii_nm"]["Li+"], pore_radius_nm
                ),
                "Mg_size_ratio": size_ratio(
                    case["ion_radii_nm"]["Mg2+"], pore_radius_nm
                ),
                "Cl_size_ratio": size_ratio(
                    case["ion_radii_nm"]["Cl-"], pore_radius_nm
                ),
                "Li_hindrance": model_inputs["diffusive_hindrance"]["Li+"],
                "Mg_hindrance": model_inputs["diffusive_hindrance"]["Mg2+"],
                "Cl_hindrance": model_inputs["diffusive_hindrance"]["Cl-"],
                "R_Li_rmse": li_rmse,
                "R_Mg_rmse": mg_rmse,
                "total_rmse": total_rmse,
            }
        )

    return pd.DataFrame(rows)


def foo2023_lmc_ph7_hindrance_floor_sensitivity(floor_values=None):
    """Evaluate validation RMSE after flooring both hindrance-factor sets."""
    if floor_values is None:
        floor_values = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]

    baseline_case = foo2023_lmc_ph7_case()
    rows = []

    for hindrance_floor in floor_values:
        hindrance_floor = float(hindrance_floor)
        case = deepcopy(baseline_case)
        model_inputs = prepare_case_transport_inputs(case)
        model_inputs["diffusive_hindrance"] = {
            ion: max(value, hindrance_floor)
            for ion, value in model_inputs["diffusive_hindrance"].items()
        }
        model_inputs["convective_hindrance"] = {
            ion: max(value, hindrance_floor)
            for ion, value in model_inputs["convective_hindrance"].items()
        }
        li_rmse, mg_rmse, total_rmse = (
            _foo2023_lmc_ph7_rmse_for_model_inputs(case, model_inputs)
        )

        rows.append(
            {
                "hindrance_floor": hindrance_floor,
                "Li_hindrance_effective": model_inputs[
                    "diffusive_hindrance"
                ]["Li+"],
                "Mg_hindrance_effective": model_inputs[
                    "diffusive_hindrance"
                ]["Mg2+"],
                "Cl_hindrance_effective": model_inputs[
                    "diffusive_hindrance"
                ]["Cl-"],
                "R_Li_rmse": li_rmse,
                "R_Mg_rmse": mg_rmse,
                "total_rmse": total_rmse,
            }
        )

    return pd.DataFrame(rows)
