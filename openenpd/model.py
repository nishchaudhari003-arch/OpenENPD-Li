"""
Case-level modeling utilities for OpenENPD-Li.

This module connects validation-case definitions to membrane-interface,
hindrance, and transport calculations.
"""

from openenpd.interface import compute_interface_state
from openenpd.hindrance import hindrance_factors
from openenpd.transport import (
    active_layer_thickness_m,
    convert_concentrations_mol_L_to_mol_m3,
)


def prepare_case_transport_inputs(case):
    """
    Prepare transport-model inputs from a validation case.

    Parameters
    ----------
    case : dict
        Validation case definition.

    Returns
    -------
    dict
        Transport-ready inputs including interface state, concentrations
        in mol/m^3, hindrance factors, diffusivities, membrane thickness,
        charges, and experimental water fluxes.
    """
    interface_state = compute_interface_state(case)

    membrane_concentrations_mol_m3 = convert_concentrations_mol_L_to_mol_m3(
        interface_state["membrane_concentrations_mol_L"]
    )

    bulk_concentrations_mol_m3 = convert_concentrations_mol_L_to_mol_m3(
        case["bulk_concentrations_mol_L"]
    )

    params = case["membrane_parameters"]

    hindrance = hindrance_factors(
        ion_radii_nm=case["ion_radii_nm"],
        pore_radius_nm=params["pore_radius_nm"],
    )

    return {
        "case_id": case["case_id"],
        "interface_state": interface_state,
        "bulk_concentrations_mol_m3": bulk_concentrations_mol_m3,
        "membrane_concentrations_mol_m3": membrane_concentrations_mol_m3,
        "diffusivities_m2_s": case["diffusivities_m2_s"],
        "charges": case["charges"],
        "diffusive_hindrance": hindrance["diffusive"],
        "convective_hindrance": hindrance["convective"],
        "active_layer_thickness_m": active_layer_thickness_m(case),
        "water_fluxes_m_s": [
            value * 1e-6 for value in case["experimental_data"]["Jw_um_s"]
        ],
    }
