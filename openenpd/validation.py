"""
Validation utilities for OpenENPD-Li.

This module connects published validation cases, model-input preparation,
and rejection prediction into reproducible validation workflows.
"""

from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.model import prepare_case_transport_inputs
from openenpd.solver import predict_rejections_for_fluxes


def run_foo2023_lmc_ph7_validation():
    """
    Run the simplified ENP rejection model for the Foo et al. 2023
    LM-C pH ~7 validation case.

    Returns
    -------
    dict
        Validation result containing case metadata, experimental data,
        and model predictions.
    """
    case = foo2023_lmc_ph7_case()
    model_inputs = prepare_case_transport_inputs(case)

    predictions = predict_rejections_for_fluxes(
        model_inputs=model_inputs,
        water_fluxes_m_s=model_inputs["water_fluxes_m_s"],
        temperature_K=case["temperature_K"],
    )

    return {
        "case_id": case["case_id"],
        "doi": case["doi"],
        "experimental_data": case["experimental_data"],
        "predictions": predictions,
    }
