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

def foo2023_lmc_ph7_comparison_rows():
    """
    Create comparison rows for Foo et al. 2023 LM-C pH ~7 validation.

    Returns
    -------
    list of dict
        Rows containing experimental and predicted Li/Mg rejection values
        at each water flux.
    """
    result = run_foo2023_lmc_ph7_validation()

    experimental = result["experimental_data"]
    predictions = result["predictions"]

    rows = []

    for index, prediction in enumerate(predictions):
        rejections = prediction["rejections"]

        R_Li_exp = experimental["R_Li"][index]
        R_Mg_exp = experimental["R_Mg"][index]
        R_Li_pred = rejections["Li+"]
        R_Mg_pred = rejections["Mg2+"]

        rows.append(
            {
                "pressure_bar": experimental["pressure_bar"][index],
                "Jw_LMH": experimental["Jw_LMH"][index],
                "Jw_m_s": prediction["water_flux_m_s"],
                "R_Li_exp": R_Li_exp,
                "R_Mg_exp": R_Mg_exp,
                "R_Li_pred": R_Li_pred,
                "R_Mg_pred": R_Mg_pred,
                "R_Li_residual": R_Li_pred - R_Li_exp,
                "R_Mg_residual": R_Mg_pred - R_Mg_exp,
            }
        )

    return rows
def rejection_rmse(rows, experimental_key, predicted_key):
    """
    Compute root-mean-square error for rejection predictions.

    Parameters
    ----------
    rows : list of dict
        Comparison rows containing experimental and predicted values.

    experimental_key : str
        Key for experimental rejection values.

    predicted_key : str
        Key for predicted rejection values.

    Returns
    -------
    float
        Root-mean-square error.
    """
    squared_errors = []

    for row in rows:
        error = row[predicted_key] - row[experimental_key]
        squared_errors.append(error**2)

    mean_squared_error = sum(squared_errors) / len(squared_errors)

    return mean_squared_error**0.5


def foo2023_lmc_ph7_validation_metrics():
    """
    Compute validation metrics for Foo et al. 2023 LM-C pH ~7 case.

    Returns
    -------
    dict
        RMSE values for Li+ and Mg2+ rejection predictions.
    """
    rows = foo2023_lmc_ph7_comparison_rows()

    return {
        "case_id": "foo2023_lmc_ph7",
        "R_Li_rmse": rejection_rmse(
            rows=rows,
            experimental_key="R_Li_exp",
            predicted_key="R_Li_pred",
        ),
        "R_Mg_rmse": rejection_rmse(
            rows=rows,
            experimental_key="R_Mg_exp",
            predicted_key="R_Mg_pred",
        ),
    }
