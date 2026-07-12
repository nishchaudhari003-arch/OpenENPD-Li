from openenpd.solver import (
    mean_concentrations,
    flux_residuals_for_permeate_concentrations,
    solve_permeate_concentrations_for_flux,
    predict_rejection_for_flux,
    predict_rejections_for_fluxes,
)


def test_mean_concentrations():
    upstream = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    downstream = {
        "Li+": 40.0,
        "Mg2+": 60.0,
        "Cl-": 180.0,
    }

    mean = mean_concentrations(
        upstream_concentrations=upstream,
        downstream_concentrations=downstream,
    )

    assert mean["Li+"] == 45.0
    assert mean["Mg2+"] == 70.0
    assert mean["Cl-"] == 195.0


def test_flux_residuals_zero_for_uniform_electroneutral_solution():
    ions = ["Li+", "Mg2+", "Cl-"]

    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    diffusivities = {
        "Li+": 1.03e-9,
        "Mg2+": 0.706e-9,
        "Cl-": 2.03e-9,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    hindrance = {
        "Li+": 1.0,
        "Mg2+": 1.0,
        "Cl-": 1.0,
    }

    residuals = flux_residuals_for_permeate_concentrations(
        permeate_values=[50.0, 80.0, 210.0],
        ions=ions,
        upstream_concentrations_mol_m3=concentrations,
        diffusivities_m2_s=diffusivities,
        charges=charges,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
        thickness_m=60e-9,
        diffusive_hindrance=hindrance,
        convective_hindrance=hindrance,
    )

    assert all(abs(value) < 1e-12 for value in residuals)


def test_solve_permeate_concentrations_uniform_solution():
    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    diffusivities = {
        "Li+": 1.03e-9,
        "Mg2+": 0.706e-9,
        "Cl-": 2.03e-9,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    hindrance = {
        "Li+": 1.0,
        "Mg2+": 1.0,
        "Cl-": 1.0,
    }

    permeate = solve_permeate_concentrations_for_flux(
        upstream_concentrations_mol_m3=concentrations,
        feed_concentrations_mol_m3=concentrations,
        diffusivities_m2_s=diffusivities,
        charges=charges,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
        thickness_m=60e-9,
        diffusive_hindrance=hindrance,
        convective_hindrance=hindrance,
    )

    assert round(permeate["Li+"], 6) == 50.0
    assert round(permeate["Mg2+"], 6) == 80.0
    assert round(permeate["Cl-"], 6) == 210.0


def test_predict_rejection_for_flux_uniform_solution():
    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    model_inputs = {
        "membrane_concentrations_mol_m3": concentrations,
        "bulk_concentrations_mol_m3": concentrations,
        "diffusivities_m2_s": {
            "Li+": 1.03e-9,
            "Mg2+": 0.706e-9,
            "Cl-": 2.03e-9,
        },
        "charges": {
            "Li+": 1,
            "Mg2+": 2,
            "Cl-": -1,
        },
        "active_layer_thickness_m": 60e-9,
        "diffusive_hindrance": {
            "Li+": 1.0,
            "Mg2+": 1.0,
            "Cl-": 1.0,
        },
        "convective_hindrance": {
            "Li+": 1.0,
            "Mg2+": 1.0,
            "Cl-": 1.0,
        },
    }

    prediction = predict_rejection_for_flux(
        model_inputs=model_inputs,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
    )

    rejections = prediction["rejections"]

    assert abs(rejections["Li+"]) < 1e-12
    assert abs(rejections["Mg2+"]) < 1e-12
    assert abs(rejections["Cl-"]) < 1e-12

def test_predict_rejections_for_multiple_fluxes_uniform_solution():
    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    model_inputs = {
        "membrane_concentrations_mol_m3": concentrations,
        "bulk_concentrations_mol_m3": concentrations,
        "diffusivities_m2_s": {
            "Li+": 1.03e-9,
            "Mg2+": 0.706e-9,
            "Cl-": 2.03e-9,
        },
        "charges": {
            "Li+": 1,
            "Mg2+": 2,
            "Cl-": -1,
        },
        "active_layer_thickness_m": 60e-9,
        "diffusive_hindrance": {
            "Li+": 1.0,
            "Mg2+": 1.0,
            "Cl-": 1.0,
        },
        "convective_hindrance": {
            "Li+": 1.0,
            "Mg2+": 1.0,
            "Cl-": 1.0,
        },
    }

    predictions = predict_rejections_for_fluxes(
        model_inputs=model_inputs,
        water_fluxes_m_s=[8.05e-6, 11.72e-6, 15.05e-6, 18.66e-6],
        temperature_K=293.15,
    )

    assert len(predictions) == 4

    for prediction in predictions:
        assert "water_flux_m_s" in prediction
        assert "permeate_concentrations_mol_m3" in prediction
        assert "rejections" in prediction
        assert abs(prediction["rejections"]["Li+"]) < 1e-12
        assert abs(prediction["rejections"]["Mg2+"]) < 1e-12
        assert abs(prediction["rejections"]["Cl-"]) < 1e-12
