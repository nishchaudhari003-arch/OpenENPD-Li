from openenpd.enp import (
    diffusive_flux,
    electromigration_flux,
    convective_flux,
    enp_flux,
)

from openenpd.enp import (
    diffusive_flux,
    electromigration_flux,
    convective_flux,
    enp_flux,
    zero_current_potential_gradient,
    enp_fluxes_zero_current,
)

def test_diffusive_flux_zero_gradient():
    flux = diffusive_flux(
        diffusivity_m2_s=1e-9,
        concentration_gradient_mol_m4=0.0,
    )

    assert flux == 0.0


def test_diffusive_flux_down_concentration_gradient():
    flux = diffusive_flux(
        diffusivity_m2_s=1e-9,
        concentration_gradient_mol_m4=-1000.0,
    )

    assert flux > 0.0


def test_electromigration_flux_zero_potential_gradient():
    flux = electromigration_flux(
        diffusivity_m2_s=1e-9,
        charge=1,
        concentration_mol_m3=100.0,
        potential_gradient_V_m=0.0,
        temperature_K=293.15,
    )

    assert flux == 0.0


def test_convective_flux_positive():
    flux = convective_flux(
        concentration_mol_m3=100.0,
        water_flux_m_s=1e-5,
    )

    assert flux == 1e-3


def test_enp_flux_sum_of_terms():
    total = enp_flux(
        diffusivity_m2_s=1e-9,
        charge=1,
        concentration_mol_m3=100.0,
        concentration_gradient_mol_m4=0.0,
        potential_gradient_V_m=0.0,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
    )

    assert total == 1e-3


def test_enp_flux_with_hindrance_factors():
    total = enp_flux(
        diffusivity_m2_s=1e-9,
        charge=1,
        concentration_mol_m3=100.0,
        concentration_gradient_mol_m4=0.0,
        potential_gradient_V_m=0.0,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
        diffusive_hindrance=0.5,
        convective_hindrance=0.8,
    )

    assert total == 8e-4

def test_zero_current_potential_gradient_returns_float():
    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    gradients = {
        "Li+": -1000.0,
        "Mg2+": -1000.0,
        "Cl-": -1000.0,
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

    potential_gradient = zero_current_potential_gradient(
        concentrations_mol_m3=concentrations,
        concentration_gradients_mol_m4=gradients,
        diffusivities_m2_s=diffusivities,
        charges=charges,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
    )

    assert isinstance(potential_gradient, float)


def test_enp_fluxes_zero_current_satisfies_zero_current():
    concentrations = {
        "Li+": 50.0,
        "Mg2+": 80.0,
        "Cl-": 210.0,
    }

    gradients = {
        "Li+": -1000.0,
        "Mg2+": -1000.0,
        "Cl-": -1000.0,
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

    fluxes = enp_fluxes_zero_current(
        concentrations_mol_m3=concentrations,
        concentration_gradients_mol_m4=gradients,
        diffusivities_m2_s=diffusivities,
        charges=charges,
        water_flux_m_s=1e-5,
        temperature_K=293.15,
    )

    current = sum(charges[ion] * fluxes[ion] for ion in fluxes)

    assert abs(current) < 1e-12


def test_zero_current_raises_for_zero_denominator():
    concentrations = {
        "Li+": 0.0,
    }

    gradients = {
        "Li+": 0.0,
    }

    diffusivities = {
        "Li+": 1.03e-9,
    }

    charges = {
        "Li+": 1,
    }

    try:
        zero_current_potential_gradient(
            concentrations_mol_m3=concentrations,
            concentration_gradients_mol_m4=gradients,
            diffusivities_m2_s=diffusivities,
            charges=charges,
            water_flux_m_s=1e-5,
            temperature_K=293.15,
        )
    except ValueError:
        assert True
    else:
        assert False
