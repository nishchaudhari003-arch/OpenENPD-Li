from openenpd.enp import (
    diffusive_flux,
    electromigration_flux,
    convective_flux,
    enp_flux,
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
