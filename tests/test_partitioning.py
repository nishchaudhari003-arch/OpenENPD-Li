from openenpd.partitioning import (
    total_partition_factor,
    total_partition_factors,
    membrane_interface_concentrations,
)


def test_total_partition_factor_positive_but_less_than_one_for_lithium():
    factor = total_partition_factor(
        charge=1,
        ion_radius_nm=0.382,
        pore_radius_nm=0.416,
        delta_psi_V=0.0,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert 0.0 < factor < 1.0


def test_total_partition_factor_zero_when_sterically_excluded():
    factor = total_partition_factor(
        charge=2,
        ion_radius_nm=0.428,
        pore_radius_nm=0.416,
        delta_psi_V=0.0,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert factor == 0.0


def test_total_partition_factors_multiple_ions():
    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    ion_radii_nm = {
        "Li+": 0.382,
        "Mg2+": 0.428,
        "Cl-": 0.332,
    }

    factors = total_partition_factors(
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=0.416,
        delta_psi_V=0.0,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert set(factors.keys()) == {"Li+", "Mg2+", "Cl-"}
    assert factors["Mg2+"] == 0.0
    assert factors["Li+"] > 0.0
    assert factors["Cl-"] > factors["Li+"]


def test_membrane_interface_concentrations():
    bulk_concentrations = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.2172,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    ion_radii_nm = {
        "Li+": 0.382,
        "Mg2+": 0.428,
        "Cl-": 0.332,
    }

    membrane_concentrations = membrane_interface_concentrations(
        bulk_concentrations=bulk_concentrations,
        charges=charges,
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=0.416,
        delta_psi_V=0.0,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert membrane_concentrations["Li+"] > 0.0
    assert membrane_concentrations["Mg2+"] == 0.0
    assert membrane_concentrations["Cl-"] > membrane_concentrations["Li+"]
