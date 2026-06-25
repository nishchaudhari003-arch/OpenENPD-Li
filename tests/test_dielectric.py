from openenpd.dielectric import (
    elementary_charge,
    born_energy_j_per_mol,
    dielectric_partition_factor,
    dielectric_partition_factors,
)


def test_elementary_charge_value():
    value = elementary_charge()
    assert round(value, 22) == round(1.602176634e-19, 22)


def test_born_energy_zero_when_dielectrics_equal():
    energy = born_energy_j_per_mol(
        charge=1,
        ion_radius_nm=0.382,
        pore_dielectric_constant=80.1,
        bulk_dielectric_constant=80.1,
    )

    assert abs(energy) < 1e-12


def test_born_energy_positive_for_lower_pore_dielectric():
    energy = born_energy_j_per_mol(
        charge=1,
        ion_radius_nm=0.382,
        pore_dielectric_constant=39.58,
        bulk_dielectric_constant=80.1,
    )

    assert energy > 0.0


def test_dielectric_partition_factor_is_one_when_no_dielectric_penalty():
    factor = dielectric_partition_factor(
        charge=1,
        ion_radius_nm=0.382,
        pore_dielectric_constant=80.1,
        bulk_dielectric_constant=80.1,
        temperature_K=293.15,
    )

    assert round(factor, 12) == 1.0


def test_dielectric_partition_factor_less_than_one_for_lower_pore_dielectric():
    factor = dielectric_partition_factor(
        charge=1,
        ion_radius_nm=0.382,
        pore_dielectric_constant=39.58,
        bulk_dielectric_constant=80.1,
        temperature_K=293.15,
    )

    assert 0.0 < factor < 1.0


def test_divalent_ion_has_stronger_dielectric_exclusion_than_monovalent_ion():
    li_factor = dielectric_partition_factor(
        charge=1,
        ion_radius_nm=0.382,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    mg_factor = dielectric_partition_factor(
        charge=2,
        ion_radius_nm=0.428,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert mg_factor < li_factor


def test_dielectric_partition_factors_multiple_ions():
    ion_radii_nm = {
        "Li+": 0.382,
        "Mg2+": 0.428,
        "Cl-": 0.332,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    factors = dielectric_partition_factors(
        ion_radii_nm=ion_radii_nm,
        charges=charges,
        pore_dielectric_constant=39.58,
        temperature_K=293.15,
    )

    assert set(factors.keys()) == {"Li+", "Mg2+", "Cl-"}
    assert factors["Mg2+"] < factors["Li+"]
    assert factors["Li+"] < 1.0
    assert factors["Cl-"] < 1.0
