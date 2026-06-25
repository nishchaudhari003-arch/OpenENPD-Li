from openenpd.steric import (
    size_ratio,
    steric_partition_factor,
    steric_partition_factors,
)


def test_size_ratio():
    value = size_ratio(ion_radius_nm=0.208, pore_radius_nm=0.416)
    assert value == 0.5


def test_steric_partition_factor_half_radius():
    factor = steric_partition_factor(
        ion_radius_nm=0.208,
        pore_radius_nm=0.416,
    )
    assert factor == 0.25


def test_steric_partition_factor_full_exclusion():
    factor = steric_partition_factor(
        ion_radius_nm=0.500,
        pore_radius_nm=0.416,
    )
    assert factor == 0.0


def test_steric_partition_factors_multiple_ions():
    ion_radii_nm = {
        "Li+": 0.382,
        "Mg2+": 0.428,
        "Cl-": 0.332,
    }

    factors = steric_partition_factors(
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=0.416,
    )

    assert factors["Li+"] > 0.0
    assert factors["Mg2+"] == 0.0
    assert factors["Cl-"] > factors["Li+"]
