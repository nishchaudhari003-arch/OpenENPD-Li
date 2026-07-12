from openenpd.hindrance import (
    diffusive_hindrance_factor,
    convective_hindrance_factor,
    hindrance_factors,
)


def test_diffusive_hindrance_half_radius():
    factor = diffusive_hindrance_factor(
        ion_radius_nm=0.208,
        pore_radius_nm=0.416,
    )

    assert factor == 0.25


def test_convective_hindrance_half_radius():
    factor = convective_hindrance_factor(
        ion_radius_nm=0.208,
        pore_radius_nm=0.416,
    )

    assert factor == 0.25


def test_diffusive_hindrance_full_exclusion():
    factor = diffusive_hindrance_factor(
        ion_radius_nm=0.500,
        pore_radius_nm=0.416,
    )

    assert factor == 0.0


def test_convective_hindrance_full_exclusion():
    factor = convective_hindrance_factor(
        ion_radius_nm=0.500,
        pore_radius_nm=0.416,
    )

    assert factor == 0.0


def test_hindrance_factors_multiple_ions():
    ion_radii_nm = {
        "Li+": 0.382,
        "Mg2+": 0.428,
        "Cl-": 0.332,
    }

    factors = hindrance_factors(
        ion_radii_nm=ion_radii_nm,
        pore_radius_nm=0.416,
    )

    assert set(factors.keys()) == {"diffusive", "convective"}
    assert set(factors["diffusive"].keys()) == {"Li+", "Mg2+", "Cl-"}
    assert set(factors["convective"].keys()) == {"Li+", "Mg2+", "Cl-"}

    assert factors["diffusive"]["Mg2+"] == 0.0
    assert factors["convective"]["Mg2+"] == 0.0

    assert factors["diffusive"]["Li+"] > 0.0
    assert factors["convective"]["Li+"] > 0.0

    assert factors["diffusive"]["Cl-"] > factors["diffusive"]["Li+"]
    assert factors["convective"]["Cl-"] > factors["convective"]["Li+"]
