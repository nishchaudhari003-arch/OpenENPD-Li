from openenpd.rejection import (
    permeate_concentration_from_flux,
    species_rejection,
    species_rejections,
    separation_factor,
)


def test_permeate_concentration_from_flux():
    concentration = permeate_concentration_from_flux(
        ion_flux_mol_m2_s=1e-3,
        water_flux_m_s=1e-5,
    )

    assert concentration == 100.0


def test_permeate_concentration_zero_water_flux_raises():
    try:
        permeate_concentration_from_flux(
            ion_flux_mol_m2_s=1e-3,
            water_flux_m_s=0.0,
        )
    except ValueError:
        assert True
    else:
        assert False


def test_species_rejection_positive():
    rejection = species_rejection(
        feed_concentration=100.0,
        permeate_concentration=40.0,
    )

    assert rejection == 0.6


def test_species_rejection_negative():
    rejection = species_rejection(
        feed_concentration=100.0,
        permeate_concentration=120.0,
    )

    assert rejection == -0.2


def test_species_rejection_zero_feed_raises():
    try:
        species_rejection(
            feed_concentration=0.0,
            permeate_concentration=10.0,
        )
    except ValueError:
        assert True
    else:
        assert False


def test_species_rejections_multiple_ions():
    feed = {
        "Li+": 49.0,
        "Mg2+": 84.3,
    }

    permeate = {
        "Li+": 58.8,
        "Mg2+": 33.72,
    }

    rejections = species_rejections(
        feed_concentrations=feed,
        permeate_concentrations=permeate,
    )

    assert round(rejections["Li+"], 3) == -0.200
    assert round(rejections["Mg2+"], 3) == 0.600


def test_separation_factor():
    sf = separation_factor(
        rejection_a=-0.2,
        rejection_b=0.6,
    )

    assert round(sf, 3) == 3.000


def test_separation_factor_denominator_zero_raises():
    try:
        separation_factor(
            rejection_a=-0.2,
            rejection_b=1.0,
        )
    except ValueError:
        assert True
    else:
        assert False
