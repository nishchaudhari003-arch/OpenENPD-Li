from openenpd.constants import R_GAS, FARADAY, EPSILON_0, AVOGADRO


def test_gas_constant_value():
    assert round(R_GAS, 6) == 8.314463


def test_faraday_constant_value():
    assert round(FARADAY, 2) == 96485.33


def test_vacuum_permittivity_value():
    assert round(EPSILON_0, 23) == 8.8541878128e-12


def test_avogadro_constant_value():
    assert AVOGADRO == 6.02214076e23
