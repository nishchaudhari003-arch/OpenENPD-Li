from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.transport import (
    active_layer_thickness_m,
    convert_concentrations_mol_L_to_mol_m3,
    linear_concentration_gradient,
    linear_concentration_gradients,
)


def test_active_layer_thickness_m():
    case = foo2023_lmc_ph7_case()
    thickness = active_layer_thickness_m(case)

    assert thickness == 60.06e-9


def test_convert_concentrations_mol_L_to_mol_m3():
    concentrations_mol_L = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.2172,
    }

    converted = convert_concentrations_mol_L_to_mol_m3(concentrations_mol_L)

    assert converted["Li+"] == 49.0
    assert converted["Mg2+"] == 84.3
    assert converted["Cl-"] == 217.2


def test_linear_concentration_gradient():
    gradient = linear_concentration_gradient(
        upstream_concentration_mol_m3=100.0,
        downstream_concentration_mol_m3=50.0,
        thickness_m=10e-9,
    )

    assert gradient == -5.0e9


def test_linear_concentration_gradient_zero_thickness_raises():
    try:
        linear_concentration_gradient(
            upstream_concentration_mol_m3=100.0,
            downstream_concentration_mol_m3=50.0,
            thickness_m=0.0,
        )
    except ValueError:
        assert True
    else:
        assert False


def test_linear_concentration_gradients_multiple_ions():
    upstream = {
        "Li+": 100.0,
        "Mg2+": 80.0,
        "Cl-": 200.0,
    }

    downstream = {
        "Li+": 50.0,
        "Mg2+": 20.0,
        "Cl-": 100.0,
    }

    gradients = linear_concentration_gradients(
        upstream_concentrations_mol_m3=upstream,
        downstream_concentrations_mol_m3=downstream,
        thickness_m=10e-9,
    )

    assert gradients["Li+"] == -5.0e9
    assert gradients["Mg2+"] == -6.0e9
    assert gradients["Cl-"] == -1.0e10
