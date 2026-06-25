import math

from openenpd.donnan import (
    thermal_voltage,
    donnan_partition_factor,
    donnan_partition_concentrations,
)


def test_thermal_voltage_at_293K():
    value = thermal_voltage(293.15)
    assert round(value, 4) == 0.0253


def test_donnan_partition_factor_neutral_potential():
    factor = donnan_partition_factor(
        charge=1,
        delta_psi_V=0.0,
        temperature_K=293.15,
    )
    assert factor == 1.0


def test_cation_partition_decreases_for_positive_potential():
    factor = donnan_partition_factor(
        charge=1,
        delta_psi_V=0.01,
        temperature_K=293.15,
    )
    assert factor < 1.0


def test_anion_partition_increases_for_positive_potential():
    factor = donnan_partition_factor(
        charge=-1,
        delta_psi_V=0.01,
        temperature_K=293.15,
    )
    assert factor > 1.0


def test_donnan_partition_concentrations_zero_potential():
    concentrations = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.2172,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    membrane_concentrations = donnan_partition_concentrations(
        concentrations=concentrations,
        charges=charges,
        delta_psi_V=0.0,
        temperature_K=293.15,
    )

    assert membrane_concentrations == concentrations


def test_donnan_partition_factor_matches_manual_expression():
    charge = 2
    delta_psi_V = 0.01
    temperature_K = 293.15

    factor = donnan_partition_factor(
        charge=charge,
        delta_psi_V=delta_psi_V,
        temperature_K=temperature_K,
    )

    expected = math.exp(-charge * 96485.33212 * delta_psi_V / (8.314462618 * temperature_K))

    assert round(factor, 10) == round(expected, 10)

from openenpd.donnan import membrane_charge_balance, solve_donnan_potential


def test_membrane_charge_balance_at_zero_potential():
    concentrations = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.2172,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    residual = membrane_charge_balance(
        delta_psi_V=0.0,
        concentrations=concentrations,
        charges=charges,
        fixed_charge_mol_L=-0.06357,
        temperature_K=293.15,
    )

    assert residual < 0.0


def test_solve_donnan_potential_foo2023_lmc_ph7():
    concentrations = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.2172,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    delta_psi = solve_donnan_potential(
        concentrations=concentrations,
        charges=charges,
        fixed_charge_mol_L=-0.06357,
        temperature_K=293.15,
    )

    residual = membrane_charge_balance(
        delta_psi_V=delta_psi,
        concentrations=concentrations,
        charges=charges,
        fixed_charge_mol_L=-0.06357,
        temperature_K=293.15,
    )

    assert abs(residual) < 1e-10
