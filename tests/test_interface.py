from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.interface import compute_interface_state


def test_compute_interface_state_has_required_keys():
    case = foo2023_lmc_ph7_case()
    state = compute_interface_state(case)

    required_keys = {
        "delta_psi_V",
        "partition_factors",
        "membrane_concentrations_mol_L",
        "charge_residual_mol_L",
    }

    assert required_keys.issubset(state.keys())


def test_compute_interface_state_charge_residual_is_small():
    case = foo2023_lmc_ph7_case()
    state = compute_interface_state(case)

    assert abs(state["charge_residual_mol_L"]) < 1e-10


def test_compute_interface_state_donnan_potential_is_negative():
    case = foo2023_lmc_ph7_case()
    state = compute_interface_state(case)

    assert state["delta_psi_V"] < 0.0


def test_compute_interface_state_contains_all_ions():
    case = foo2023_lmc_ph7_case()
    state = compute_interface_state(case)

    expected_ions = {"Li+", "Mg2+", "Cl-"}

    assert set(state["partition_factors"].keys()) == expected_ions
    assert set(state["membrane_concentrations_mol_L"].keys()) == expected_ions


def test_compute_interface_state_magnesium_steric_exclusion():
    case = foo2023_lmc_ph7_case()
    state = compute_interface_state(case)

    assert state["partition_factors"]["Mg2+"] == 0.0
    assert state["membrane_concentrations_mol_L"]["Mg2+"] == 0.0
