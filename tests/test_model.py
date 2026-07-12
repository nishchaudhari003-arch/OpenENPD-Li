from openenpd.cases import foo2023_lmc_ph7_case
from openenpd.model import prepare_case_transport_inputs


def test_prepare_case_transport_inputs_required_keys():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    required_keys = {
        "case_id",
        "interface_state",
        "bulk_concentrations_mol_m3",
        "membrane_concentrations_mol_m3",
        "diffusivities_m2_s",
        "charges",
        "diffusive_hindrance",
        "convective_hindrance",
        "active_layer_thickness_m",
        "water_fluxes_m_s",
    }

    assert required_keys.issubset(inputs.keys())


def test_prepare_case_transport_inputs_case_id():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    assert inputs["case_id"] == "foo2023_lmc_ph7"


def test_prepare_case_transport_inputs_bulk_concentration_conversion():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    assert inputs["bulk_concentrations_mol_m3"]["Li+"] == 49.0
    assert inputs["bulk_concentrations_mol_m3"]["Mg2+"] == 84.3
    assert round(inputs["bulk_concentrations_mol_m3"]["Cl-"], 3) == 217.200


def test_prepare_case_transport_inputs_interface_state_valid():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    interface_state = inputs["interface_state"]

    assert interface_state["delta_psi_V"] < 0.0
    assert abs(interface_state["charge_residual_mol_L"]) < 1e-10


def test_prepare_case_transport_inputs_hindrance_values():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    assert inputs["diffusive_hindrance"]["Mg2+"] == 0.0
    assert inputs["convective_hindrance"]["Mg2+"] == 0.0
    assert inputs["diffusive_hindrance"]["Li+"] > 0.0
    assert inputs["convective_hindrance"]["Li+"] > 0.0


def test_prepare_case_transport_inputs_thickness():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    assert round(inputs["active_layer_thickness_m"], 12) == round(60.06e-9, 12)


def test_prepare_case_transport_inputs_water_fluxes():
    case = foo2023_lmc_ph7_case()
    inputs = prepare_case_transport_inputs(case)

    expected = [8.05e-6, 11.72e-6, 15.05e-6, 18.66e-6]

    assert len(inputs["water_fluxes_m_s"]) == 4
    assert round(inputs["water_fluxes_m_s"][0], 10) == round(expected[0], 10)
    assert round(inputs["water_fluxes_m_s"][-1], 10) == round(expected[-1], 10)
