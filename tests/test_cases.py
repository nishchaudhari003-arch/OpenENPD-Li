from openenpd.cases import foo2023_lmc_ph7_case


def test_foo2023_case_has_required_top_level_keys():
    case = foo2023_lmc_ph7_case()

    required_keys = {
        "case_id",
        "paper_id",
        "doi",
        "membrane",
        "temperature_K",
        "crossflow_velocity_m_s",
        "bulk_concentrations_mol_L",
        "charges",
        "ion_radii_nm",
        "membrane_parameters",
        "experimental_data",
        "rejection_definition",
        "activity_model",
        "diffusivities_m2_s",
    }

    assert required_keys.issubset(case.keys())


def test_foo2023_case_feed_composition():
    case = foo2023_lmc_ph7_case()
    concentrations = case["bulk_concentrations_mol_L"]

    assert concentrations["Li+"] == 0.0490
    assert concentrations["Mg2+"] == 0.0843
    assert concentrations["Cl-"] == 0.2172


def test_foo2023_case_membrane_parameters():
    case = foo2023_lmc_ph7_case()
    params = case["membrane_parameters"]

    assert params["pore_radius_nm"] == 0.416
    assert params["pore_dielectric_constant"] == 39.58
    assert params["fixed_charge_mol_m3"] == -63.57
    assert params["fixed_charge_mol_L"] == -0.06357
    assert params["active_layer_thickness_nm"] == 60.06


def test_foo2023_case_experimental_data_lengths():
    case = foo2023_lmc_ph7_case()
    experimental = case["experimental_data"]

    lengths = {len(values) for values in experimental.values()}

    assert lengths == {4}


def test_foo2023_case_experimental_rejection_values():
    case = foo2023_lmc_ph7_case()
    experimental = case["experimental_data"]

    assert experimental["R_Li"][0] == -0.207
    assert experimental["R_Mg"][-1] == 0.653
    
def test_foo2023_case_diffusivities():
    case = foo2023_lmc_ph7_case()
    diffusivities = case["diffusivities_m2_s"]

    assert diffusivities["Li+"] == 1.03e-9
    assert diffusivities["Mg2+"] == 0.706e-9
    assert diffusivities["Cl-"] == 2.03e-9
