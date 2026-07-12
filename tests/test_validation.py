from openenpd.validation import run_foo2023_lmc_ph7_validation


def test_run_foo2023_lmc_ph7_validation_required_keys():
    result = run_foo2023_lmc_ph7_validation()

    required_keys = {
        "case_id",
        "doi",
        "experimental_data",
        "predictions",
    }

    assert required_keys.issubset(result.keys())


def test_run_foo2023_lmc_ph7_validation_case_metadata():
    result = run_foo2023_lmc_ph7_validation()

    assert result["case_id"] == "foo2023_lmc_ph7"
    assert result["doi"] == "10.1021/acs.est.2c08584"


def test_run_foo2023_lmc_ph7_validation_prediction_count():
    result = run_foo2023_lmc_ph7_validation()

    assert len(result["predictions"]) == 4


def test_run_foo2023_lmc_ph7_validation_prediction_structure():
    result = run_foo2023_lmc_ph7_validation()

    first_prediction = result["predictions"][0]

    assert "water_flux_m_s" in first_prediction
    assert "permeate_concentrations_mol_m3" in first_prediction
    assert "rejections" in first_prediction


def test_run_foo2023_lmc_ph7_validation_contains_li_mg_rejections():
    result = run_foo2023_lmc_ph7_validation()

    first_prediction = result["predictions"][0]
    rejections = first_prediction["rejections"]

    assert "Li+" in rejections
    assert "Mg2+" in rejections
    assert "Cl-" in rejections
