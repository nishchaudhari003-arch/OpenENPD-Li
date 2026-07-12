from openenpd.validation import (
    run_foo2023_lmc_ph7_validation,
    foo2023_lmc_ph7_comparison_rows,
)


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

def test_foo2023_lmc_ph7_comparison_rows_length():
    rows = foo2023_lmc_ph7_comparison_rows()

    assert len(rows) == 4


def test_foo2023_lmc_ph7_comparison_rows_required_keys():
    rows = foo2023_lmc_ph7_comparison_rows()
    first_row = rows[0]

    required_keys = {
        "pressure_bar",
        "Jw_LMH",
        "Jw_m_s",
        "R_Li_exp",
        "R_Mg_exp",
        "R_Li_pred",
        "R_Mg_pred",
    }

    assert required_keys.issubset(first_row.keys())


def test_foo2023_lmc_ph7_comparison_rows_experimental_values():
    rows = foo2023_lmc_ph7_comparison_rows()

    assert rows[0]["pressure_bar"] == 6
    assert rows[0]["Jw_LMH"] == 28.98
    assert rows[0]["R_Li_exp"] == -0.207
    assert rows[-1]["R_Mg_exp"] == 0.653


def test_foo2023_lmc_ph7_comparison_rows_predicted_values_are_numeric():
    rows = foo2023_lmc_ph7_comparison_rows()

    for row in rows:
        assert isinstance(row["R_Li_pred"], float)
        assert isinstance(row["R_Mg_pred"], float)
