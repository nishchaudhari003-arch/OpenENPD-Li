from openenpd.data import load_foo2023_lmc_ph7


def test_load_foo2023_lmc_ph7_shape():
    df = load_foo2023_lmc_ph7()
    assert df.shape[0] == 4


def test_load_foo2023_lmc_ph7_columns():
    df = load_foo2023_lmc_ph7()
    expected_columns = ["pressure_bar", "Jw_LMH", "Jw_um_s", "pH", "R_Li", "R_Mg"]
    assert list(df.columns) == expected_columns


def test_load_foo2023_lmc_ph7_values():
    df = load_foo2023_lmc_ph7()
    assert df["pressure_bar"].tolist() == [6, 8, 10, 12]
    assert round(df["R_Li"].iloc[0], 3) == -0.207
    assert round(df["R_Mg"].iloc[-1], 3) == 0.653
