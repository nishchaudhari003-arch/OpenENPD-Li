from openenpd.electroneutrality import charge_balance, is_electroneutral


def test_foo2023_lmc_feed_charge_balance():
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

    balance = charge_balance(concentrations, charges)

    assert abs(balance) < 1e-3


def test_foo2023_lmc_feed_is_electroneutral():
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

    assert is_electroneutral(concentrations, charges, tolerance=1e-3)


def test_non_electroneutral_solution_fails():
    concentrations = {
        "Li+": 0.0490,
        "Mg2+": 0.0843,
        "Cl-": 0.1000,
    }

    charges = {
        "Li+": 1,
        "Mg2+": 2,
        "Cl-": -1,
    }

    assert not is_electroneutral(concentrations, charges, tolerance=1e-3)
