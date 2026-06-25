from openenpd.species import LI, MG, CL, COMMON_IONS


def test_lithium_definition():
    assert LI.name == "lithium"
    assert LI.symbol == "Li+"
    assert LI.charge == 1


def test_magnesium_definition():
    assert MG.name == "magnesium"
    assert MG.symbol == "Mg2+"
    assert MG.charge == 2


def test_chloride_definition():
    assert CL.name == "chloride"
    assert CL.symbol == "Cl-"
    assert CL.charge == -1


def test_common_ions_lookup():
    assert COMMON_IONS["Li+"].charge == 1
    assert COMMON_IONS["Mg2+"].charge == 2
    assert COMMON_IONS["Cl-"].charge == -1
