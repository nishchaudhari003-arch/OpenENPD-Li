from openenpd.units import lmh_to_m_s, m_s_to_lmh, nm_to_m


def test_lmh_to_m_s():
    assert round(lmh_to_m_s(36), 8) == 0.00001


def test_m_s_to_lmh():
    assert round(m_s_to_lmh(1e-5), 2) == 36.00


def test_roundtrip_flux_conversion():
    value_lmh = 28.98
    converted = m_s_to_lmh(lmh_to_m_s(value_lmh))
    assert round(converted, 2) == value_lmh


def test_nm_to_m():
    assert nm_to_m(0.416) == 0.416e-9

def test_mol_L_to_mol_m3():
    assert mol_L_to_mol_m3(0.0490) == 49.0


def test_mol_m3_to_mol_L():
    assert mol_m3_to_mol_L(49.0) == 0.0490


def test_concentration_conversion_roundtrip():
    value_mol_L = 0.0843
    converted = mol_m3_to_mol_L(mol_L_to_mol_m3(value_mol_L))
    assert converted == value_mol_L
