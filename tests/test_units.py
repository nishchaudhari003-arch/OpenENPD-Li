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
