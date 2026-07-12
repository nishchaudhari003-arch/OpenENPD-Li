"""
Published validation cases for OpenENPD-Li.

This module stores benchmark case definitions used for reproducible
model validation.
"""


def foo2023_lmc_ph7_case():
    """
    Return the Foo et al. 2023 LM-C pH ~7 validation case.

    This case corresponds to the simplified Li+-Mg2+-Cl- brine using
    NF270 at approximately neutral pH.

    Returns
    -------
    dict
        Case definition containing feed composition, ion properties,
        membrane parameters, operating conditions, and experimental
        rejection data.
    """
    return {
        "case_id": "foo2023_lmc_ph7",
        "paper_id": "foo_2023_est_lmc_ph7",
        "doi": "10.1021/acs.est.2c08584",
        "membrane": "NF270",
        "temperature_K": 293.15,
        "crossflow_velocity_m_s": 0.17,
        "bulk_concentrations_mol_L": {
            "Li+": 0.0490,
            "Mg2+": 0.0843,
            "Cl-": 0.2172,
        },
        "charges": {
            "Li+": 1,
            "Mg2+": 2,
            "Cl-": -1,
        },
        "ion_radii_nm": {
            "Li+": 0.382,
            "Mg2+": 0.428,
            "Cl-": 0.332,
        },

        "diffusivities_m2_s": {
            "Li+": 1.03e-9,
            "Mg2+": 0.706e-9,
            "Cl-": 2.03e-9,
        },
        
        "membrane_parameters": {
            "pore_radius_nm": 0.416,
            "pore_dielectric_constant": 39.58,
            "fixed_charge_mol_m3": -63.57,
            "fixed_charge_mol_L": -0.06357,
            "active_layer_thickness_nm": 60.06,
        },
        "experimental_data": {
            "pressure_bar": [6, 8, 10, 12],
            "Jw_LMH": [28.98, 42.18, 54.17, 67.18],
            "Jw_um_s": [8.05, 11.72, 15.05, 18.66],
            "pH": [6.94, 6.71, 6.83, 6.92],
            "R_Li": [-0.207, -0.184, -0.142, -0.125],
            "R_Mg": [0.521, 0.600, 0.599, 0.653],
        },
        "rejection_definition": "R_i = 1 - C_i,p / C_i,f",
        "activity_model": "ideal baseline; nonideality not yet included",
    }
