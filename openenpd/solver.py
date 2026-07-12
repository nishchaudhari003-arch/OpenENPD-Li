"""
Simplified membrane-scale ENP solver for OpenENPD-Li.

This module provides an initial approximate solver that predicts permeate
concentrations and species rejection for a single water flux.

The current formulation is intentionally simple:

- feed-side membrane-interface concentrations are known
- permeate-side concentrations are unknown
- concentration profiles are approximated as linear across the active layer
- ENP fluxes are computed with the zero-current condition
- permeate concentration is constrained by J_i = C_i,p * J_v

This is the first model-level solver and will later be upgraded to a more
complete boundary-value formulation.
"""

from scipy.optimize import root

from openenpd.enp import enp_fluxes_zero_current
from openenpd.rejection import species_rejections
from openenpd.transport import linear_concentration_gradients


def mean_concentrations(upstream_concentrations, downstream_concentrations):
    """
    Compute mean concentrations across the membrane active layer.

    Parameters
    ----------
    upstream_concentrations : dict
        Feed-side membrane-interface concentrations in mol/m^3.

    downstream_concentrations : dict
        Permeate-side concentrations in mol/m^3.

    Returns
    -------
    dict
        Mean concentrations in mol/m^3.
    """
    return {
        ion: 0.5 * (upstream_concentrations[ion] + downstream_concentrations[ion])
        for ion in upstream_concentrations
    }


def flux_residuals_for_permeate_concentrations(
    permeate_values,
    ions,
    upstream_concentrations_mol_m3,
    diffusivities_m2_s,
    charges,
    water_flux_m_s,
    temperature_K,
    thickness_m,
    diffusive_hindrance,
    convective_hindrance,
):
    """
    Compute residuals for permeate concentration unknowns.

    Residual for each ion:

        residual_i = J_i - C_i,p * J_v

    where J_i is computed from the ENP equation and C_i,p * J_v is the
    ion flux implied by the permeate concentration.

    Parameters
    ----------
    permeate_values : list or array
        Trial permeate concentrations in mol/m^3.

    ions : list
        Ordered ion labels.

    upstream_concentrations_mol_m3 : dict
        Feed-side membrane-interface concentrations in mol/m^3.

    diffusivities_m2_s : dict
        Ion diffusivities in m^2/s.

    charges : dict
        Ion charge numbers.

    water_flux_m_s : float
        Water flux in m/s.

    temperature_K : float
        Temperature in kelvin.

    thickness_m : float
        Membrane active-layer thickness in meters.

    diffusive_hindrance : dict
        Ion-specific diffusive hindrance factors.

    convective_hindrance : dict
        Ion-specific convective hindrance factors.

    Returns
    -------
    list
        Residuals in mol m^-2 s^-1.
    """
    permeate_concentrations = {
        ion: value for ion, value in zip(ions, permeate_values)
    }

    gradients = linear_concentration_gradients(
        upstream_concentrations_mol_m3=upstream_concentrations_mol_m3,
        downstream_concentrations_mol_m3=permeate_concentrations,
        thickness_m=thickness_m,
    )

    local_concentrations = mean_concentrations(
        upstream_concentrations=upstream_concentrations_mol_m3,
        downstream_concentrations=permeate_concentrations,
    )

    fluxes = enp_fluxes_zero_current(
        concentrations_mol_m3=local_concentrations,
        concentration_gradients_mol_m4=gradients,
        diffusivities_m2_s=diffusivities_m2_s,
        charges=charges,
        water_flux_m_s=water_flux_m_s,
        temperature_K=temperature_K,
        diffusive_hindrance=diffusive_hindrance,
        convective_hindrance=convective_hindrance,
    )

    residuals = []

    for ion in ions:
        residuals.append(
            fluxes[ion] - permeate_concentrations[ion] * water_flux_m_s
        )

    return residuals


def solve_permeate_concentrations_for_flux(
    upstream_concentrations_mol_m3,
    feed_concentrations_mol_m3,
    diffusivities_m2_s,
    charges,
    water_flux_m_s,
    temperature_K,
    thickness_m,
    diffusive_hindrance,
    convective_hindrance,
):
    """
    Solve permeate concentrations for one water flux.

    Parameters
    ----------
    upstream_concentrations_mol_m3 : dict
        Feed-side membrane-interface concentrations in mol/m^3.

    feed_concentrations_mol_m3 : dict
        Bulk feed concentrations in mol/m^3, used as initial guesses.

    diffusivities_m2_s : dict
        Ion diffusivities in m^2/s.

    charges : dict
        Ion charge numbers.

    water_flux_m_s : float
        Water flux in m/s.

    temperature_K : float
        Temperature in kelvin.

    thickness_m : float
        Membrane active-layer thickness in meters.

    diffusive_hindrance : dict
        Ion-specific diffusive hindrance factors.

    convective_hindrance : dict
        Ion-specific convective hindrance factors.

    Returns
    -------
    dict
        Predicted permeate concentrations in mol/m^3.
    """
    ions = list(feed_concentrations_mol_m3.keys())
    initial_guess = [feed_concentrations_mol_m3[ion] for ion in ions]

    solution = root(
        flux_residuals_for_permeate_concentrations,
        initial_guess,
        args=(
            ions,
            upstream_concentrations_mol_m3,
            diffusivities_m2_s,
            charges,
            water_flux_m_s,
            temperature_K,
            thickness_m,
            diffusive_hindrance,
            convective_hindrance,
        ),
    )

    if not solution.success:
        raise RuntimeError(f"Permeate concentration solver failed: {solution.message}")

    return {
        ion: value for ion, value in zip(ions, solution.x)
    }


def predict_rejection_for_flux(
    model_inputs,
    water_flux_m_s,
    temperature_K,
):
    """
    Predict species rejection for one water flux.

    Parameters
    ----------
    model_inputs : dict
        Prepared case transport inputs from prepare_case_transport_inputs().

    water_flux_m_s : float
        Water flux in m/s.

    temperature_K : float
        Temperature in kelvin.

    Returns
    -------
    dict
        Predicted permeate concentrations and species rejections.
    """
    permeate_concentrations = solve_permeate_concentrations_for_flux(
        upstream_concentrations_mol_m3=model_inputs["membrane_concentrations_mol_m3"],
        feed_concentrations_mol_m3=model_inputs["bulk_concentrations_mol_m3"],
        diffusivities_m2_s=model_inputs["diffusivities_m2_s"],
        charges=model_inputs["charges"],
        water_flux_m_s=water_flux_m_s,
        temperature_K=temperature_K,
        thickness_m=model_inputs["active_layer_thickness_m"],
        diffusive_hindrance=model_inputs["diffusive_hindrance"],
        convective_hindrance=model_inputs["convective_hindrance"],
    )

    rejections = species_rejections(
        feed_concentrations=model_inputs["bulk_concentrations_mol_m3"],
        permeate_concentrations=permeate_concentrations,
    )

    return {
        "water_flux_m_s": water_flux_m_s,
        "permeate_concentrations_mol_m3": permeate_concentrations,
        "rejections": rejections,
    }
