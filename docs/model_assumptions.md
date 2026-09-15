# Model assumptions

This document records the modelling conventions and assumptions in OpenENPD-Li.
It distinguishes the tested **baseline** physics from the **diagnostic**
extensions. Nothing here is a claim of a validated physical model.

## Species and concentration conventions

- The benchmark system is a simplified Li⁺–Mg²⁺–Cl⁻ brine (Foo 2023 LM-C pH ≈ 7).
- Ion charges follow the usual convention (Li⁺ = +1, Mg²⁺ = +2, Cl⁻ = −1).
- Bulk and permeate concentrations are handled in **mol/L**; the transport layer
  converts to **mol/m³** through the existing unit utilities rather than by
  embedding conversion factors in the physics.
- Membrane radii and pore radius are in **nanometers**; the active-layer
  thickness is converted to meters for transport.

## Rejection definition

Species (observed/apparent) rejection is

    R_i = 1 - C_p,i / C_f,i

where `C_p,i` is the permeate concentration and `C_f,i` the feed concentration.
Rejection is never clipped: `C_p,i > C_f,i` (hence `R_i < 0`) is allowed.

## Donnan partitioning

Ideal Donnan equilibrium at a solution/membrane interface:

    C_i,m = C_i,b * exp(-z_i F Δψ_D / (R T))

The Donnan potential `Δψ_D` is chosen to satisfy membrane electroneutrality,
`Σ_i z_i C_i,m + X = 0`, where `X` is the signed fixed-charge concentration.

## Steric partitioning

A simple hard-sphere-in-cylindrical-pore steric factor,

    φ_s,i = (1 - λ_i)^2   for λ_i < 1,   with   λ_i = r_i / r_p,

and `φ_s,i = 0` for `λ_i ≥ 1` (full size exclusion). This hard cutoff is the
**baseline** behaviour; diagnostic variants can soften it (see below).

## Dielectric / Born exclusion

A Born-type dielectric exclusion penalty for moving an ion from bulk into a pore
of different dielectric constant:

    ΔG_i = N_A z_i² e² / (8 π ε₀ r_i) (1/ε_p - 1/ε_b),   K_i = exp(-ΔG_i / (R T))

## Hindrance treatment

Baseline diffusive and convective hindrance factors use the same
`(1 - λ_i)^2` form (0 for `λ_i ≥ 1`). More detailed Deen/DSPM-DE correlations
are a possible future refinement, not part of the current baseline.

## Two-interface ENP structure

The two-interface model (diagnostic, layered on top of the baseline) adds the
permeate-side interface that the simplified baseline omits:

- feed-side membrane concentrations `C_m0,i = C_f,i Φ_i exp(-z_i F Δψ_D,f/RT)`,
- permeate-side membrane concentrations
  `C_mL,i = C_p,i Φ_i exp(-z_i F Δψ_D,p/RT)` (same solution-to-membrane
  convention), with `Φ_i = φ_s,i φ_diel,i`,
- a constant-gradient / mean-concentration extended Nernst–Planck flux inside
  the membrane (diffusion + electromigration + convection), reusing the
  existing ENP utilities,
- residuals for feed-side and permeate-side membrane electroneutrality, the
  zero-current condition `Σ_i z_i J_i = 0`, and per-ion flux closure
  `J_i,ENP − C_p,i J_v = 0`.

Permeate free-solution electroneutrality is reported as a dependent diagnostic
(for nonzero `J_v` it is implied by flux closure plus zero current).

## Residual scaling

The raw two-interface residuals mix units (electroneutrality in mol/L; zero
current and flux closure in mol m⁻² s⁻¹). The solver therefore scales
electroneutrality residuals by a representative concentration and the
zero-current / flux residuals by a representative molar flux, and judges
convergence on the **scaled** residuals — never on the raw mixed-unit norm or on
the optimizer's own termination flag alone.

## Ideal activity assumption

Electrochemical equilibria and transport use concentrations in place of
activities (ideal activities). Concentrated-solution nonideality and
ion-specific excess chemical potentials are not modelled.

## What is baseline vs diagnostic

- **Baseline (default, tested, unchanged):** single-interface partitioning +
  simplified transport, hard steric/hindrance cutoff, ideal activities.
- **Diagnostic (opt-in):** the two-interface solver, the constitutive
  steric/hindrance variants (floor/soft, radius/pore scaling), and the
  parameter-estimation layer. These are exploratory tools; their outputs are
  diagnostic and are **not** a validated physical model.
