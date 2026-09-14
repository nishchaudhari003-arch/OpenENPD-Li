# Two-interface solver: implementation status (Milestone 3)

This note records the scope and honest status of the minimal nonlinear
two-interface solver in `openenpd/two_interface_solver.py`. It is intentionally
conservative. Nothing here claims the two-interface model is validated, and no
physical parameter is fitted.

## What the minimal solver does

- Solves the **state variables** of the algebraic two-interface ENP-Donnan
  system for **one prescribed water flux**, on top of the Milestone-1 state
  assembler and Milestone-2 residual functions.
- Unknown vector for `N` active ions: the `N` permeate concentrations
  `C_p,i` (mol/L) plus three potentials — feed-side Donnan, permeate-side
  Donnan, and the membrane potential drop `Δψ_m`. Length `N + 3`.
- Drives a **scaled core residual vector** to zero using
  `scipy.optimize.least_squares`: feed-side membrane electroneutrality,
  permeate-side membrane electroneutrality, the zero-current condition, and one
  ENP flux-closure residual `J_i,ENP − C_p,i·J_v` per active ion. Permeate
  free-solution electroneutrality can optionally be appended as a documented
  dependent diagnostic.
- Enforces concentration positivity with **direct bounds** on `least_squares`
  (lower bound + an upper bound that is a multiple of the feed concentration),
  and bounds the three potentials symmetrically.
- Eliminates **hard-cutoff species** (steric factor 0) from the unknown vector,
  fixing their permeate concentration to 0, and surfaces them in the result.
- Returns a frozen `TwoInterfaceSolveResult` with a success flag, message,
  scaled and raw residual diagnostics, permeate concentrations, the three
  potentials, predicted rejections, hard-cutoff species, and optimizer
  metadata. Convenience wrappers solve the Foo 2023 LM-C pH 7 case for one flux
  and for all four experimental fluxes.

## What it does not do

- It does **not** fit or tune any physical/model parameter (pore radius, fixed
  charge, dielectric constant, ion radii, diffusivities, hindrance). Only state
  variables are solved.
- It does **not** modify or import the simplified baseline solver
  (`openenpd/solver.py`); the baseline behaviour is unchanged.
- It does **not** add new validation cases or literature data. The controlled
  cases used in tests are synthetic fixtures, clearly labelled as such.
- It does **not** implement concentration polarization, nonideal activities, a
  spatial boundary-value (`solve_bvp`) formulation, or continuation between
  fluxes. These remain future work (design doc §6.2, §10).

## Why `least_squares` (not `root`)

`scipy.optimize.least_squares` is used because it supports bounds (hence
concentration positivity without a log transform), tolerates non-square
residual sets (useful when the permeate-EN diagnostic is included), and is more
robust for an early controlled implementation. `root` requires a square system
and gives no positivity guarantee, so it is deferred.

## Why residual scaling is required

The raw residuals mix units: electroneutrality residuals are in mol/L, while
the zero-current and flux-closure residuals are in mol m⁻² s⁻¹ (numerically
tiny, ~1e-3 and smaller). A raw L2 norm would be dominated by the
concentration-unit entries and would treat a physically large flux imbalance as
negligible, so it is not a safe convergence target. The solver therefore
divides electroneutrality residuals by a concentration scale (total feed
concentration) and the zero-current / flux residuals by a molar-flux scale
(`max(C_ref·J_v, floor)`), and **judges success on the maximum absolute scaled
residual**, never on the raw norm or the optimizer's own termination flag
alone. Both raw and scaled residuals are exposed on the result.

## Controlled-case result

The controlled symmetric monovalent case (equal diffusivities, zero fixed
charge, large pore, pore dielectric equal to bulk) converges: the maximum
absolute scaled residual reaches ~1e-11, both potentials and the membrane drop
go to ~0, and the cation/anion permeate concentrations are equal, as symmetry
requires. This confirms the solver infrastructure is working.

## Foo 2023 LM-C pH 7 result (honest, not validated)

For the Foo 2023 case the solver **runs without crashing for every flux but
does not converge** to the scaled tolerance. With Mg²⁺ eliminated as a
hard-cutoff species and the negative fixed charge in place, the reduced Li–Cl
membrane electroneutrality cannot be satisfied to tolerance (max absolute
scaled residual ~1e-2), and the four solves are reported as `success=False`.
The best-effort predicted Li rejection is strongly positive (~0.82) and Mg²⁺
rejection is 1.0 by construction (full size exclusion). **The two-interface
solver as implemented does not reproduce the experimental negative Li
rejection**, and this milestone makes no such claim. This is consistent with
the design document's expectation (§9) that the unchanged hard steric cutoff
may make the coupled interface problem infeasible.

The `success=False` outcome here is the intended, honest behaviour: it
demonstrates why success is judged on scaled residuals rather than the
optimizer's `gtol` termination flag, which reports a local minimum as
"satisfied".

## Known limitations

- Convergence on Foo 2023 is not achieved; the failure is structured and
  reported, not hidden.
- Hard-cutoff elimination is exact-zero, not a steric floor; whether the hard
  cutoff must be replaced is a separate, later physics decision (not made here).
- The constant-gradient / mean-concentration transport approximation is
  inherited from the baseline; a converged solve would prove balance
  consistency for that approximation, not full-profile accuracy.
- Direct concentration bounds (rather than a log transform) keep the MVP simple
  but rely on the potential bounds to prevent Donnan-exponential overflow for
  multivalent non-cutoff ions; the default potential bound is conservative.
- Success depends on the initial guess (deterministic feed-like permeate, zero
  potentials); no multi-start or continuation is implemented yet.

## Next milestone

Milestone 4 — two-interface validation against Foo 2023 LM-C pH 7: compare the
two-interface predictions and residuals against the existing simplified
baseline and the experimental data, reporting RMSE and residual diagnostics
honestly, without tuning parameters to force agreement.
