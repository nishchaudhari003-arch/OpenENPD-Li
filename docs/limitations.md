# Limitations and future work

This document lists the current limitations of OpenENPD-Li and optional future
development directions. The future-work list is a menu of possible next steps,
**not** a manuscript plan or a commitment.

## Model limitations

- **Ideal activities.** Electrochemical equilibria and transport use
  concentrations in place of activities; concentrated-solution nonideality is
  absent.
- **Hard steric/hindrance cutoff (baseline).** An ion at or above the pore
  radius is fully excluded, which drives the Mg²⁺ hard cutoff and the
  two-interface convergence failure on Foo 2023.
- **Constant-gradient transport approximation.** The two-interface flux uses a
  linear membrane concentration profile and a mean concentration, not a
  resolved spatial profile.
- **Diagnostic variants are not validated physics.** The floor/soft cutoff and
  radius/pore-scaling variants are exploratory relaxations, not established
  constitutive laws.

## Data limitations

- Only the **Foo 2023 LM-C pH ≈ 7** case is included. No pH 2, LM-S sulfate, or
  full multicomponent brine data are present, and none are fabricated.
- Four water fluxes give at most **eight rejection residuals** — a small dataset
  for estimation.

## Identifiability limitations

- With eight rejection points, only **one or two** effective parameters can be
  fitted, and two-parameter fits are **weakly identifiable / ill-conditioned**
  (overlapping parameter effects, bound-hitting, near-singular Jacobians).
- Fitted values are **effective diagnostic parameters**, not unique physical
  estimates.

## Numerical limitations

- The two-interface objective is **discontinuous** where the hard cutoff
  releases; the solver mitigates this with a deterministic grid seed, but the
  landscape remains non-convex.
- Convergence is judged on scaled residuals; some configurations do not converge
  and are reported as explicit failures.

## Physical mechanisms not yet included

- Activity coefficients / nonideal thermodynamics.
- Concentration polarization (feed- and permeate-side boundary layers).
- More detailed steric/hindrance (e.g. Deen/DSPM-DE) correlations.
- pH- and speciation-dependent effects.

## Optional future work

These are possible next development directions, in no particular order and with
no timeline:

- Nonideal activity coefficients.
- A concentration-polarization model.
- Sulfate and pH-dependent cases — **only** after documented extraction of the
  required data into the repository (no fabricated values).
- Improved steric/hindrance correlations.
- An expanded validation dataset.
- Uncertainty quantification for the estimation layer.

The most direct open scientific question remains: **no current model variant
reproduces the experimental negative Li⁺ rejection**, which likely requires
physics beyond a softened cutoff (e.g. nonideal activities or coupling effects
not yet modelled).
