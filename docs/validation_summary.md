# Validation summary

This document consolidates the diagnostic status of OpenENPD-Li on its single
benchmark case. Its conclusion is conservative: validation and diagnostic
**workflows** are implemented and tested, but the current physics is **not
validation success**.

## Validation case

Foo et al. 2023 LM-C pH ≈ 7 (simplified Li⁺–Mg²⁺–Cl⁻ brine, NF270 membrane),
four water fluxes. Source dataset committed under `data/`. DOI
[10.1021/acs.est.2c08584](https://doi.org/10.1021/acs.est.2c08584).

## Experimental trend

Li⁺ rejection is **negative** (≈ −0.13 to −0.21 across the four fluxes) and
Mg²⁺ rejection is **positive** (≈ 0.52 to 0.65). Negative Li⁺ rejection is the
key qualitative feature a model would need to reproduce.

## Baseline result (single interface)

The simplified baseline **over-rejects** both ions: it predicts strongly
positive Li⁺ rejection (~0.99) and full Mg²⁺ rejection (1.0, driven by the hard
steric cutoff). Baseline rejection RMSE is large (Li ≈ 1.16, Mg ≈ 0.41).

## Two-interface result

The two-interface solver infrastructure is unit-tested and converges on a
controlled symmetric case. On Foo 2023, under the hard steric/hindrance
assumptions, Mg²⁺ is size-excluded and **all four solves fail to converge** to
the scaled tolerance. Failed solves are reported explicitly, never hidden. The
current assumptions are insufficient for this case.

## Constitutive variants

Diagnostic variants that relax the hard cutoff (effective ion-radius or
pore-radius scaling) remove the Mg²⁺ hard cutoff and let all four two-interface
solves converge, lowering diagnostic RMSE (to roughly 0.32–0.54 total). Steric-
or hindrance-floor-only variants relax the cutoff flag but do not by themselves
produce converged solves. Importantly, **no variant reproduces the experimental
negative Li⁺ rejection** — the improvement is convergence and RMSE, not the
qualitative physics.

## Parameter estimation

A bounded, grid-seeded fit finds a best single **effective diagnostic
parameter**, pore-radius scale ≈ 1.31, giving total rejection RMSE ≈ 0.30 with
all four solves converged. This is an effective parameter, not a unique physical
estimate.

## Two-parameter identifiability limitations

Two-parameter fits are **weakly identifiable**: the best pair collapses onto the
single pore-radius-scale solution, with the second parameter driven to a bound
and an ill-conditioned (near-singular) Jacobian. Four flux points (eight
rejection residuals) cannot constrain two overlapping effective parameters.

## Conclusion

- Data handling, baseline modelling, two-interface assembly/residuals/solver,
  constitutive-variant analysis, parameter estimation, and identifiability
  diagnostics are **implemented and tested**.
- The baseline over-rejects; the hard-cutoff two-interface model fails to
  converge; diagnostic variants improve convergence and RMSE but **do not
  reproduce negative Li⁺ rejection**; effective-parameter fits are diagnostic
  and weakly identifiable.
- Therefore the current state is **not validation success** and not a validated
  physical model. Results should be interpreted as diagnostics that motivate
  further physics (see `docs/limitations.md`).
