# Parameter estimation and identifiability (Milestone 6)

Bounded estimation of a small number of **effective diagnostic parameters** for
the Foo 2023 LM-C pH 7 case, with identifiability diagnostics. This is **not a
validation-success milestone** and **not parameter tuning for pretty plots**.
The aim is an honest answer to two questions: can one or two effective
parameters improve agreement with experiment, and are they identifiable from
only four flux points?

Reproduce with:

```python
from openenpd.parameter_estimation import (
    parameter_spec, EstimationConfig, fit_foo2023_parameters,
    run_foo2023_single_parameter_fits, run_foo2023_two_parameter_fits,
    foo2023_parameter_estimation_summary,
    objective_scan_1d, objective_scan_2d,
)
```

## Why parameter estimation is being added

Milestone 5 showed that relaxing the hard steric/hindrance cutoff (via effective
radius or pore scaling) lets the two-interface solver converge and lowers RMSE,
but no variant reproduced the experimental negative Li rejection. This milestone
asks the natural follow-up quantitatively: fit a few of those effective
parameters to the data, and measure how well the four-point dataset constrains
them.

## What data are used

The Foo 2023 LM-C pH 7 experimental Li and Mg rejection at the four water
fluxes. For each fitted species and flux the residual is
`R_i,predicted - R_i,experimental`. With Li and Mg over four fluxes there are at
most `2 x 4 = 8` rejection residuals.

## What parameters are allowed

Each maps onto a `ConstitutiveVariantConfig` field or a copy of the case's
membrane parameters, with explicit bounds:

| Parameter | Target | Bounds |
|---|---|---|
| `ion_radius_scale` | variant | 0.40 - 1.20 |
| `pore_radius_scale` | variant | 0.80 - 2.50 |
| `steric_floor` | variant (forces `steric_model="floor"`) | 0.0 - 0.30 |
| `hindrance_floor` | variant (forces `hindrance_model="floor"`) | 0.0 - 0.30 |
| `fixed_charge_scale` | case (scales fixed charge) | 0.25 - 2.00 |
| `dielectric_scale` | case (scales pore dielectric) | 0.50 - 1.50 |

Negative physical scales are rejected by `ParameterSpec` validation.

## Why only one- and two-parameter fits by default

Eight rejection points cannot constrain many parameters. The code fits **one or
two parameters at a time** by default; asking for three or more raises an
identifiability warning, and degrees of freedom (`8 - n_parameters`) are
reported on every result. Fitting more than two parameters here would overfit.

## How failed solves are treated

Failed solves are never hidden. The objective is a fixed-length vector: the
eight rejection residuals (best-effort by default) followed by one penalty slot
per flux that equals `failed_solve_penalty` when that flux's solve did not
converge. Every result reports `number_of_successful_solves` /
`number_of_failed_solves` and whether its RMSE is best-effort
(`rmse_is_best_effort`). A fit whose model solves all fail is reported
`success=False` and is not treated as a usable result.

Because the hard-cutoff release makes the objective discontinuous, the local
`least_squares` fit is seeded from a deterministic coarse grid over the bounds
(the penalty pushes the seed toward converging regions) and then refined. This
is a documented robustness step, not tuning toward the experimental target.

## Why identifiability is limited with four flux points

Four fluxes give at most eight rejection residuals, and several candidate
parameters have overlapping effects (for example `ion_radius_scale` and
`pore_radius_scale` both change the size ratio, so they trade off). With so few
points these parameters are only weakly identifiable, and the diagnostics
(degrees of freedom, Jacobian condition number, parameter correlation, and
bound-hitting warnings) are there to make that explicit rather than to be
overcome.

## What the best diagnostic fits show (current run)

Representative results (grid-seeded bounded `least_squares`):

- **Best one-parameter diagnostic fit:** `pore_radius_scale ≈ 1.31`,
  total rejection RMSE ≈ 0.30 (Li ≈ 0.36, Mg ≈ 0.23), 4/4 model solves
  converged, no parameter on a bound.
- **Best two-parameter diagnostic fit:** collapses to essentially the same
  `pore_radius_scale ≈ 1.31`, with the second parameter (`hindrance_floor`)
  driven to its lower bound (~0) and an **ill-conditioned Jacobian
  (condition number → ∞)**. In other words, the second parameter adds nothing
  the data can constrain — a clear weak-identifiability signal, with the
  bound-hitting and ill-conditioning warnings raised automatically.

So a single effective parameter already captures whatever these data can
constrain; a second one is not identifiable from four flux points.

## Whether negative Li rejection is reproduced

**No.** No one- or two-parameter fit, and no scan point, produced a negative
predicted Li rejection. Every result reports this via
`reproduces_negative_Li_rejection`. A lower RMSE that still predicts positive Li
rejection is **not** evidence of correct physics — the experimental sign of Li
rejection is not recovered.

## Limitations

- Fitted values are **effective diagnostic parameters**, not unique physical
  estimates, and not validation success.
- The best RMSE (~0.30) is a diagnostic improvement over the hard baseline, not
  agreement: the qualitative negative-Li behaviour is still missing.
- Identifiability is weak: overlapping parameter effects, few data points, and
  (for two-parameter fits) ill-conditioned or bound-hitting solutions.
- The objective is discontinuous at the hard-cutoff release; the grid seed makes
  the local fit robust but the landscape remains non-convex.
- Ideal activities, the constant-gradient transport approximation, and the
  absence of concentration polarization are unchanged; the missing physics for
  negative Li rejection likely lies there, not in these effective parameters.

## Next recommended milestone

Milestone 7 — release-ready documentation and GitHub polish: consolidate the
diagnostic findings, README, and reproducibility notes into a clean portfolio
presentation, being careful to keep every claim conservative and supported by
the actual outputs (no validation or manuscript-readiness claims).
