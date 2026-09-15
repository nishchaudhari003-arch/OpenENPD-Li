# Two-interface validation summary (Milestone 4)

Validation workflow implemented; current model failure documented. This note
records what the Milestone-4 diagnostics compare and what the current numbers
actually show for the Foo 2023 LM-C pH 7 case. It uses conservative language on
purpose: the two-interface model is **not** validated, and it does **not**
reproduce Foo 2023 under the current constitutive assumptions.

Reproduce with:

```python
from openenpd.two_interface_validation import (
    two_interface_foo2023_lmc_ph7_comparison_dataframe,
    two_interface_foo2023_lmc_ph7_metrics,
    two_interface_foo2023_lmc_ph7_solver_diagnostics,
)
```

## What was compared

Per experimental water flux (four fluxes), the workflow places three sources
side by side:

1. **Experiment** — Foo 2023 LM-C pH 7 Li and Mg rejection data from the case
   definition.
2. **Simplified baseline** — the existing single-interface solver
   (`openenpd/solver.py`) via `openenpd.validation`.
3. **Two-interface solver** — the minimal nonlinear solver
   (`openenpd/two_interface_solver.py`), reporting best-effort predicted
   rejections plus solver diagnostics (success flag, message, scaled/raw
   residuals, hard-cutoff species).

## What the simplified baseline does

The baseline establishes equilibrium only at the feed/membrane interface and
transports ions toward the permeate with a linear-profile closure. For this
case it predicts **strongly positive Li rejection** (~0.99) and **full Mg
rejection** (1.0, driven by the hard steric cutoff). Experiment shows negative
Li rejection and only partial Mg rejection, so the baseline over-predicts both.

## What the two-interface solver does

The two-interface solver adds the permeate-side interface and solves the
coupled electroneutrality / zero-current / flux-closure residuals for the
active ions (Milestone 3). Mg²⁺ is a hard-cutoff (fully size-excluded) species
at the baseline pore radius and is eliminated from the active unknowns, so its
predicted rejection is 1.0 by construction.

## Did the two-interface solves succeed?

**No.** For all four Foo 2023 fluxes the solver runs without crashing but does
**not** converge to the scaled tolerance:

- successful solves: **0 of 4**
- maximum absolute scaled residual per flux: ~1.13e-2 to ~1.21e-2
- hard-cutoff species (every flux): `("Mg2+",)`

Because there are zero successful solves, the successful-only RMSE metrics are
reported as `None`, and the summary states plainly that no validated
two-interface predictions were obtained. The two-interface rejection numbers
below are **best-effort diagnostics from non-converged solves**, not validated
predictions.

## Was negative Li rejection reproduced?

**No.** The best-effort two-interface Li rejection is ~+0.82 across all fluxes,
while experiment is negative (~-0.13 to -0.21). No two-interface prediction —
best-effort or from a successful solve — reproduces the experimental negative
Li rejection.

## Metrics (current run)

RMSE against experiment (dimensionless rejection units):

| Metric | Baseline | Two-interface best-effort (diagnostic) | Two-interface successful-only |
|---|---|---|---|
| Li rejection RMSE | ~1.16 | ~0.98 | None (no successful solves) |
| Mg rejection RMSE | ~0.41 | ~0.41 | None (no successful solves) |
| Total RMSE | ~0.87 | ~0.75 | None (no successful solves) |

The two-interface Mg RMSE equals the baseline Mg RMSE because both fully reject
Mg²⁺ (hard cutoff). The lower two-interface best-effort Li RMSE is **not** a
validation success: it comes from non-converged solves and is only slightly
less wrong than the baseline (both predict positive Li rejection where
experiment is negative).

## What the current failure implies

The two-interface **infrastructure is tested and works** (it converges on a
controlled symmetric case), but the **current constitutive assumptions remain
insufficient** for Foo 2023 LM-C pH 7. Specifically, the hard steric cutoff
forces Mg²⁺ out of the active problem and pins its rejection at 1.0, and the
remaining Li–Cl system cannot satisfy membrane electroneutrality to tolerance.
Adding the second interface alone — with the unchanged hard cutoff — does not
recover the negative Li rejection. This is consistent with the two-interface
design document's stated risk that the hard cutoff may make the coupled problem
infeasible.

## Next recommended milestone

Milestone 5 — constitutive model variants for steric/hindrance treatment:
introduce and diagnose alternative steric/hindrance formulations (e.g. a steric
floor or a softer entry function) to test whether relaxing the hard cutoff lets
the two-interface solves converge and whether the physics moves toward the
observed rejection behaviour — without tuning parameters to force agreement.
