# Constitutive model variants (Milestone 5)

Diagnostic study of steric/hindrance constitutive variants for the Foo 2023
LM-C pH 7 case. This is the first physics-improvement milestone, but it remains
**diagnostic**: nothing here is tuned to force agreement, and the corrected
model is **not** claimed to be physically true or validated.

Reproduce with:

```python
from openenpd.constitutive_variants import (
    default_variants,
    variant_foo2023_lmc_ph7_metrics,
    run_foo2023_lmc_ph7_variant_comparison,
    scan_ion_radius_scale, scan_pore_radius_scale,
    scan_steric_floor, scan_hindrance_floor,
)
```

## Why the hard steric/hindrance cutoff is problematic

The baseline steric partition factor and the diffusive/convective hindrance
factors are `(1 - λ)²` for `λ = r_i / r_p < 1` and **exactly 0** for `λ >= 1`.
For Foo 2023, Mg²⁺ (0.428 nm) in the NF270 pore (0.416 nm) has `λ ≈ 1.03`, so it
is fully size-excluded: its steric factor, membrane concentration, and ENP flux
are all zeroed. The two-interface solver therefore eliminates Mg²⁺ from the
active problem, and the remaining Li–Cl system cannot satisfy membrane
electroneutrality — every Foo flux fails to converge (Milestone 4). The abrupt
zero at `λ = 1` is also numerically fragile.

## What variants were added

A single `ConstitutiveVariantConfig` selects, independently for steric and
hindrance:

- **`"hard"`** (default): the existing behaviour, `(max(1-λ, 0))²`.
- **`"floor"`**: `max(hard, floor)` — a minimum positive factor.
- **`"soft"`**: a smooth, everywhere-positive softplus replacement.

Plus parameter-level knobs: `ion_radius_scale`, `pore_radius_scale` /
`pore_radius_nm_override`, `steric_floor`, `hindrance_floor`,
`soft_cutoff_width`. The default config reproduces the hard baseline exactly
(verified against the Milestone-1 assembler and the Milestone-3 solver in the
tests).

### Equations / rules (as functions of the effective size ratio)

`λ_eff = (ion_radius_scale · r_i) / r_p_eff`, where `r_p_eff` is
`pore_radius_nm_override` if set, else `pore_radius_scale · r_p`.

- hard: `s(λ_eff) = (max(1 - λ_eff, 0))²`
- floor: `max(s(λ_eff), floor)`
- soft: `s_soft(λ_eff) = (w · ln(1 + exp((1 - λ_eff)/w)))²`, `w = soft_cutoff_width`,
  then `max(s_soft, floor)`. As `w → 0`, `s_soft → s`.

The effective ion radius also rescales the Born/dielectric penalty, so radius
scaling is applied consistently. Hard-cutoff is reported when the effective
steric factor is exactly 0 (never under floor/soft).

## Sensitivity scans implemented

Deterministic scans over `ion_radius_scale`, `pore_radius_scale`,
`steric_floor`, and `hindrance_floor`, each returning per-value diagnostic
metrics (RMSE, solve counts, Mg hard-cutoff status, negative-Li flag). Default
scan grids are provided; tests use short grids for speed.

## Was the Mg²⁺ hard cutoff relaxed?

**Yes, by several variants** — but relaxation means different things:

- `steric_floor` (floor > 0) makes the steric factor positive, so Mg²⁺ is no
  longer flagged as steric-hard-cutoff.
- `ion_radius_scale < 1`, `pore_radius_scale > 1`, or a `pore_radius_nm_override`
  push `λ_eff < 1` geometrically, also removing the cutoff.
- `hindrance_floor` alone does **not** change the steric hard-cutoff flag (Mg²⁺
  stays steric-excluded).

## Did any variant reproduce negative Li rejection?

**No.** Across the default variants and the scans, no variant produced a
negative predicted Li rejection. Experiment shows Li rejection ≈ −0.13 to −0.21;
the variants predict positive Li rejection. Relaxing the cutoff changed
convergence and RMSE but not the qualitative Li sign.

## Diagnostic RMSE results (current run, default variants)

Total = pooled Li+Mg rejection RMSE vs experiment. "best-effort" means at least
one solve did not converge; those RMSE values are diagnostic only.

| Variant | Successful solves | Mg hard cutoff | Negative Li | Total RMSE | RMSE label |
|---|---|---|---|---|---|
| hard_baseline | 0 / 4 | present | no | ~0.75 | best-effort |
| steric_floor_0p01 | 0 / 4 | relaxed | no | ~0.83 | best-effort |
| hindrance_floor_0p01 | 0 / 4 | present | no | ~0.75 | best-effort |
| ion_radius_scale_0p85 | 4 / 4 | relaxed | no | ~0.54 | from successful solves |
| pore_radius_scale_1p25 | 4 / 4 | relaxed | no | ~0.32 | from successful solves |
| combined_diagnostic | 4 / 4 | relaxed | no | ~0.41 | from successful solves |

### How to interpret best-effort RMSE

A best-effort RMSE (`rmse_is_best_effort_diagnostic=True`) is computed from
non-converged solves and is **not** a validation result — it only says how far a
non-solution sits from experiment. Even the variants that converge (radius/pore
scaling, `rmse_is_best_effort_diagnostic=False`) are **not validated**: they do
not reproduce the negative Li rejection, so a lower RMSE here reflects a
less-wrong positive-Li prediction, not correct physics.

## What remains physically uncertain

- Whether the true effective transport radius of Mg²⁺/Li⁺ inside the pore is
  smaller than the bulk hydrated radius (motivating `ion_radius_scale`), or the
  effective pore is larger than reported (motivating `pore_radius_scale`) — the
  data here cannot distinguish these.
- Why negative Li rejection is not recovered: it likely requires coupling
  effects (or nonideal/activity terms, concentration polarization) not yet in
  the model, not merely a softened cutoff.
- Whether any single variant is physically meaningful, or whether several
  parameters trade off against each other (an identifiability question).

## Why these variants are diagnostic, not validated

They are controlled relaxations of one modelling assumption, evaluated without
fitting. Some improve convergence and RMSE, but none reproduce the key
qualitative observation (negative Li rejection), and four rejection points
cannot justify choosing one relaxation as "true". Calling any of them validated
would overclaim.

## What the results suggest for the next milestone

Geometry-level relaxations (effective radius / pore radius) are what let the
two-interface solver converge, while floors alone do not. But convergence plus
lower RMSE is still not the observed physics. This motivates **Milestone 6 —
parameter estimation and identifiability prototype**: estimate a small number of
these effective parameters with bounds, quantify how poorly four data points
constrain them, and report identifiability honestly — without declaring
validation.
