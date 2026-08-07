# Foo 2023 LM-C pH ~7 baseline evaluation

## Scope

This note records the first controlled comparison between the current
OpenENPD-Li solver and the Foo et al. (2023) LM-C pH ~7 dataset. The calculation
is a **baseline ideal ENP-Donnan implementation**. It is useful as an integrated,
reproducible software baseline, but it is not an adequate predictive model for
this validation case.

The source data and generated artifacts are available in
[`data/foo2023_lmc_ph7_full.csv`](../data/foo2023_lmc_ph7_full.csv) and
[`results/foo2023_lmc_ph7/`](../results/foo2023_lmc_ph7/), respectively.

## Dataset summary

The dataset contains four NF270 operating points at 293.15 K. Pressure ranges
from 6 to 12 bar, water flux from 28.98 to 67.18 LMH, and measured pH from 6.71
to 6.94. The case definition uses feed concentrations of 0.0490 mol/L Li+, 0.0843
mol/L Mg2+, and 0.2172 mol/L Cl-. Experimental Li rejection is negative at all
four points (-0.207 to -0.125), while experimental Mg rejection ranges from
0.521 to 0.653. The repository identifies the source as Foo et al. (2023), DOI
`10.1021/acs.est.2c08584`.

## Current model assumptions

The implemented baseline uses the following assumptions and approximations:

- ideal solution activities; concentration nonideality is not included;
- a single, pressure-independent set of membrane parameters from the case
  definition;
- feed-side membrane partitioning from Donnan, steric, and Born/dielectric
  factors, with membrane electroneutrality used to solve the Donnan potential;
- identical baseline diffusive and convective hindrance factors,
  `(1 - ion radius / pore radius)^2`, with complete hindrance when the ion
  radius is at least the pore radius;
- linear concentration profiles across the active layer and arithmetic mean
  concentrations for evaluating local fluxes;
- a zero-current electric-field condition;
- the permeate closure `J_i = C_i,p J_v`;
- no explicit external concentration-polarization film, despite crossflow
  velocity being stored in the case definition;
- no parameter fitting, uncertainty propagation, or independent validation in
  this comparison.

## Predicted versus experimental rejection

Residuals are defined as predicted minus experimental rejection.

| Pressure (bar) | Water flux (LMH) | Li exp. | Li pred. | Li residual | Mg exp. | Mg pred. | Mg residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 28.98 | -0.207 | 0.989 | 1.196 | 0.521 | 1.000 | 0.479 |
| 8 | 42.18 | -0.184 | 0.990 | 1.174 | 0.600 | 1.000 | 0.400 |
| 10 | 54.17 | -0.142 | 0.991 | 1.133 | 0.599 | 1.000 | 0.401 |
| 12 | 67.18 | -0.125 | 0.991 | 1.116 | 0.653 | 1.000 | 0.347 |

The corresponding dimensionless rejection RMSE values are:

- Li: **1.1553**
- Mg: **0.4095**

## Assessment

The current simplified solver is not physically adequate for the Foo 2023
LM-C pH ~7 case. It predicts nearly complete Li rejection and therefore misses
both the magnitude and the negative sign of the measured Li rejection. It also
predicts complete Mg rejection at every operating point and overpredicts the
measured Mg rejection by 0.347 to 0.479.

One direct contributor is the baseline hard size cutoff: the case-defined Mg
ion radius (0.428 nm) exceeds the case-defined pore radius (0.416 nm), so the
current steric and hindrance functions set Mg entry and transport factors to
zero. Li is also close to the hard cutoff. This makes the predictions highly
sensitive to the effective-radius and pore-radius definitions and prevents the
current baseline from representing the observed partial Mg passage and enhanced
Li passage. This diagnosis does not establish that the stored parameters are
wrong; it shows that the present combination of parameters and simplified
constitutive relations is inadequate.

These results should be described as a software-integrated baseline and a first
controlled comparison, not as successful validation.

## Limitations

- Only four operating points from one membrane, composition, and pH condition
  are evaluated.
- Activity coefficients and electrochemical-potential nonideality are omitted
  for the concentrated multicomponent feed.
- The linear-profile approximation is not a full one-dimensional boundary-value
  solution.
- Only the feed-side partitioned state is supplied explicitly to the transport
  solver; a complete two-interface membrane formulation is not implemented.
- The hard steric cutoff and baseline hindrance expressions have not been
  validated for this case.
- Concentration polarization and boundary-layer mass transfer are omitted.
- The nonlinear root solve is unconstrained and does not explicitly enforce
  nonnegative trial concentrations.
- Membrane parameters are treated as fixed, without sensitivity,
  identifiability, or uncertainty analysis.

## Next required physics improvements

1. Replace the baseline hard-cutoff hindrance treatment with a documented,
   validated pore-transport correlation suitable for the intended DSPM/ENP
   formulation. Audit the provenance and interpretation of effective ion radii
   and pore radius before changing any values.
2. Implement the full one-dimensional coupled ENP boundary-value problem with
   concentration and electric-potential profiles, electroneutrality, and
   partitioning conditions at both membrane interfaces.
3. Add a nonideal activity model appropriate for concentrated Li-Mg-Cl brines.
   The activity model and any required parameters remain **TODO** pending a
   documented scientific choice; no values are assumed here.
4. Add an external mass-transfer/concentration-polarization model once the
   required cell geometry and correlation inputs are documented. The stored
   crossflow velocity alone is insufficient to select those parameters.
5. Add bounded or transformed concentration unknowns and explicit physical
   solution checks to the nonlinear solver.
6. After the governing equations are upgraded, run sensitivity and
   identifiability analyses before fitting membrane charge, pore size,
   dielectric constant, or active-layer thickness. Reserve independent data for
   validation rather than fitting and validation on the same four points.

Until these steps materially improve the Li sign/magnitude and the Mg partial
rejection trend, the model should not support manuscript claims of quantitative
validation or membrane-specific predictive performance.
