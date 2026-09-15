# OpenENPD-Li

![Tests](https://github.com/nishchaudhari003-arch/OpenENPD-Li/actions/workflows/tests.yml/badge.svg)

OpenENPD-Li is a tested, reproducible Python toolkit for **extended Nernst–Planck–Donnan (ENP-Donnan) modeling and diagnostics** of membrane-based lithium/brine separations. It provides transparent, unit-tested building blocks for ion partitioning and transport through nanofiltration membranes, plus validation and diagnostic workflows built around the Foo et al. 2023 LM-C pH ≈ 7 Li⁺/Mg²⁺/Cl⁻ benchmark. It is a GitHub/PhD/job **portfolio project**, not a manuscript project.

## What the project currently does

- **Foo 2023 LM-C pH ≈ 7 data handling** — loads the benchmark dataset (pressure, water flux, pH, Li⁺/Mg²⁺ rejection, feed composition, NF270 membrane parameters).
- **Baseline ENP-Donnan modeling** — Donnan, steric, and dielectric/Born partitioning; a single-interface simplified solver; rejection metrics.
- **Two-interface ENP-Donnan model** — an immutable state assembler, transparent residual functions (feed/permeate-side electroneutrality, zero-current, ENP flux closure), and a minimal bounded nonlinear solver (one water flux at a time).
- **Diagnostic constitutive variants** — configurable hard / floor / soft steric and hindrance treatments and effective radius/pore scaling, with the hard baseline as the default.
- **Parameter estimation and identifiability diagnostics** — bounded one- and two-parameter fits over the benchmark, with degrees-of-freedom, condition-number, and bound-hitting warnings, plus objective scans.
- **Validation/diagnostic plots** — backend-agnostic figures for rejection comparisons, residuals, variant metrics, and objective scans.

## Honest scientific status

- A **validation workflow is implemented**, and the solver and diagnostic infrastructure are unit-tested and work on controlled cases.
- The experimental trend is **negative Li⁺ rejection** and positive Mg²⁺ rejection.
- The **simplified baseline over-rejects** both Li⁺ and Mg²⁺.
- Under the hard steric/hindrance assumptions, the **two-interface solves for Foo 2023 do not converge** (Mg²⁺ is size-excluded at the baseline pore radius).
- Diagnostic **constitutive variants** (e.g. effective pore-radius scaling) relax the Mg²⁺ cutoff, let the two-interface solves converge, and lower diagnostic RMSE — but they do **not fix the physics**.
- Parameter estimation finds an **effective pore-radius-scale improvement (best diagnostic RMSE ≈ 0.30)**, but it is **diagnostic only**; two-parameter fits are **weakly identifiable / ill-conditioned**.
- **No current model variant reproduces the experimental negative Li⁺ rejection.**
- Current results should be treated as **diagnostic, not final validation.** This project makes no claim of a validated physical model, solved physics, or manuscript readiness.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest
```

## Quickstart

```python
from openenpd.validation import foo2023_lmc_ph7_validation_metrics
from openenpd.two_interface_validation import (
    two_interface_foo2023_lmc_ph7_solver_diagnostics,
)
from openenpd.constitutive_variants import (
    ConstitutiveVariantConfig,
    variant_foo2023_lmc_ph7_metrics,
)
from openenpd.parameter_estimation import (
    EstimationConfig,
    fit_foo2023_parameters,
    parameter_spec,
)

# 1. Baseline (single-interface) validation metrics.
print(foo2023_lmc_ph7_validation_metrics())

# 2. Two-interface solver validation diagnostics (solve counts, negative-Li flag).
print(two_interface_foo2023_lmc_ph7_solver_diagnostics())

# 3. Small constitutive-variant comparison (hard baseline vs a diagnostic variant).
for config in (
    ConstitutiveVariantConfig(label="hard_baseline"),
    ConstitutiveVariantConfig(pore_radius_scale=1.25, label="pore_radius_scale_1p25"),
):
    print(variant_foo2023_lmc_ph7_metrics(config))

# 4. Small, diagnostic parameter-estimation fit (effective parameter only).
result = fit_foo2023_parameters(
    EstimationConfig(
        parameter_specs=(parameter_spec("pore_radius_scale"),),
        seed_grid_points=3,
        max_nfev=20,
    )
)
print(result.parameter_values, result.total_rmse, result.reproduces_negative_Li_rejection)
```

A ready-to-run version of this tour is in [`examples/foo2023_lmc_ph7_workflow.py`](examples/foo2023_lmc_ph7_workflow.py):

```bash
PYTHONPATH=. python examples/foo2023_lmc_ph7_workflow.py
```

## Repository structure

```
OpenENPD-Li/
├── openenpd/     # library: partitioning, transport, solvers, variants, estimation, plots
├── data/         # Foo 2023 LM-C pH ~7 benchmark dataset
├── docs/         # model assumptions, validation summary, limitations, design notes
├── tests/        # pytest suite (unit + workflow + docs/example smoke tests)
├── scripts/      # reproducible artifact-generation scripts
├── examples/     # runnable end-to-end diagnostic workflow
└── results/      # committed baseline validation artifacts
```

## Reproducibility

- A comprehensive **pytest** suite covers the numerical building blocks and the diagnostic workflows.
- **GitHub Actions** runs the suite on every push and pull request (see the badge above).
- Diagnostic utilities (comparison rows, metrics, scans) are **deterministic**.
- There is **no hidden fitting** in the validation workflow: baseline and two-interface validation report predictions and residuals as-is, and failed solves are always surfaced explicitly.
- Regenerate diagnostic artifacts with:

  ```bash
  PYTHONPATH=. python scripts/generate_release_artifacts.py
  ```

## Limitations

- The partitioning/transport baseline assumes **ideal activities** (no activity coefficients).
- There is **no full concentration-polarization model**.
- The hard/floor/soft cutoff and radius/pore-scaling variants are **diagnostic**, not validated physics.
- Only the **Foo 2023 LM-C pH ≈ 7** case is currently included.
- **Negative Li⁺ rejection remains unreproduced** by all current model variants.

See [`docs/limitations.md`](docs/limitations.md) for details and optional future-work directions, and [`docs/validation_summary.md`](docs/validation_summary.md) for the full diagnostic status.

## Citation

The validation dataset is based on:

Foo, Z. H.; Rehman, D.; Bouma, A. T.; Monsalvo, S.; Lienhard, J. H.
*Lithium Concentration from Salt-Lake Brine by Donnan-Enhanced Nanofiltration.*
Environmental Science & Technology, 2023, 57, 6320–6330.
DOI: [10.1021/acs.est.2c08584](https://doi.org/10.1021/acs.est.2c08584)
