# Changelog

All notable changes to OpenENPD-Li are documented here. This project is a
diagnostic research/portfolio toolkit; version numbers track the codebase, not a
validated physical model.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project aims to follow semantic-ish versioning for the software API.

## [0.1.0] - Unreleased

First tagged snapshot of the diagnostic toolkit. The physics is diagnostic, not
validated: the current models do not reproduce the experimental negative Li⁺
rejection (see `docs/validation_summary.md`).

### Added

- **Baseline Foo 2023 workflow** — LM-C pH ≈ 7 data loading, Donnan/steric/
  dielectric partitioning, a simplified single-interface ENP solver, rejection
  metrics, validation comparison tables, and backend-safe plots.
- **Two-interface ENP-Donnan infrastructure** — an immutable state assembler,
  transparent residual functions (feed/permeate-side electroneutrality,
  zero-current, ENP flux closure) with explicit scaling, and a minimal bounded
  nonlinear solver that reports failed solves explicitly.
- **Two-interface validation diagnostics** — experiment vs baseline vs
  two-interface comparison rows, metrics, and a solver-diagnostic summary.
- **Constitutive model variants** — configurable hard/floor/soft steric and
  hindrance treatments and effective radius/pore scaling, with the hard baseline
  as the default, plus variant comparison and sensitivity scans.
- **Parameter estimation and identifiability diagnostics** — bounded one- and
  two-parameter fits with degrees-of-freedom, condition-number, correlation, and
  bound-hitting warnings, plus objective scans.
- **Documentation polish** — README, model assumptions, validation summary,
  limitations, release checklist; a runnable example workflow; and a reproducible
  artifact-generation script.

### Notes

- No validated physical model, solved physics, or manuscript readiness is
  claimed.
- Default (baseline) numerical behaviour is unchanged across the diagnostic
  additions; the two-interface, variant, and estimation layers are opt-in.
- Only the Foo 2023 LM-C pH ≈ 7 case is included; no data were fabricated.
