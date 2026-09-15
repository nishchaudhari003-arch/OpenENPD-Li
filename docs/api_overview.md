# API overview

A concise map of the main public entry points. See module docstrings for full
detail. Names below are accurate as of v0.1.0.

## Core physics (baseline building blocks)

- `openenpd.constants` — physical constants (`R_GAS`, `FARADAY`, `EPSILON_0`, `AVOGADRO`).
- `openenpd.units` — unit conversions (`lmh_to_m_s`, `nm_to_m`, `mol_L_to_mol_m3`, …).
- `openenpd.species` — `Ion` definitions and `COMMON_IONS`.
- `openenpd.electroneutrality` — `charge_balance`, `is_electroneutral`.
- `openenpd.donnan` — `donnan_partition_factor`, `solve_donnan_potential`.
- `openenpd.steric` — `size_ratio`, `steric_partition_factor(s)`.
- `openenpd.dielectric` — `born_energy_j_per_mol`, `dielectric_partition_factor(s)`.
- `openenpd.partitioning` — `total_partition_factor(s)`, `solve_partitioning_potential`.
- `openenpd.hindrance` — `hindrance_factors`.
- `openenpd.enp` — `enp_flux`, `enp_fluxes_zero_current`.
- `openenpd.transport` — concentration/thickness helpers and gradients.
- `openenpd.rejection` — `species_rejection(s)`, `separation_factor`.

## Cases and data

- `openenpd.cases.foo2023_lmc_ph7_case()` — the benchmark case definition.
- `openenpd.data` — dataset loading utilities.

## Baseline model and validation

- `openenpd.model.prepare_case_transport_inputs`.
- `openenpd.solver` — simplified single-interface solver
  (`predict_rejection_for_flux`, `predict_rejections_for_fluxes`).
- `openenpd.validation` — `foo2023_lmc_ph7_comparison_rows/_dataframe`,
  `foo2023_lmc_ph7_validation_metrics`.
- `openenpd.diagnostics` — partitioning / radius / steric-entry diagnostics.

## Two-interface model (diagnostic)

- `openenpd.two_interface` — `TwoInterfaceInputs`, `TwoInterfaceState`,
  `assemble_two_interface_state`, `two_interface_inputs_from_case`.
- `openenpd.two_interface_residuals` — residual functions, `residual_vector`,
  `compute_two_interface_residuals`, `TwoInterfaceTransportInputs`.
- `openenpd.two_interface_solver` — `TwoInterfaceSolverOptions`,
  `TwoInterfaceSolveResult`, `solve_two_interface_for_flux`,
  `solve_foo2023_lmc_ph7_flux`, `solve_foo2023_lmc_ph7_all_fluxes`.
- `openenpd.two_interface_validation` — comparison rows/dataframe, metrics, and
  solver diagnostics.

## Constitutive variants (diagnostic)

- `openenpd.constitutive_variants` — `ConstitutiveVariantConfig`,
  `steric_partition_factor_variant`, `hindrance_factor_variant`,
  `variant_foo2023_lmc_ph7_comparison_rows/_metrics`, `default_variants`, and
  sensitivity scans.

## Parameter estimation (diagnostic)

- `openenpd.parameter_estimation` — `ParameterSpec`, `parameter_spec`,
  `EstimationConfig`, `EstimationResult`, `fit_foo2023_parameters`,
  `run_foo2023_single_parameter_fits`, `run_foo2023_two_parameter_fits`,
  `foo2023_parameter_estimation_summary`, `objective_scan_1d`,
  `objective_scan_2d`.

## Plots (backend-agnostic)

- `openenpd.plots` — baseline, two-interface, constitutive-variant, and
  objective-scan diagnostic figures (all accept an `output_path`).

## Examples and scripts

- `examples/foo2023_lmc_ph7_workflow.py` — runnable end-to-end diagnostic tour
  (`main()`).
- `scripts/generate_release_artifacts.py` — reproducible artifact generation
  (`main(output_directory=...)`).
