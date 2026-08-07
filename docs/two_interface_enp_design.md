# Two-interface ENP-Donnan solver design

## Scope

This document specifies the next solver architecture for the Foo 2023 LM-C pH 7 case. It is a design, not a change to the current model. The existing one-interface solver, parameter values, partitioning functions, hindrance functions, and validation cases remain the baseline.

The immediate implementation target is a two-interface, constant-gradient extended Nernst-Planck (ENP)-Donnan model. It is deliberately the smallest system that can solve feed-side and permeate-side partitioning together. A spatial boundary-value formulation is a later refinement after the algebraic system is verified.

## 1. Motivation

The current Foo 2023 LM-C pH 7 calculation predicts strongly positive Li rejection and overpredicts Mg rejection, whereas the experiment reports negative Li rejection and only partial Mg rejection. The diagnostic results identify two related problems:

- the present hydrated-radius treatment makes Li transport extremely hindered and gives Mg a hard zero steric-entry factor at the baseline pore radius;
- increasing the pore radius or softening entry in diagnostic sweeps reduces the severity of exclusion but does not reproduce negative Li rejection.

Those results show that changing one local factor is not enough. The current solver establishes equilibrium only at the feed/membrane interface and then transports ions toward a permeate concentration without imposing a second membrane/solution equilibrium. It therefore cannot represent the coupled redistribution produced by two Donnan jumps, the membrane electric field, permeate composition, and zero current. Negative Li rejection is a multicomponent coupling effect that may require this full set of constraints rather than a weaker feed-side entry penalty alone.

The next physics upgrade should consequently add the permeate-side interface and solve all electrochemical constraints simultaneously. This is a structural diagnostic step; it does not guarantee improved agreement, and the existing hard cutoff may still prevent a physical solution.

## 2. Current baseline limitations

The current simplified solver has the following limitations for this case:

1. **One feed-side interface only.** It solves a combined Donnan, steric, and dielectric partitioning problem between bulk feed and the membrane entrance. There is no corresponding membrane/permeate equilibrium.
2. **Linear concentration-profile approximation.** It replaces each membrane profile with an endpoint difference over the membrane thickness and evaluates electromigration and convection at an arithmetic mean concentration.
3. **No permeate-side Donnan partitioning.** The exit concentration inside the membrane is not connected to the bulk permeate by a second electrochemical partition relation.
4. **No simultaneous solution of permeate concentrations and interface potentials.** The feed-side potential is solved first; permeate concentrations are then obtained from a reduced transport closure.
5. **Hard steric and hindrance treatment.** A hydrated ion at or above the pore radius has a zero steric factor and zero hindrance. Near the cutoff, the factors can be extremely small. In the baseline LM-C case this behavior dominates Mg entry and strongly suppresses Li transport.
6. **Ideal activity coefficients.** Concentration is used in place of activity, so concentrated-solution nonideality and ion-specific excess chemical potentials are absent.
7. **No concentration polarization.** The bulk feed concentration is applied directly at the membrane surface, and no feed- or permeate-side boundary layer is solved.

The two-interface minimum viable implementation (MVP) below fixes items 1, 3, and 4. It retains the other approximations initially so that the effect of the structural change can be isolated.

## 3. Proposed two-interface model structure

### 3.1 Coordinates and state variables

Let the membrane occupy \(0 \le x \le L\), with feed at \(x=0\) and permeate at \(x=L\). For each ion \(i\):

- \(C_{f,i}\): bulk feed concentration. This is prescribed by the validation case.
- \(C_{m,0,i}\): membrane-phase concentration immediately inside the feed-side interface, \(x=0^+\).
- \(C_{m,i}(x)\): membrane-phase concentration profile.
- \(C_{m,L,i}\): membrane-phase concentration immediately inside the permeate-side interface, \(x=L^-\).
- \(C_{p,i}\): well-mixed bulk permeate concentration.

The fluxes and potentials are:

- \(\Delta\psi_{D,f}=\psi_m(0)-\psi_f\): feed-side Donnan potential jump, defined from feed solution to membrane.
- \(\Delta\psi_{D,p}=\psi_m(L)-\psi_p\): permeate-side Donnan potential jump, defined from permeate solution to membrane. Using the same solution-to-membrane convention at both sides prevents an accidental inversion of the exit partition relation.
- \(\psi_m(x)\): electric potential inside the membrane.
- \(E_m=d\psi_m/dx\): membrane electric-potential gradient. For the algebraic MVP, \(E_m\approx\Delta\psi_m/L\), where \(\Delta\psi_m=\psi_m(L)-\psi_m(0)\).
- \(J_i\): molar flux of ion \(i\), positive from feed to permeate.
- \(J_v\): prescribed water-volume flux, positive from feed to permeate.

Only potential differences are observable. Set \(\psi_f=0\) as the gauge; do not include an absolute potential as an unknown. With the definitions above,

\[
\psi_p-\psi_f=\Delta\psi_{D,f}+\Delta\psi_m-\Delta\psi_{D,p}.
\]

For a pressure-driven, electrically open membrane, this overall streaming potential is an output. The zero-current condition determines the required membrane field.

### 3.2 Partition factors

Define the non-electrostatic entry factor

\[
\Phi_i=\phi_{s,i}\,\phi_{\mathrm{diel},i},
\]

where \(\phi_{s,i}\) is the current steric factor and \(\phi_{\mathrm{diel},i}\) is the current dielectric/Born factor evaluated with the same case parameters. The Donnan exponential is kept separate because its potential differs at the two interfaces.

The MVP uses ideal activities. A later nonideal version should replace concentrations in electrochemical equilibrium with activities \(a_i=\gamma_i C_i\), including the appropriate solution-to-membrane activity-coefficient ratio.

## 4. Unknowns for one water flux

For \(N\) actively transported ions at one prescribed \(J_v\), the recommended algebraic unknown vector is

\[
\mathbf y=
\left[
u_{p,1},\ldots,u_{p,N},
\Delta\psi_{D,f},
\Delta\psi_{D,p},
\Delta\psi_m
\right],
\qquad
C_{p,i}=C_{\mathrm{ref}}\exp(u_{p,i}).
\]

This gives \(N+3\) unknowns. Log concentrations enforce \(C_{p,i}>0\) during nonlinear iterations. \(C_{\mathrm{ref}}\) is a fixed concentration scale, such as the total feed ionic concentration or 1 mol/L expressed in the solver's internal units.

Before packing this vector, classify any species for which the unchanged hard-cutoff model makes both partitioning and transport identically zero. Its equations force \(C_{m,0,i}=C_{m,L,i}=J_i=C_{p,i}=0\) for nonzero \(J_v\). Store those exact zeros in the full result, but exclude their log concentration and tautological flux-closure equation from the active nonlinear system. In the equation counts below, \(N\) means the size of this active set. This is algebraic elimination of the current model's exact constraint, not a steric floor or parameter change. If the remaining active ions cannot satisfy either membrane electroneutrality condition, the preflight or solve must report infeasibility.

The two interface concentrations are dependent variables computed from the partition relations. The ion fluxes should also be dependent variables in the first implementation: calculate \(J_i^{\mathrm{ENP}}\) from the transport equation and compare it with \(C_{p,i}J_v\). This avoids adding \(N\) unnecessary flux unknowns.

An equivalent formulation may include \(J_i\) as explicit unknowns, but then it must add both \(N\) ENP equations and \(N\) permeate-flux equations. The equation count must be documented and tested so that the nonlinear system remains square.

## 5. Governing equations

All concentrations in the equations below must use one consistent unit system. The existing code should continue to own unit conversion rather than embedding conversion factors in the residual function. Let \(z_i\) be ion charge, \(F\) Faraday's constant, \(R\) the gas constant, \(T\) temperature, and \(X\) the signed fixed-charge concentration in the membrane.

### 5.1 Feed-side partitioning

With \(\Delta\psi_{D,f}=\psi_m(0)-\psi_f\), ideal electrochemical equilibrium gives

\[
C_{m,0,i}
=C_{f,i}\,\Phi_i
\exp\!\left(-\frac{z_iF\Delta\psi_{D,f}}{RT}\right).
\]

This relation reuses the current steric and dielectric calculations. Only the Donnan potential is part of the coupled nonlinear solve.

### 5.2 Permeate-side partitioning

Using the same solution-to-membrane sign convention,

\[
C_{m,L,i}
=C_{p,i}\,\Phi_i
\exp\!\left(-\frac{z_iF\Delta\psi_{D,p}}{RT}\right).
\]

The factor is not inverted: the relation maps the permeate solution concentration into the adjacent membrane phase, just as the feed relation maps feed solution into membrane. If interface-specific material properties are introduced later, use \(\Phi_{f,i}\) and \(\Phi_{p,i}\); the MVP uses the same membrane properties on both sides.

### 5.3 Membrane electroneutrality at both interfaces

The membrane phase must satisfy fixed-charge electroneutrality at each endpoint:

\[
r_{\mathrm{EN},f}
=\sum_i z_iC_{m,0,i}+X=0,
\]

\[
r_{\mathrm{EN},p}
=\sum_i z_iC_{m,L,i}+X=0.
\]

The sign and units of \(X\) must exactly match the current electroneutrality helper. These two residuals determine the two Donnan potentials as part of the coupled system.

### 5.4 ENP transport inside the membrane

For each ion, the one-dimensional ENP flux is

\[
J_i^{\mathrm{ENP}}
=-K_{d,i}D_i\frac{dC_{m,i}}{dx}
-K_{d,i}D_i\frac{z_iF}{RT}C_{m,i}\frac{d\psi_m}{dx}
+K_{c,i}C_{m,i}J_v,
\]

where \(D_i\) is the ion diffusivity, \(K_{d,i}\) is diffusive hindrance, and \(K_{c,i}\) is convective hindrance.

For the algebraic two-interface MVP, retain the current constant-gradient approximation but apply it between the two membrane-side interface concentrations:

\[
\frac{dC_{m,i}}{dx}
\approx\frac{C_{m,L,i}-C_{m,0,i}}{L},
\qquad
\bar C_{m,i}=\frac{C_{m,0,i}+C_{m,L,i}}{2},
\qquad
\frac{d\psi_m}{dx}\approx\frac{\Delta\psi_m}{L}.
\]

Thus,

\[
J_i^{\mathrm{ENP}}
=-K_{d,i}D_i\frac{C_{m,L,i}-C_{m,0,i}}{L}
-K_{d,i}D_i\frac{z_iF}{RT}\bar C_{m,i}\frac{\Delta\psi_m}{L}
+K_{c,i}\bar C_{m,i}J_v.
\]

This is still an approximation to the profile \(C_{m,i}(x)\). It is useful because it introduces the missing exit interface without combining that structural change with a new spatial discretization.

### 5.5 Permeate flux relation

For a well-mixed permeate with negligible diffusive boundary-layer flux, steady solute balance gives

\[
r_{J,i}=J_i^{\mathrm{ENP}}-C_{p,i}J_v=0
\qquad\text{for every ion }i.
\]

Predicted rejection is then

\[
R_i=1-\frac{C_{p,i}}{C_{f,i}}.
\]

The solver must permit \(C_{p,i}>C_{f,i}\), and therefore \(R_i<0\); it must not clip rejection or permeate concentration.

### 5.6 Zero-current condition

For an electrically open membrane,

\[
r_I=\sum_i z_iJ_i^{\mathrm{ENP}}=0.
\]

This equation determines the membrane potential drop or electric-field degree of freedom. If physical current density is reported, multiply the molar-charge sum by \(F\); the zero is unchanged.

### 5.7 Optional bulk-permeate electroneutrality

The bulk permeate should satisfy

\[
\sum_i z_iC_{p,i}=0.
\]

For nonzero \(J_v\), this condition is algebraically implied by \(J_i=C_{p,i}J_v\) for every ion and \(\sum_i z_iJ_i=0\). It should therefore be a reported validation residual in the initial formulation, not an additional equation. Adding it without removing another dependent constraint would overdetermine the system. If a later formulation treats one permeate ion as dependent, permeate electroneutrality can instead be used to calculate that ion explicitly.

### 5.8 Square residual system

With the recommended \(N+3\) unknowns, return these \(N+3\) residuals:

1. one feed-side membrane electroneutrality residual;
2. one permeate-side membrane electroneutrality residual;
3. \(N\) ENP/permeate flux-matching residuals;
4. one zero-current residual.

The result object should additionally report bulk-permeate electroneutrality, minimum concentration, and unscaled residuals for auditing.

## 6. Numerical strategy

### 6.1 Recommended first solver: algebraic nonlinear system

Start with the square algebraic system and `scipy.optimize.root`. This is the safer next implementation step because it:

- adds the missing permeate interface while retaining the already tested constant-gradient transport approximation;
- reuses the current partitioning, hindrance, ENP, unit, and case-data functions;
- exposes a small, explicit residual vector whose equation count and individual balances can be unit tested;
- makes sign errors at the second interface easier to isolate;
- supports continuation from a controlled symmetric case and between neighboring water fluxes;
- separates failures caused by the new interface constraints from failures caused by spatial discretization.

The root solve should use transformed concentrations and scaled residuals:

- solve for \(u_{p,i}=\log(C_{p,i}/C_{\mathrm{ref}})\) to prevent negative trial concentrations;
- express potentials internally in volts but scale them by the thermal voltage \(RT/F\) when constructing initial guesses or diagnostics;
- scale electroneutrality residuals by \(C_{\mathrm{ref}}\);
- scale flux residuals by a nonzero reference flux such as \(\max(J_vC_{\mathrm{ref}},J_{\mathrm{floor}})\);
- scale the zero-current residual consistently with the flux residuals.

Success must require both the optimizer's success flag and explicit physical residual tolerances. A solver message alone is not sufficient. The returned result should include the initial guess, final unknown vector, number of function evaluations, scaled and unscaled residuals, interface concentrations, permeate concentrations, potentials, fluxes, and rejection.

Good initial guesses are:

1. solve or reuse the feed-side Donnan potential from the current interface helper;
2. initialize \(C_{p,i}\) from a controlled case, the existing baseline prediction, or a bounded fraction of \(C_{f,i}\), without clipping the final solution;
3. solve the permeate-side Donnan relation at that trial permeate composition;
4. initialize \(\Delta\psi_m\) from the current zero-current field estimate;
5. after the first successful water flux, use its solution for the neighboring flux (continuation).

### 6.2 Later solver: boundary-value formulation

`scipy.integrate.solve_bvp` is the appropriate later target when \(C_{m,i}(x)\) and \(\psi_m(x)\) must be resolved rather than approximated. A BVP would treat the ENP relations as first-order differential equations, keep each \(J_i\) constant across the membrane, and impose the two partition relations as boundary conditions. Unknown constant parameters can include the ion fluxes and a streaming-potential degree of freedom.

Starting directly with the BVP is riskier because interface algebra, differential state variables, unknown flux parameters, boundary conditions, gauge choice, and mesh convergence would all change at once. It is also easier to provide the wrong number of boundary conditions or to hide a sign/unit error behind mesh or Jacobian behavior. Move to `solve_bvp` only after the two-interface algebraic residuals pass conservation and symmetric-case tests. The BVP should then be checked against the algebraic solution in a regime where profiles are nearly linear.

## 7. Minimum viable implementation

The smallest useful implementation should:

- live in a new solver module or a clearly separated two-interface class so the existing baseline solver remains unchanged;
- use ideal activities;
- reuse the current steric, dielectric, Donnan, diffusive-hindrance, and convective-hindrance functions without tuning;
- retain the current hard steric cutoff, eliminate species that are constrained to exact zero from the active nonlinear vector, and fail clearly if the remaining coupled interface problem is infeasible;
- use the two-interface constant-gradient ENP equations above;
- solve one prescribed water flux at a time;
- support only the existing Foo 2023 LM-C pH 7 case initially;
- return predicted Li and Mg rejection, plus Cl and all residual diagnostics;
- never mutate the source case object; construct an immutable input state or copy the required values;
- leave concentration polarization and nonideal activity models out of the MVP.

A useful result type should contain at least:

- permeate concentrations and rejections by ion;
- feed- and permeate-side membrane concentrations;
- both Donnan potentials and the membrane potential drop;
- ENP ion fluxes and \(C_{p,i}J_v\) fluxes;
- raw and scaled residuals, including diagnostic permeate electroneutrality;
- convergence status, solver message, and iteration/function-evaluation count.

The MVP is successful when it solves the balances reproducibly. Agreement with experiment is a later validation question and must not be used to tune this first implementation.

## 8. Tests required before validation

### 8.1 Unit tests

1. **Feed partition sign convention:** a simple cation/anion case reproduces the existing feed-interface concentrations for the same Donnan potential and factors.
2. **Permeate partition direction:** the exit relation maps \(C_p\) into \(C_{m,L}\) using the same solution-to-membrane convention, and reverses exactly when solved for \(C_p\).
3. **Unknown/residual dimensions:** an \(N\)-ion state produces exactly \(N+3\) unknowns and \(N+3\) residuals.
4. **Positive concentration transform:** every finite log-concentration vector produces strictly positive permeate concentrations.
5. **Feed-interface electroneutrality:** the converged unscaled residual \(\sum_i z_iC_{m,0,i}+X\) is below a documented tolerance.
6. **Permeate-interface electroneutrality:** the converged unscaled residual \(\sum_i z_iC_{m,L,i}+X\) is below the same documented tolerance.
7. **Zero current:** \(\sum_i z_iJ_i\) is below a documented flux/current tolerance.
8. **Flux closure:** every \(J_i^{\mathrm{ENP}}-C_{p,i}J_v\) residual is below tolerance.
9. **Nonnegative physical state:** all feed, membrane-interface, and permeate concentrations are finite and nonnegative; solved permeate concentrations are strictly positive under the log transform.
10. **No rejection clipping:** a constructed state with \(C_{p,i}>C_{f,i}\) returns a negative rejection.
11. **Hard-cutoff handling:** an exactly excluded species is reported with zero membrane concentration, flux, and permeate concentration without entering a logarithm; a separate charge-balance-infeasible case returns a clear typed failure rather than NaNs or false convergence.

### 8.2 Controlled integration tests

1. **Symmetric electrolyte:** solve a controlled symmetric monovalent salt with equal transport properties, zero fixed charge, and symmetric partition factors. The solution should have equal cation/anion concentrations and fluxes, electroneutral endpoints, and a zero or symmetry-consistent electric field.
2. **Continuation consistency:** solving a nearby water flux from a converged state should reach the same solution, within tolerance, as solving it from the standard initial guess.
3. **Algebraic conservation:** independently recompute interface electroneutrality, flux closure, zero current, and permeate electroneutrality from the returned physical state rather than trusting stored residuals.

### 8.3 Foo 2023 smoke and workflow tests

1. **One successful LM-C pH 7 flux solve:** require finite concentrations, finite Li/Mg rejection, optimizer success, and all physical residuals below tolerance. This is a convergence test, not an agreement-with-experiment test.
2. **Four-flux workflow:** the existing four Foo 2023 LM-C pH 7 water fluxes return four ordered Li predictions and four ordered Mg predictions, with one result object per flux.
3. **Input immutability:** the Foo case data and membrane parameters are unchanged after single- and multi-flux solves.
4. **Determinism:** repeated runs from the same inputs and initial guess return the same solution within numerical tolerance.

No acceptance test should require lower RMSE until the conservation tests pass and the new model has been independently validated.

## 9. Risks and expected failure modes

### Nonconvergence

The coupled exponentials, very small partition factors, and disparate concentration scales can produce a poorly conditioned Jacobian. Use dimensionless residual scaling, log concentrations, continuation in \(J_v\), and diagnostic reporting of the largest raw residual. Do not accept an optimizer success flag with failed physical tolerances.

### Negative or nonfinite concentrations

An unconstrained root solve can try negative concentrations, and Donnan exponentials can overflow. Use log permeate concentrations, evaluate exponentials with finite-range checks, and return an explicit failure for nonfinite interface states.

### Overdetermined or rank-deficient system

Adding permeate electroneutrality on top of every flux-closure equation and zero current creates a redundant constraint for nonzero \(J_v\). Conversely, adding explicit \(J_i\) unknowns without both corresponding equation sets creates an underdetermined system. Centralize unknown packing and residual construction, assert their dimensions, and document dependent constraints.

### Bad initial guesses and multiple roots

Large Donnan potentials or an inappropriate permeate composition can lead to a different mathematical root or no root. Begin with the controlled symmetric case, use separately solved interface potentials for initialization, continue through water flux in small steps, and record the initial guess in results.

### Parameter identifiability

Ion radii, pore radius, fixed charge, dielectric penalty, and hindrance all affect exclusion. A more complete solver does not make these parameters identifiable from four rejection points. Keep baseline parameters fixed while evaluating the structural solver change; parameter inference requires a separate design and additional observables.

### Hard steric cutoff causing zero concentrations

At the baseline pore radius, the current steric model gives Mg a zero entry factor. A zero \(\Phi_i\), together with zero cutoff hindrance, forces both membrane-side concentrations, Mg flux, and Mg permeate concentration to zero regardless of either Donnan potential. Eliminate this exact constrained state from the active nonlinear vector and report its zeros explicitly. The reduced ion set may then be unable to satisfy membrane electroneutrality; that is a genuine infeasibility and must be reported. The MVP must not silently add a floor or alter the radius. A subsequent, separately reviewed model change can replace the cutoff if the two-interface diagnostic confirms the need.

### Unit and sign errors

Donnan voltage, membrane voltage, electric field, and concentration units are easy to mix. Define both Donnan jumps solution-to-membrane, use volts internally, keep a single concentration unit within residual evaluation, and test the exit relation independently. Report all three potential differences so the total potential identity can be checked.

### False confidence from the algebraic profile

The algebraic MVP still assumes constant gradients and mean concentrations. A converged root proves balance consistency for that approximation, not that the full membrane profiles are accurate. Compare it with a later BVP before treating the model as a full spatial ENP solution.

## 10. Recommended implementation sequence

1. **Define immutable inputs and results.** Add a separate two-interface input structure containing species, feed concentrations, membrane parameters, temperature, thickness, and one \(J_v\). Define a result structure with physical state, convergence metadata, and raw residuals. Do not change the existing solver API.
2. **Implement and test a pure two-interface state assembler.** Classify exact hard-cutoff species, then, given the reduced \(N+3\) unknown vector, unpack active log permeate concentrations and potentials, compute both sets of interface concentrations using the stated sign conventions, and compute the constant-gradient ENP fluxes. Reinsert exact zeros for inactive species in the returned physical state. This function should have no optimizer and no mutation. This is the recommended first coding step.
3. **Implement scaled residual construction.** Return the two interface electroneutrality residuals, \(N\) flux-closure residuals, and zero-current residual. Assert the \(N+3\) dimension and separately calculate diagnostic permeate electroneutrality.
4. **Add controlled unit tests.** Test partition direction, positivity, equation count, residual scaling, negative rejection, exact-zero hard-cutoff elimination, and explicit infeasibility reporting before calling a nonlinear solver.
5. **Wrap `scipy.optimize.root`.** Add initial-guess construction, physical convergence checks, typed failure reporting, and complete result packaging. Never treat the optimizer flag alone as success.
6. **Solve a controlled symmetric electrolyte.** Establish a known, reproducible solution and independently audit both electroneutrality balances, flux closure, zero current, and permeate electroneutrality.
7. **Add one Foo 2023 LM-C pH 7 smoke solve.** Use unchanged baseline factors and parameters. If the hard Mg cutoff makes the system infeasible, report that result explicitly rather than softening it inside the solver.
8. **Add continuation across the four Foo water fluxes.** Solve one flux at a time, seed each neighboring flux from the previous solution, preserve input ordering, and return four Li/Mg predictions with per-flux diagnostics.
9. **Compare structure, not fitted performance.** Compare the new solver's residuals and predictions with the existing baseline without tuning parameters or claiming improvement. Decide separately whether the hard entry model must change.
10. **Implement the spatial BVP only after the algebraic system is stable.** Promote \(C_{m,i}(x)\) and \(\psi_m(x)\) to differential states, treat constant ion fluxes as BVP parameters, impose both partition boundaries, and verify the BVP against the algebraic solver in a near-linear regime.

This sequence makes the second interface, conservation equations, and numerical representation independently testable. It also ensures that a failure caused by the current hard steric cutoff is visible rather than hidden by an unreviewed parameter or physics change.
