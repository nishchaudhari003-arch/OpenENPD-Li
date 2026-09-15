# Release checklist (v0.1.0)

A checklist for preparing the repository as a clean, reproducible portfolio
project. This is a diagnostic toolkit; the checklist deliberately avoids any
claim of validated physics or manuscript readiness.

## Quality gates

- [x] Test suite passes locally (`PYTHONPATH=. pytest`).
- [x] GitHub Actions CI runs the suite on push and pull request.
- [x] README complete (description, status, install, quickstart, structure,
      reproducibility, limitations, citation).
- [x] Core docs complete: `docs/model_assumptions.md`,
      `docs/validation_summary.md`, `docs/limitations.md`.
- [x] Runnable example: `examples/foo2023_lmc_ph7_workflow.py`.
- [x] Reproducible artifact script: `scripts/generate_release_artifacts.py`.
- [x] Scientific claims kept conservative (diagnostic, not validation success).
- [x] No new physics, parameters, validation cases, or fabricated data added in
      the release-polish milestone.

## Reproducing artifacts

```bash
PYTHONPATH=. pytest
PYTHONPATH=. python examples/foo2023_lmc_ph7_workflow.py
PYTHONPATH=. python scripts/generate_release_artifacts.py   # writes to outputs/
```

Generated `outputs/` are gitignored; regenerate them rather than relying on
committed copies. A small set of baseline artifacts is committed under
`results/`.

## License decision needed

There is **no `LICENSE` file** in the repository yet, and no license is
otherwise declared. The owner should choose and add one before public release
(for an open-source portfolio project, common permissive choices include MIT or
BSD-3-Clause; MIT is a common default). No license was added automatically
because none was previously specified.

## Suggested GitHub repository description

> Tested ENP-Donnan modeling and diagnostics for membrane-based lithium/brine separations.

## Suggested GitHub topics

```
python
chemical-engineering
membrane-transport
lithium-extraction
nernst-planck
donnan
nanofiltration
scientific-computing
```

## Before tagging a release

- [ ] Choose and add a `LICENSE` (see above).
- [ ] Confirm the CI badge is green on the default branch.
- [ ] Optionally add `CITATION.cff` metadata (author, ORCID, repository URL).
- [ ] Re-read README and docs for any accidental overclaim.
