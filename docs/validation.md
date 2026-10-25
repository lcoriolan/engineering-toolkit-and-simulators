# Validation evidence and claim limits

Validation date: 2026-10-24. This record covers the executable package. [Machine-readable summary](../evidence/validation-summary.json), [backend versions](../benchmarks/backend-versions.json), [methodology](methodology.md) and [coverage matrix](coverage.md) distinguish analytical verification, reference comparisons, regression and integration.

## Executed public release checks

| Scope | Observed result |
|---|---|
| Full public pytest suite, Linux x86-64 / Python 3.12.3 | **1,348 passed; zero failures and zero skips**, 177.04 seconds |
| Calculator references and registry checks, included above | 1,232 reference cases for all 827 calculators across 50 families, plus 3 registry/example/negative controls |
| Other pytest checks, included above | 19 execution/browser/HTTP; 12 core numerical; 54 scientific workers; 3 FEM; 8 parametric; 12 extended algorithms; 5 configuration and portable-export checks |
| Native FreeCAD 1.1.3, separate from pytest | **27 passed**: 15 import/build/export, 8 fit/contact/motion checks and 4 feature/hydrostatic checks |
| Standalone solver benchmark report | **20 passed**, with expected/actual results and errors in [solvers.json](../benchmarks/solvers.json); methods overlap the test suite |
| Actual public WebGL UI | **6 passed**: finite initialization/advance for 2D and 3D, pause holding the step count, and stopped worker rejecting advance |
| CPU/GPU execution | **200-step synthetic freefall passed** on CPU and Tesla P4 with Warp 1.9.1 / CUDA 12.8; maximum state-component difference 0 |
| Built wheel outside the source checkout | Installation and `pip check` passed; CPU world demo and water-molecule science request passed |
| Installed-module inspection | 241 imports passed, 4 modules require the separate FreeCAD interpreter, 1 bridge is flat-export-only; zero unexpected failures across 246 enumerated submodules |

The 13 synthetic gear poses are samples inside one of the eight native checker checks, not 13 additional tests. The 247 source-module count includes the package root; installed-module enumeration excludes that root. The import report is dependency evidence, not numerical validation. Counts across tables must not be added as independent physical validations.

The full suite emitted two SciPy runtime warnings (division by zero and invalid division in a frequency response at a pole) in the stability-margin reference case; that case passed its declared criterion. No optional-dependency tests were silently skipped.

See the [browser record](../evidence/public-browser-validation.json), [GPU record](../evidence/gpu-validation.json) and [installed-module record](../evidence/installed-module-check.json). The browser checks establish routing, control and finite state, not force accuracy or mesh convergence. GPU equality is a backend parity check, not independent physical validation.

## Reproduce

Install the dependencies described in [installation](dependencies.md), then run:

```sh
python -m pytest -q
python tools/benchmark_report.py --output benchmarks
python tools/benchmark_solvers.py --fea
python tools/check_installation.py
python tools/check_release.py
```

Run the separate native checks with this package available to a FreeCAD-compatible interpreter:

```sh
python tests/cad_checks.py
python tests/checker_cad_checks.py
python tests/feature_cad_checks.py
```

`engineering-sim serve --port 8765` provides the public WebGL demonstration. Initialize and advance each solver; pause must retain its step count and stop must reject further work. For the tested older Pascal GPU, install `warp-lang==1.9.1` in a separate environment and call `engineering_sim.demo.run_world(duration_s=0.2, dt_s=0.001, device="cpu")` and the same function with `device="cuda:0"`. Exact output and backend constraints are recorded in the GPU evidence and dependency notes.

## Remaining limits

- The external SU2 adapter is packaged, but an external SU2 numerical solve was not run. Complete aircraft trim, all land/contact scenarios, full mold-routing optimization and advanced molecular/materials workflows remain unqualified beyond their stated checks.
- The latest Warp CUDA 13 build rejected the tested Pascal GPU; the documented CUDA 12 environment passed. NGSolve required a compatible upstream source build on the older CPU. See the exact dependency notes rather than assuming all binary wheels work everywhere.
- Native geometry recognition is bounded. Unsupported geometry retains exact BREP; recovered editable feature history is not claimed. Discrete assembly/gear samples do not prove continuous-path clearance.
- Photo-to-CAD reconstruction is not implemented. Dimension parsing and specification validation are available.
- No experimental, hardware, full production scenario or manufacturing release qualification is claimed.

## Configuration and portable exports

Five checks cover caller-selected inventory and local stock, explicit retrosynthesis paths, and offline chemical-run export. Inventory is read from the configured location rather than discovered in ancestor directories. A synthetic export is replayed in an isolated subprocess and checked for deterministic archive bytes. These checks are included in the 1,348-test result above.
