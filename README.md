# Engineering Toolkit and Simulators

Executable engineering calculations, scientific workers, a shared simulation world, controlled browser flow solvers, and parametric CAD tools. Python package version **0.1.0**.

The repository contains real solver and checker implementations with synthetic examples. It includes the 50-family mechanical calculator library and its complete 1,232-case reference dataset, scientific operators, land/water/space/acoustic domains, electromagnetic and finite-element operators, exact CAD import/rebuild, geometry features, and tolerance/fit/motion checks.

The release passed **1,348 Python tests and 27 native CAD checks**, with zero test skips. A synthetic 200-step CPU/GPU run matched exactly; [validation details and limits](docs/validation.md) identify the tested backends and scope.

## Run

```sh
git clone https://github.com/lcoriolan/engineering-toolkit-and-simulators.git
cd engineering-toolkit-and-simulators
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
engineering-sim demo --device cpu
engineering-sim serve --port 8765
```

Open [the local flow demo](http://localhost:8765). Choose D2Q9 or D3Q19, initialize, then advance. Python grants each work chunk and receives the numerical result; pause and stop control actual computation.

For chemistry, thermodynamics and the full calculator reference suite:

```sh
python -m pip install -e '.[science,advanced-science,fea,space,plots,test]'
engineering-sim run science --input examples/water_molecule.json
python -m pytest -q
python tools/benchmark_report.py --output benchmarks
python tools/benchmark_solvers.py --fea
```

Optional dependencies have platform constraints. Read [installation and tested backends](docs/dependencies.md), particularly the NGSolve CPU-build and older NVIDIA GPU notes. FreeCAD uses its own compatible Python interpreter.

## CAD and parametric checking

With this package importable by a FreeCAD-enabled Python interpreter:

```sh
python -m engineering_sim cad-build examples/parametric_solid.json scratch-solid
python -m engineering_sim cad-import scratch-solid/model.step scratch-import --name synthetic_model
python tests/cad_checks.py
python tests/checker_cad_checks.py
```

`cad-build` creates a native parametric document and STEP from explicit JSON operations. `cad-import` produces Python and native reconstruction for supported primitives/extrusions and preserves exact BREP when recognition is insufficient. The generated package is checked by volume, area, bounds, and both Boolean differences. See [CAD contracts](docs/cad-python-contracts.md).

The [checker APIs](docs/checkers.md) include worst-case/RSS tolerances, thermal clearance, frame alignment, joint/loop closure, SVD mobility, exact solid separation/interference, sampled assembly paths, involute gears, flexure linkages, and mold withdrawal/tool-access screens. All supplied dimensions are synthetic or explicit caller inputs.

## Calculations, benchmarks and coverage

| Evidence | Contents |
|---|---|
| [Calculator benchmark results](benchmarks/README.md) | All 50 families, each equation/procedure, variables and units, assumptions, inputs, expected/actual results, error and tolerance |
| [Solver benchmark results](benchmarks/solvers.json) | Analytical, manufactured-solution, conservation and scaling checks with actual numerical errors |
| [Methodology and primary references](docs/methodology.md) | What was checked against external sources, what is analytical, and what is regression/integration only |
| [Coverage matrix](docs/coverage.md) | Public implementations, dependencies, examples/tests, actual coverage and precise exclusion categories |
| [API inventory](evidence/api-inventory.json) | Every packaged Python module and its definitions/imports; test references are not line-coverage claims |
| [Validation record](docs/validation.md) | Executed suites, CAD/browser results and limitations |
| [Architecture](docs/architecture.md) | Execution dispatch, world/domain adapters, independent science processes, browser workers and CAD |

The expected calculator values are maintained separately from the solvers. Most are analytical worked references. Same-backend comparisons and recalled textbook landmarks are labelled. Passing tests do not establish empirical accuracy for arbitrary designs, manufacturing readiness, or universal physical validation.

The implementation inventory is broader than the tested surface. Importable functions without applicable numerical tests remain unqualified. See the coverage matrix for unsupported features, external dependencies and unexercised operating regimes.

## Roadmap and voting

Help choose future work: [read the proposed roadmap](docs/roadmap.md) and [vote on feature requests](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues?q=is%3Aissue%20is%3Aopen%20label%3Aroadmap). The proposals cover a versioned design record, dependency-aware workflows, manufacturing/supplier/logistics adapters, multidisciplinary optimization, traceable reports and a generic gearbox demonstrator.

Add a **👍 reaction to each proposal's opening issue post** to support it; voting for multiple options is welcome. Comment with your use case or [suggest an alternative](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/new?template=feature_request.yml). These integrated features are proposed, not currently implemented or guaranteed commitments.

## Licensing

**Free for personal noncommercial hobby use. Commercial use outside the PolyForm grant requires a separately negotiated paid license.** Repository-owned material is licensed under [PolyForm Noncommercial 1.0.0](LICENSE); keep the [required notice](NOTICE) when sharing it.

PolyForm also permits other noncommercial purposes and use by its listed charitable, educational, public research, public safety/health, environmental and government organizations, regardless of funding source or obligations resulting from funding. This is broader than hobby-only permission. The full license governs.

This public repository is **source-available**, not OSI open-source. See [commercial licensing](COMMERCIAL_LICENSE.md), [license details](LICENSE_STATUS.md) and [third-party exceptions](THIRD_PARTY_NOTICES.md). Upstream libraries and reference data retain their own terms; the project cannot sell or restrict rights it does not control.

`PUBLIC_FILES.json` records the package files, sizes and SHA-256 hashes. Run `python tools/check_release.py` on a clean checkout to verify them.
