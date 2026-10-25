# Installation and tested dependencies

Python 3.11+ is declared; the complete fresh-environment run used Linux x86-64 Python 3.12.3. The exact observed packages are in [backend versions](../benchmarks/backend-versions.json). Base installation includes NumPy, SciPy, Pint, SymPy and NVIDIA Warp. Extras in `pyproject.toml` separate chemistry, advanced molecular/materials tools, FEM, orbital data models, plotting and tests.

The runtime was tested in an isolated virtual environment without system site-packages. Scientific workers use the current interpreter by default, or `ENGINEERING_SIM_PYTHON`. Data goes beneath the current directory's `.engineering-sim`, overridable by documented environment variables.

FreeCAD/Part are native application libraries, not supplied by pip. The CAD checks used FreeCAD 1.1.3 and its compatible Python environment. Run with this repository's `src` importable or install the package into that interpreter.

NGSolve 6.2.2608's downloaded wheel raised an illegal-instruction fault on the tested older CPU. FEM checks instead used independently installed upstream-source builds of NGSolve/Netgen 6.2.2607 with native-architecture optimization disabled, `-march=ivybridge`, OpenBLAS, no MKL and no CUDA. A compatible wheel or [official source build](https://docu.ngsolve.org/latest/install/install_sources.html) is necessary on older CPUs; the ordinary pip wheel is not claimed portable to them.

Warp 1.18.0 worked on CPU. Its CUDA 13.4 build rejected the tested sm_61 Pascal GPU at kernel launch, correctly requiring sm_75 or newer. The [upstream Warp documentation](https://github.com/NVIDIA/warp) explains CUDA build/driver requirements. A separate fresh environment with `warp-lang==1.9.1` (CUDA 12.8) successfully executed the same 200-step synthetic freefall on CPU and a Tesla P4, with zero component difference. For that older GPU use `python -m pip install "warp-lang==1.9.1"`; modern-GPU compatibility was not separately tested. See [the exact record](../evidence/gpu-validation.json). CPU is the default.

SU2 must be installed separately and selected with `SU2_CFD`. Its adapter is included, but this release did not run an external SU2 numerical solve. Materials phase calculations require separately licensed thermodynamic database data via `ENGINEERING_SIM_TDB_DIR`, or a separately installed pycalphad example database. TDB data is not bundled. Public network services, external molecule/structure files, large models and optional binary tools are additional dependencies for the operations that use them.

Minimal controls test: `python -m pytest tests/test_execution.py tests/test_browser_workers.py tests/test_http.py`. Full scientific tests require all relevant extras. Tests do not silently skip unavailable packages: install the backend or run the explicitly narrower suite.

Optional runtime inputs are explicit: `ENGINEERING_SIM_DATA` selects the local science data directory; `ENGINEERING_SIM_INVENTORY` selects a caller-owned inventory JSON file (no ancestor directories are searched); `ENGINEERING_SIM_RETRO_DATA` selects an AiZynthFinder data/configuration directory; and `ENGINEERING_SIM_RETRO_PYTHON` selects its interpreter. The latter defaults to the current interpreter and data beneath the selected science directory. Supply the inventory records and AiZynthFinder models needed by those operations.
