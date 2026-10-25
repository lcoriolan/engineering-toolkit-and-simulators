# Benchmark methodology and calculation sources

A benchmark is a declared problem, expected result and error criterion. An imported module, successful dispatch, attractive plot or self-comparison is not independent physical validation. The [calculator results](../benchmarks/README.md) contain every exercised calculator family's equations or numerical procedure, variable descriptions and SI units, assumptions, inputs, expected/actual values, tolerances, errors and original source-confidence notes. [Solver results](../benchmarks/solvers.json) provide the corresponding record for synthetic numerical problems.

| Evidence class | What it establishes | What it does not establish |
|---|---|---|
| Published reference landmark | Agreement with a specified public reference value/model | General empirical accuracy, a complete standard's compliance or validation beyond the sampled condition |
| Analytical / manufactured solution | Governing equation, boundary condition and numerical error for an idealized problem | Accuracy of a different constitutive law or a real design |
| Conservation / scaling invariant | Internal physical/mathematical consistency under the stated assumptions | Independence from the model or correctness of omitted physics |
| Regression / parity | Behavior agrees with a former kernel or the same backend | Independent physical validation |
| Integration / geometry check | Routing, failure handling, dependencies, editable rebuild or finite geometric agreement | Physical qualification or universal geometric recognition |

Most of the 1,232 expected calculator values were independently specified by hand substitution. Some cited landmarks were originally recalled rather than re-read; that confidence remains visible. The nitrogen CoolProp enthalpy and Cantera flame-temperature comparisons are backend regressions. They are not reclassified as independent measurements. The default oracle criterion is relative error ≤0.002; exact-zero references use absolute 1e-12 unless a scalar case explicitly supplies another tolerance. Structured results apply that zero rule to each numeric leaf. Wrong-reference negative controls are included.

## Primary sources checked for this release

The following publisher/author/maintainer pages were opened on 2026-10-24. “Method basis” means the method/reference was verified, not that every stored worked example was transcribed from that page. Full paywalled standards were not read. Textbook recollections in individual calculator metadata are not silently upgraded by a general subject reference.

| Family or method | Primary source and verification scope |
|---|---|
| Standard atmosphere | [NASA US Standard Atmosphere 1976 record](https://ntrs.nasa.gov/citations/19770009539): report identity, model/table coverage and public distribution verified. The large PDF was not fetched by the browser; stored table landmarks retain their original provenance. |
| Compressible flow | [NASA isentropic relations](https://www.grc.nasa.gov/www/k-12/airplane/isentrop.html), [normal shock](https://www.grc.nasa.gov/www/k-12/airplane/normal.html), [Prandtl–Meyer function](https://www.grc.nasa.gov/www/k-12/airplane/pranmyer.html): governing relations and ideal-gas assumptions checked. |
| Thermochemistry and reaction energy | NIST WebBook [water](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1) and [methane](https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=1): species, gas-phase data and source tables checked. Standard-state, phase and temperature must match; full property data remains in its original references. |
| Water/steam and fluid properties | [IAPWS IF97 release](https://iapws.org/technical-guidance/release/IF97-Rev), [CoolProp nitrogen EOS references](https://coolprop.org/fluid_properties/fluids/Nitrogen.html): formulation and underlying EOS references verified. Comparing two calls to CoolProp remains regression. |
| Statistical process control | [NIST Shewhart charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc321.htm): equations and n=5 factors checked. NIST prints D4=2.115; the retained rounded-input landmark uses 2.114, a difference within its declared 0.2% tolerance. |
| Structural mechanics and fit | [MIT Mechanics & Materials](https://ocw.mit.edu/courses/2-001-mechanics-materials-i-fall-2006/): force, deformation and geometric-fit framework verified. The repository's beam/section/bolt oracles are analytical worked cases, not copied MIT experiments. |
| Rigid-body dynamics and vibration | [MIT Engineering Dynamics](https://ocw.mit.edu/courses/2-003sc-engineering-dynamics-fall-2011/): Newton/Euler, reference frames, angular momentum and vibration teaching sources verified. Freefall and orbital-energy checks use explicit analytic invariants. |
| Involute geometry | [MIT machine elements course material](https://ocw.mit.edu/courses/2-017j-design-of-electromechanical-robotic-systems-fall-2009/16cb0f850752422026a85d838f30e340_MIT2_017JF09_machines.pdf): gear geometry/compatibility method basis. The 13-position test checks synthetic Boolean overlap, not tooth stress or measured efficiency. |
| Airfoil section geometry | [PDAS NACA four-digit thickness](https://www.pdas.com/naca456thick4.html): thickness function and finite trailing-edge convention checked. This implementation explicitly uses the closed-edge coefficient -0.1036 instead of the original -0.1015 and allows caller trailing-edge thickness. |
| Atmospheric acoustic absorption | [ISO 9613-1 publisher abstract](https://www.iso.org/standard/17426.html): method scope and meteorological inputs verified. Full standard tables/compliance were not verified. |
| Poisson/FEM | [NGSolve official Poisson tutorial](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-1.1-poisson/poisson.html): weak-form construction checked. Our manufactured `sin(pi*x)sin(pi*y)` unit-square problem is independently specified, not the tutorial's example. |
| Quantum chemistry | [PySCF SCF documentation](https://pyscf.org/user/scf.html): HF/KS model and energy formulation verified. Tests identify the method, basis and molecular input; finite-basis calculations are not exact molecular energies. |
| Warp runtime | [NVIDIA Warp](https://github.com/NVIDIA/warp): backend/build framework. CPU/GPU parity is a backend regression class. |
| Native CAD export | [FreeCAD Part Export](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Part_Export.md): upstream export interface. Two-way Boolean and edit/rebuild tests are the repository's own synthetic checks. |

## Representative governing equations

The world uses `m dv/dt = F` with declared integration and float precision. Orbital energy is `v²/2 − μ/r`. Finite-depth waves satisfy `ω² = g k tanh(kh)`. Sabine decay is `T60 = 24 ln(10) V / (c ΣαS)`. Linear elasticity uses `σ=2με+λtr(ε)I`; magnetic operators use scalar `−div(ν grad A)=J` or vector curl–curl form. Thermal channel screens use `Δp=f(L/Dh)ρv²/2` and `ΔT=Q/(ṁcp)`. VLM solves the no-penetration influence system and applies Kutta–Joukowski panel forces. Tolerance, gear and mechanism equations are described in [checker methods](checkers.md).

The complete expressions, numerical procedure code, assumptions and variable units for the calculator families are in the per-family JSON files. Other module functions are catalogued in the API inventory with source docstrings. Unexercised functions remain unqualified; neither an all-source inventory nor passing regression tests removes that limitation.
