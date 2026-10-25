# Coverage matrix

This matrix lists packaged capabilities, required dependencies, executed checks and technical limits. The package contains **247 Python modules** and **1,778 top-level function/class definitions**, including helpers and classes. These counts are not independent solver or validation counts.

| Family | Public implementation | Dependencies | Example / executed evidence | Limits |
|---|---|---|---|---|
| 50 calculator families | science/mech; science/data/oracles | NumPy/SciPy/Pint/SymPy; CoolProp for property cases | 1,232 reference cases, registry coverage, dimensional/examples check and wrong-reference negative control | All registered calculator IDs covered; not all input branches or physical regimes |
| Chemistry, thermo, biochemistry, chemical engineering, quantum/MD, materials | science | science / advanced-science extras | water_molecule.json; 54 thermo/biochem/chemeng/chemphys tests | Advanced docking, protein preparation, retrospective synthesis and arbitrary materials databases not fully exercised |
| Execution and HTTP/browser control | execution; web_server; web | stdlib; WebGL2 browser | 19 execution/browser/HTTP tests; actual public 2D and 3D flow UI | Local demonstration server; no production multi-user service |
| World and environment | core | Warp, NumPy/SciPy | Synthetic CPU freefall, US76 states; GPU compatibility record | All domain combinations, turbulence grids and contact regimes not qualified |
| Land, water, space | domains/land; water; space | Warp; space extras | Displacement, density, wave/host parity, two-body energy and orbital backend parity | Land scenarios and optional space-weather/SGP4 data not comprehensively benchmarked |
| Acoustics, room, camera/multirotor, motion control, rocket, fin aero | domains | Base; optional plotting/space dependencies | Sabine law and gimbal settling | Outdoor propagation, camera/multirotor and rocket trajectories not fully numerically qualified in this public suite |
| Electromagnetics and FEM | electromagnetics; fea | SciPy; NGSolve/Netgen | Field linearity, reluctance coenergy/drive balance, bearing force symmetry; 3 FEM tests | Design-specific meshes, boundary conditions and convergence require separate validation |
| Flow and flight dynamics | flow; flight_dynamics | SciPy; external SU2 | VLM scaling and zero lift; eigenmode analytical metrics | SU2 solve and complete aircraft stability/trim unexercised; attached-flow assumptions |
| Thermal, structures, cooling, grilles, mold process | thermal; structures; cooling; grilles; molding_process | Base | Section/beam formulas, heat balance, slab scaling, channel and line-of-sight checks | Full thermal/mechanical/molding qualification remains unexercised |
| Native CAD, editable import and 20 feature operations | cad | FreeCAD/Part | 15 import/build/export and 4 feature/hydrostatic checks; synthetic inputs | Exact retained BREP is not recovered feature history; not all feature combinations tested |
| Tolerances, fit, motion, contacts and manufacturing screens | checks; cad/gears | Base; FreeCAD for solid checks | 8 parametric tests; 8 native checker checks incl. 13 gear poses; 12 extended algorithm tests | Discrete samples are not continuous-motion proof; full joint/mold routing studies unexercised |
| Generic geometry and dimension/spec parsing | geometry; vision | NumPy; caller CAD data | NACA, mesh volume, circle fit, mass inertia, dimension parse | Photo-to-CAD reconstruction is not implemented |

## Interpreting coverage

Dimension parsing and feature-spec validation are implemented; photo-to-CAD reconstruction is not. Operations using external solvers, models or thermodynamic databases require those dependencies and their applicable permissions. See [installation](dependencies.md).

See [API inventory](../evidence/api-inventory.json) for every packaged definition and import. Test-text references in that file are navigation aids, not measured function/line coverage. A packaged function without a directly applicable numerical test remains unqualified. No claim of “all physics validated” is made.
