# Parametric checking APIs

| API | Inputs and method | Evidence and limits |
|---|---|---|
| `checks.tolerances.fit_clearance`, `stack`, `thermal_shift` | Explicit hole/shaft intervals; `WC=sum(abs(s*t))`; `RSS=sqrt(sum((s*t)^2))`; differential `alpha*L*deltaT` | Analytical checks; RSS requires an independence/statistical-basis assumption |
| `checks.transforms`, `checks.mechanisms` | Rigid transforms, SE(3) joint/loop residuals, constraint-Jacobian SVD rank | Hinge has one free DOF and zero when locked; a non-rigid matrix is rejected as valid closure evidence |
| `checks.mechanisms.pair_check` and contact helpers | FreeCAD minimum distance and Boolean overlap, scoped contact regions, cylindrical bore datums | Synthetic passing clearance, failing clearance and positive interference; not loaded contact mechanics |
| `checks.mounted_fit` | Declared interface frames, coaxial shafts/bores, bolt patterns, feet, key clocking and clashes | Frame transform and cylindrical datum exercised; complete assembly qualification requires design-specific checks |
| `checks.assembly` | Discrete translations at 1 mm intervals; tool envelopes and radial-separation prefilter | Obstructed and clear paths exercised; thin obstacles between samples can be missed; no continuous or rotational insertion proof |
| `cad.gears` | Standard 20° zero-shift external/internal involute geometry, explicit tooth count/module/backlash | Pitch/backlash formulas and 13 poses of a synthetic 24:36 pair; no loaded tooth contact, wear or continuous collision guarantee |
| `checks.flexures` | Constant-curvature bend and fixed-length crank/pushrod closure | Analytical strain and rod closure checks; imposed kinematics omit elastic equilibrium, fatigue and friction |
| `checks.rules`, `checks.joints` | Explicit wall/cutter/fit/load rules; complete local-joint sensitivity equations | Invalid wall/material paths exercised; joint material/fastener inputs are caller-supplied; no physical joint qualification |
| `checks.seals`, `checks.molding` | Gland fill/squeeze intervals, explicit cross-section uncertainty, shrink/tolerance stack | Analytical dimensional checks; material and process assumptions are caller inputs |
| `checks.mold_geometry`, `mold_routing`, `mold_routing_study` | Exact insert construction/overlap checks, discrete withdrawal, sampled cutter/draft checks, cavity graph and shortest-cost gate paths | Reusable construction/validation implementation; full route-optimization/mold qualification unexercised in this release |
| `geometry.*`, `vision.dimensions`, `vision.spec` | Mesh mass properties, sectional measurement, NACA/caller DAT coordinates, dimension-text parse and explicit feature-spec validation | Synthetic mesh/section/parser tests; image-to-CAD reconstruction is not implemented |

Run `python -m pytest tests/test_parametric_checks.py tests/test_extended_algorithms.py` for numeric checks, and the standalone CAD scripts under FreeCAD Python. Every configured tolerance belongs to a specific algorithm and unit; a geometric pass is not a manufacturing or service-life certificate.
