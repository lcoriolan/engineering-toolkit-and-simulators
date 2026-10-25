# Calculator benchmark results

Expected values are maintained separately from the calculation implementation. Each linked JSON contains every governing equation or numerical procedure, variable meaning and SI unit, assumptions, synthetic inputs, reference result, actual result, absolute and relative error, tolerance, and source confidence. Backend versions are in [backend-versions.json](backend-versions.json).

**1232/1232 cases passed; 827 calculators in 50 families.**

Most references are hand substitutions into governing equations. Recalled textbook landmarks remain labelled; same-backend fluid/combustion values are regression checks. These results are not 1,232 independently measured physical validations.

Reproduce with `python tools/benchmark_report.py --output benchmarks`. See [methodology](../docs/methodology.md) for primary sources and interpretation.

| Family | Calculators | Cases | Passed | Failed |
|---|---:|---:|---:|---:|
| [acoustics](calculators/acoustics.json) | 13 | 21 | 21 | 0 |
| [aerospace](calculators/aerospace.json) | 14 | 20 | 20 | 0 |
| [automotive](calculators/automotive.json) | 17 | 26 | 26 | 0 |
| [beams](calculators/beams.json) | 11 | 17 | 17 | 0 |
| [bearings](calculators/bearings.json) | 12 | 18 | 18 | 0 |
| [belts_chains](calculators/belts_chains.json) | 14 | 21 | 21 | 0 |
| [bolted](calculators/bolted.json) | 13 | 22 | 22 | 0 |
| [clutches_brakes](calculators/clutches_brakes.json) | 16 | 25 | 25 | 0 |
| [columns](calculators/columns.json) | 9 | 12 | 12 | 0 |
| [combustion](calculators/combustion.json) | 8 | 19 | 19 | 0 |
| [compressible](calculators/compressible.json) | 15 | 17 | 17 | 0 |
| [controls](calculators/controls.json) | 10 | 30 | 30 | 0 |
| [cycles](calculators/cycles.json) | 21 | 28 | 28 | 0 |
| [design_fund](calculators/design_fund.json) | 8 | 11 | 11 | 0 |
| [economics](calculators/economics.json) | 17 | 35 | 35 | 0 |
| [electrical](calculators/electrical.json) | 13 | 20 | 20 | 0 |
| [ergonomics](calculators/ergonomics.json) | 4 | 14 | 14 | 0 |
| [fatigue](calculators/fatigue.json) | 12 | 22 | 22 | 0 |
| [fea](calculators/fea.json) | 15 | 28 | 28 | 0 |
| [fluid_dynamics](calculators/fluid_dynamics.json) | 30 | 35 | 35 | 0 |
| [fluid_statics](calculators/fluid_statics.json) | 12 | 16 | 16 | 0 |
| [fracture](calculators/fracture.json) | 9 | 14 | 14 | 0 |
| [gears](calculators/gears.json) | 24 | 28 | 28 | 0 |
| [heat_transfer](calculators/heat_transfer.json) | 60 | 79 | 79 | 0 |
| [hvac](calculators/hvac.json) | 17 | 25 | 25 | 0 |
| [hydraulics](calculators/hydraulics.json) | 8 | 15 | 15 | 0 |
| [ic_engines](calculators/ic_engines.json) | 14 | 19 | 19 | 0 |
| [instrumentation](calculators/instrumentation.json) | 19 | 31 | 31 | 0 |
| [kinematics](calculators/kinematics.json) | 10 | 13 | 13 | 0 |
| [kinetics](calculators/kinetics.json) | 18 | 21 | 21 | 0 |
| [manufacturing](calculators/manufacturing.json) | 42 | 58 | 58 | 0 |
| [materials_mech](calculators/materials_mech.json) | 39 | 53 | 53 | 0 |
| [materials_sci](calculators/materials_sci.json) | 24 | 33 | 33 | 0 |
| [math](calculators/math.json) | 23 | 31 | 31 | 0 |
| [mechanisms](calculators/mechanisms.json) | 24 | 33 | 33 | 0 |
| [metrology](calculators/metrology.json) | 15 | 30 | 30 | 0 |
| [power_screws](calculators/power_screws.json) | 4 | 6 | 6 | 0 |
| [pressure_vessels](calculators/pressure_vessels.json) | 10 | 16 | 16 | 0 |
| [reliability](calculators/reliability.json) | 10 | 14 | 14 | 0 |
| [renewables](calculators/renewables.json) | 13 | 35 | 35 | 0 |
| [robotics](calculators/robotics.json) | 12 | 22 | 22 | 0 |
| [section_props](calculators/section_props.json) | 16 | 22 | 22 | 0 |
| [shafts](calculators/shafts.json) | 8 | 14 | 14 | 0 |
| [springs](calculators/springs.json) | 12 | 24 | 24 | 0 |
| [statics](calculators/statics.json) | 18 | 22 | 22 | 0 |
| [thermo](calculators/thermo.json) | 36 | 44 | 44 | 0 |
| [tribology](calculators/tribology.json) | 7 | 12 | 12 | 0 |
| [turbomachinery](calculators/turbomachinery.json) | 24 | 28 | 28 | 0 |
| [vibrations](calculators/vibrations.json) | 22 | 26 | 26 | 0 |
| [welded](calculators/welded.json) | 5 | 7 | 7 | 0 |
