# Capability inventory

Use the [coverage matrix](coverage.md) to compare available implementation families, dependencies, executed checks and remaining limits. The [module/API inventory](../evidence/api-inventory.json) lists all 247 Python modules and 1,778 top-level functions and classes. These counts include helpers and are not counts of independent solvers or validated methods.

The [calculator benchmark index](../benchmarks/README.md) covers 827 calculators across 50 families with 1,232 declared reference cases. [Solver benchmarks](../benchmarks/solvers.json) cover 20 synthetic numerical problems. Native CAD checks, browser control tests and CPU/GPU parity have separately stated scopes in [validation](validation.md).

Available code includes mechanical calculators; chemistry, thermodynamics, biochemistry and materials operators; particle/world and domain models; flow and finite-element operators; exact CAD import/rebuild and parametric geometry; and tolerance, fit, motion and manufacturing checks. Follow each API's units, assumptions and required inputs. An importable function without applicable validation remains unqualified.
