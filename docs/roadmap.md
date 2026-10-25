# Proposed roadmap and community voting

**These are proposed future capabilities.** The integrated features described below are not currently implemented, and inclusion here is not a guaranteed commitment, release date or implementation order. Current capabilities and tested limits are documented in [coverage](coverage.md) and [validation](validation.md).

## How to vote

1. Sign in to GitHub and open a proposal below.
2. Add a **👍 reaction to the opening issue post**. You may vote for more than one proposal; each account contributes at most one thumbs-up per issue.
3. Comment with your use case, inputs, constraints and what a useful demonstration would need to prove. Suggest alternatives or a smaller first step. Written “+1” comments are not reaction votes.

Votes help identify demand alongside feasibility, dependencies and validation effort. They do not guarantee implementation. The table order is not a priority ranking.

## Options

| Proposal | Potential capability | Vote and discuss |
|---|---|---|
| Versioned design record | A shared record for geometry, materials, units, loads, tolerances, BOM and requirements. | [#1](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/1) |
| Dependency-aware engineering workflows | Rerun affected CAD, checks and simulations when inputs change, with explicit dependencies and stale-result tracking. | [#2](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/2) |
| Manufacturing, supplier and logistics adapters | Bring prices, lead times, machine capacity, process constraints and shipping estimates into engineering comparisons. | [#3](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/3) |
| Multidisciplinary design optimization | Compare feasible designs across performance, cost, manufacturability and delivery. | [#4](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/4) |
| Traceable engineering reports | Connect results to assumptions, sources, uncertainty, design revisions and validation coverage. | [#5](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/5) |
| Generic gearbox workflow demonstrator | An end-to-end example connecting CAD, fits, sampled gear motion, supported load cases, BOM and synthetic cost/lead-time comparisons. | [#6](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/6) |

The gearbox demonstrator could connect the design record, workflow, adapters and reporting proposals in one reproducible example. Cost and lead-time comparisons would use synthetic inputs initially. Supported load cases and model assumptions would be explicit; sampled motion would not establish continuous clearance, fatigue life or manufacturing readiness.

Browse [all proposals](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues?q=is%3Aissue%20is%3Aopen%20label%3Aroadmap) or [suggest an alternative](https://github.com/lcoriolan/engineering-toolkit-and-simulators/issues/new?template=feature_request.yml). Check for an existing issue first, then explain the engineering problem and a practical success criterion.
