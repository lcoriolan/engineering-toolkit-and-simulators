# CAD ↔ Python

A parameter edit is meaningful only when it regenerates checked geometry. `cad.builder` accepts box, cylinder, cone, sphere, torus, union, cut and common nodes with explicit dimensions/placements. It writes native FreeCAD features and verifies STEP export. The public input `examples/parametric_solid.json` is a new synthetic box with a cylindrical bore.

`cad.importer.convert(source, destination, name)` accepts supported native/exchange solids. Recognized analytic primitives and eligible planar-profile extrusions become editable native features/sketches and Python. Unsupported shapes retain exact BREP and are explicitly labelled as such. A mesh is not promoted to a precise parametric solid, and an arbitrary feature history cannot be recovered from STEP.

Equivalence compares volume, surface area, optimal bounding bounds and both material differences `original.cut(candidate)` / `candidate.cut(original)`. The returned record includes actual tolerances. These are finite numerical checks, not symbolic identity proofs. Imported source BREP is not healed or silently fitted. Exported source packs include the geometry needed to reconstruct the caller-supplied shape.

Generated source packs preserve required dispatcher code and native reconstruction. Builds avoid interpreter bytecode contaminating deterministic archives. Tests exercise primitive recognition, placement, native parameter changes, unsupported-shape retention, STEP round trips and repeatable ZIP bytes.

`cad.features` also supplies 20 caller-parameterized operations: plates/pads, holes, pockets, slots, flanges, gussets, fillets/chamfers, tubes, scoops/ducts, fins, noses, mounts and envelopes. Inspect the required parameters in the API inventory and supply each dimension explicitly. Missing required dimensions raise an error. Feature geometry coverage is listed separately from the importer tests.

`cad.hydrostatics` clips exact closed hull solids and bisects upright heave equilibrium from mass and fluid density. It reports overload instead of assigning a floating waterline. This excludes trim, roll stability and seakeeping; water-domain modules provide separate approximate models.
