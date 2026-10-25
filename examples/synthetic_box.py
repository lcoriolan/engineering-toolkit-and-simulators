#!/usr/bin/env python3
"""A new synthetic FreeCAD example: exact box, native save, STEP, and reread checks.

Run with a Python interpreter that can import FreeCAD and Part. No source model,
internal dimension, or private toolkit code is used. Output must be a new folder.
"""
import argparse
import json
import math
from pathlib import Path


def positive(value):
    """Accept finite positive millimetre dimensions."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Use a finite positive dimension.')
    return value


def run(output, width=10.0, depth=20.0, height=30.0):
    """Build and independently reread one deliberately generic rectangular solid."""
    import FreeCAD as App
    import Part
    dimensions = [positive(v) for v in (width, depth, height)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = Part.makeBox(*dimensions)
    doc = App.newDocument('SyntheticExample')
    feature = doc.addObject('PartDesign::Feature', 'SyntheticBox')
    feature.Shape = original
    doc.recompute()
    native_path, step_path = output/'synthetic.FCStd', output/'synthetic.step'
    doc.saveAs(str(native_path))
    Part.export([feature], str(step_path))
    App.closeDocument(doc.Name)
    exchange = Part.read(str(step_path))
    reopened = App.openDocument(str(native_path))
    try:
        native = reopened.getObject('SyntheticBox').Shape.copy()
    finally:
        App.closeDocument(reopened.Name)
    expected = math.prod(dimensions)
    tolerance = max(1e-7, expected*1e-9)
    checks = {}
    for name, shape in [('original', original), ('step', exchange), ('native', native)]:
        checks[name+'_valid_single_solid'] = shape.isValid() and len(shape.Solids) == 1
        checks[name+'_analytic_volume'] = abs(shape.Volume-expected) <= tolerance
        checks[name+'_symmetric_difference'] = original.cut(shape).Volume + shape.cut(original).Volume <= tolerance
    report = dict(example='synthetic-public-example/1', freecad_version='.'.join(App.Version()[:3]),
                  dimensions_mm=dimensions, expected_volume_mm3=expected,
                  tolerance_mm3=tolerance, checks=checks, passed=all(checks.values()))
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('Synthetic round-trip validation failed.')
    return report


def main():
    """Select synthetic dimensions and a scratch destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=positive, default=10.0)
    parser.add_argument('--depth', type=positive, default=20.0)
    parser.add_argument('--height', type=positive, default=30.0)
    args = parser.parse_args()
    run(args.output, args.width, args.depth, args.height)


if __name__ == '__main__':
    main()
