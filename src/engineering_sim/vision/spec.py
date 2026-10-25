import hashlib
import json
def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
from copy import deepcopy
import math
import re
SPEC_VERSION = '1.0'
PROVENANCE = ('ocr', 'measured', 'scaled', 'matched', 'engine', 'assumed')
SOURCE_KINDS = ('drawing', 'image', 'images', 'video', 'manual')
ADDITIVE = ('extrude', 'revolve')
SUBTRACTIVE = ('pocket', 'hole', 'hole_pattern')
NAME = re.compile('[a-z][a-z0-9_]{0,63}')
MAX_FEATURES = 64
MAX_POLYGON_POINTS = 256
MAX_PATTERN_COUNT = 360
OP_FIELDS = {'extrude': ('z0', 'height'), 'revolve': ('angle_deg',), 'pocket': ('z_top', 'depth'), 'hole': ('cx', 'cy', 'diameter', 'z_top', 'depth')}
PATTERN_FIELDS = {'circular': ('cx', 'cy', 'pcd', 'start_deg', 'diameter', 'z_top', 'depth'), 'linear': ('x0', 'y0', 'dx', 'dy', 'diameter', 'z_top', 'depth')}
PROFILE_FIELDS = {'rect': ('x', 'y', 'width', 'height'), 'circle': ('cx', 'cy', 'r')}

class SpecError(ValueError):
    """A spec that the compiler must refuse; the message names the offending path."""

def _finite(value, where):
    """Require a real finite number (bool is rejected: True would silently mean 1 mm)."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SpecError(f'{where}: expected a finite number.')
    return float(value)

def _check_value(value, keys, where):
    """Validate one V (literal, parameter key or linear combination); return keys it reads."""
    if isinstance(value, str):
        if value not in keys:
            raise SpecError(f'{where}: unknown parameter "{value}".')
        return {value}
    if isinstance(value, dict):
        if set(value) - {'terms', 'const'} or not isinstance(value.get('terms', {}), dict):
            raise SpecError(f'{where}: a linear value has only "terms" and "const".')
        _finite(value.get('const', 0), where + '.const')
        for key, coef in value.get('terms', {}).items():
            if key not in keys:
                raise SpecError(f'{where}: unknown parameter "{key}".')
            _finite(coef, f'{where}.terms.{key}')
        return set(value.get('terms', {}))
    _finite(value, where)
    return set()

def resolve(value, parameters):
    """Evaluate one V against concrete parameter values (already validated)."""
    if isinstance(value, str):
        return float(parameters[value])
    if isinstance(value, dict):
        return float(value.get('const', 0)) + sum((coef * float(parameters[key]) for key, coef in value.get('terms', {}).items()))
    return float(value)

def feature_keys(feature, keys):
    """Return the sorted parameter keys one feature reads; also validates its fields.

    The compiler uses this exact set as the node's declared parameters, so the CAD
    cache key contains every input the builder uses and nothing it does not.
    """
    where = 'features.' + str(feature.get('id'))
    op = feature.get('op')
    used = set()
    if op in ADDITIVE or op == 'pocket':
        profile = feature.get('profile')
        if not isinstance(profile, dict):
            raise SpecError(where + '.profile: required.')
        kind = profile.get('kind')
        if op == 'revolve' and kind != 'polygon':
            raise SpecError(where + '.profile: revolve needs a polygon of [r, z] points.')
        if kind in PROFILE_FIELDS:
            for name in PROFILE_FIELDS[kind]:
                used |= _check_value(profile.get(name), keys, f'{where}.profile.{name}')
        elif kind == 'polygon':
            points = profile.get('points')
            if not isinstance(points, list) or not 3 <= len(points) <= MAX_POLYGON_POINTS:
                raise SpecError(f'{where}.profile.points: 3 to {MAX_POLYGON_POINTS} points required.')
            for i, point in enumerate(points):
                if not isinstance(point, list) or len(point) != 2:
                    raise SpecError(f'{where}.profile.points[{i}]: expected [x, y].')
                for j, coordinate in enumerate(point):
                    used |= _check_value(coordinate, keys, f'{where}.profile.points[{i}][{j}]')
        else:
            raise SpecError(where + '.profile.kind: use rect, circle or polygon.')
    if op in OP_FIELDS:
        names = OP_FIELDS[op]
    elif op == 'hole_pattern':
        if feature.get('pattern') not in PATTERN_FIELDS:
            raise SpecError(where + '.pattern: use circular or linear.')
        names = PATTERN_FIELDS[feature['pattern']]
        count = feature.get('count')
        if isinstance(count, str):
            if keys.get(count) != 'integer':
                raise SpecError(where + '.count: must reference an integer parameter.')
            used.add(count)
        elif type(count) is not int or not 1 <= count <= MAX_PATTERN_COUNT:
            raise SpecError(f'{where}.count: whole number 1 to {MAX_PATTERN_COUNT}.')
    else:
        raise SpecError(where + '.op: use extrude, revolve, pocket, hole or hole_pattern.')
    for name in names:
        used |= _check_value(feature.get(name), keys, f'{where}.{name}')
    return sorted(used)

def validate(spec):
    """Raise SpecError unless the spec is complete, bounded and internally consistent."""
    if not isinstance(spec, dict):
        raise SpecError('Spec must be a JSON object.')
    if spec.get('spec_version') != SPEC_VERSION:
        raise SpecError(f'spec_version must be "{SPEC_VERSION}".')
    if not isinstance(spec.get('name'), str) or not NAME.fullmatch(spec['name']):
        raise SpecError('name: lowercase letters, digits and underscores, starting with a letter.')
    if spec.get('units') != 'mm':
        raise SpecError('units: only "mm" is supported; convert before emitting a spec.')
    precision = spec.get('precision_mm')
    if precision not in (1, 0.1, 0.01, 0.001):
        raise SpecError('precision_mm: use 1, 0.1, 0.01 or 0.001.')
    source = spec.get('source')
    if not isinstance(source, dict) or source.get('kind') not in SOURCE_KINDS or (not isinstance(source.get('engine'), str)):
        raise SpecError('source: requires kind (' + ', '.join(SOURCE_KINDS) + ') and engine.')
    parameters = spec.get('parameters')
    if not isinstance(parameters, list) or not parameters:
        raise SpecError('parameters: at least one parameter is required.')
    keys = {}
    for i, parameter in enumerate(parameters):
        where = f'parameters[{i}]'
        key = parameter.get('key') if isinstance(parameter, dict) else None
        if not isinstance(key, str) or not NAME.fullmatch(key) or key in keys:
            raise SpecError(where + '.key: unique lowercase identifier required.')
        kind = parameter.get('type', 'number')
        if kind not in ('number', 'integer'):
            raise SpecError(where + '.type: number or integer.')
        low, high, value = (_finite(parameter.get(n), f'{where}.{n}') for n in ('min', 'max', 'value'))
        if not low <= value <= high:
            raise SpecError(where + ': value must lie within min..max.')
        if kind == 'integer' and any((int(v) != v for v in (low, high, value))):
            raise SpecError(where + ': integer parameters need whole min, max and value.')
        if parameter.get('provenance') not in PROVENANCE:
            raise SpecError(where + '.provenance: one of ' + ', '.join(PROVENANCE) + '.')
        if parameter.get('unit', 'count' if kind == 'integer' else 'mm') not in ('mm', 'deg', 'count'):
            raise SpecError(where + '.unit: mm, deg or count.')
        keys[key] = kind
    features = spec.get('features')
    if not isinstance(features, list) or not 1 <= len(features) <= MAX_FEATURES:
        raise SpecError(f'features: 1 to {MAX_FEATURES} features required.')
    ids = set()
    for i, feature in enumerate(features):
        if not isinstance(feature, dict) or not isinstance(feature.get('id'), str) or (not NAME.fullmatch(feature['id'])) or (feature['id'] in ids):
            raise SpecError(f'features[{i}].id: unique lowercase identifier required.')
        ids.add(feature['id'])
        if i == 0 and feature.get('op') not in ADDITIVE:
            raise SpecError('features[0]: the first feature must be extrude or revolve.')
        feature_keys(feature, keys)
    return spec

def _round(value, digits):
    """Quantize a float to the spec precision; -0.0 is normalized so it hashes like 0.0."""
    result = round(float(value), digits)
    return 0.0 if result == 0 else result

def canonical(spec):
    """Return a validated copy with every number rounded to precision_mm.

    Rounding happens once, here, so float noise from fitting or OCR can never
    produce two different recipe identities for the same design intent. Counts
    and integer parameters stay integers. Order of parameters and features is
    preserved because it is meaningful (form order, feature order).
    """
    spec = deepcopy(validate(spec))
    digits = max(0, -int(round(math.log10(spec['precision_mm']))))

    def walk(value):
        if isinstance(value, bool) or isinstance(value, str) or value is None:
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return _round(value, digits)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict) and 'terms' in value:
            return dict(terms={k: round(float(c), 12) for k, c in value['terms'].items()}, const=_round(value.get('const', 0), digits))
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value
    for parameter in spec['parameters']:
        for name in ('min', 'max', 'value'):
            parameter[name] = int(parameter[name]) if parameter.get('type') == 'integer' else _round(parameter[name], digits)
    spec['features'] = walk(spec['features'])
    return validate(spec)

def spec_id(spec):
    """SHA-256 of the canonical spec: the reproducibility handle recorded in every report."""
    return identity(canonical(spec))
