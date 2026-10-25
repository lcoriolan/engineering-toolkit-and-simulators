import re
NUMBER = '(\\d+(?:\\.\\d+)?)'
TOLERANCE = '(?:\\s*±\\s*' + NUMBER + '|\\s*\\+\\s*' + NUMBER + '\\s*/?\\s*-\\s*' + NUMBER + ')?'
COUNT = '(?:(\\d+)\\s*x\\s*)?'
UNIT = '(?:\\s*(mm|in|"))?'
PATTERNS = (('thread', re.compile(COUNT + 'M' + NUMBER + '(?:\\s*x\\s*' + NUMBER + ')?', re.I)), ('chamfer', re.compile(NUMBER + '\\s*x\\s*45\\s*°')), ('diameter', re.compile(COUNT + '(?:DIA\\.?|⌀)\\s*' + NUMBER + TOLERANCE + UNIT, re.I)), ('radius', re.compile(COUNT + '\\bR\\s*' + NUMBER + TOLERANCE + UNIT)), ('angle', re.compile(NUMBER + '\\s*°' + TOLERANCE)), ('linear', re.compile(NUMBER + TOLERANCE + UNIT)))
ZERO_AS_DIAMETER = re.compile('(?<![\\d.])0(?=[1-9])')
PLUS_AS_PLUSMINUS = re.compile('(?<=\\d)(\\s*)\\+(?=\\s*\\d+(?:\\.\\d+)?(?![\\d.])(?!\\s*/?\\s*-\\s*\\d))')

def normalize_with_corrections(text):
    """Return (normalized text, corrections) where each correction is dict(index, rule).

    Length-changing spellings are unified first; the glyph corrections run last and
    are one-for-one character swaps, so their indexes are valid in the returned text.
    """
    text = re.sub('(?<=\\d),(?=\\d)', '.', text)
    text = re.sub('[Øø∅]', '⌀', text)
    text = text.replace('+/-', '±').replace('+-', '±')
    text = re.sub('\\bdeg\\b', '°', text, flags=re.I)
    corrections = []
    chars = list(text)
    for match in ZERO_AS_DIAMETER.finditer(text):
        chars[match.start()] = '⌀'
        corrections.append(dict(index=match.start(), rule='ocr_zero_read_as_diameter'))
    for match in PLUS_AS_PLUSMINUS.finditer(text):
        index = match.end() - 1
        chars[index] = '±'
        corrections.append(dict(index=index, rule='ocr_plus_read_as_plusminus'))
    text = ''.join(chars)
    text = re.sub('[×X](?=\\s*(?:\\d|M|⌀|R|DIA))', 'x', text)
    return (text, sorted(corrections, key=lambda c: c['index']))

def normalize(text):
    """Map the many OCR spellings of engineering symbols onto one form before matching."""
    return normalize_with_corrections(text)[0]

def _tolerance(groups):
    """Return (plus, minus) in source units from the three tolerance capture groups."""
    symmetric, plus, minus = groups
    if symmetric is not None:
        return (float(symmetric), float(symmetric))
    if plus is not None:
        return (float(plus), float(minus))
    return (None, None)

def parse(text, default_unit='mm'):
    """Return dimension records ordered by position in the normalized text.

    Each record: kind, text, start, and the fields for its kind
    (value_mm / value_deg, tolerance_plus_mm / tolerance_minus_mm, count, pitch_mm).
    Records touched by a glyph correction also carry `corrections` (rule names), and
    a ± restored from a lone + carries tolerance_ambiguous=True.
    """
    if default_unit not in ('mm', 'in'):
        raise ValueError('default_unit: mm or in.')
    text, corrections = normalize_with_corrections(text)
    claimed = []
    records = []
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if span[0] == span[1] or any((span[0] < end and start < span[1] for start, end in claimed)):
                continue
            g = match.groups()
            record = dict(kind=kind, text=match.group(0).strip(), start=span[0])
            if kind == 'thread':
                record.update(count=int(g[0]) if g[0] else 1, nominal_mm=float(g[1]), pitch_mm=float(g[2]) if g[2] else None)
            elif kind == 'chamfer':
                record.update(value_mm=float(g[0]), value_deg=45.0)
            elif kind in ('diameter', 'radius'):
                scale = 25.4 if (g[5] or default_unit) in ('in', '"') else 1.0
                plus, minus = _tolerance(g[2:5])
                record.update(count=int(g[0]) if g[0] else 1, value_mm=float(g[1]) * scale, tolerance_plus_mm=None if plus is None else plus * scale, tolerance_minus_mm=None if minus is None else minus * scale)
            elif kind == 'angle':
                plus, minus = _tolerance(g[1:4])
                record.update(value_deg=float(g[0]), tolerance_plus_deg=plus, tolerance_minus_deg=minus)
            else:
                scale = 25.4 if (g[4] or default_unit) in ('in', '"') else 1.0
                plus, minus = _tolerance(g[1:4])
                record.update(value_mm=float(g[0]) * scale, tolerance_plus_mm=None if plus is None else plus * scale, tolerance_minus_mm=None if minus is None else minus * scale)
            applied = [c['rule'] for c in corrections if span[0] <= c['index'] < span[1]]
            if applied:
                record['corrections'] = applied
                if 'ocr_plus_read_as_plusminus' in applied:
                    record['tolerance_ambiguous'] = True
            claimed.append(span)
            records.append(record)
    return sorted(records, key=lambda r: r['start'])
