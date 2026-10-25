# DESCRIPTION: Task A: PubChem record lookup with a local JSON cache (urllib only).
"""Cached PubChem PUG REST lookups.

Every property value extracted from PubChem is a cite.value record with
kind='cited', verified=True, source = the exact PUG View URL it was read from,
and `raw` = PubChem's original string kept verbatim next to the parsed number.
Numbers are parsed conservatively: we prefer °C/K for temperatures and never
convert units, so the reported value is always exactly what PubChem printed.
"""
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request

from engineering_sim.science import cite, paths

BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
VIEW_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view'
UA = 'EngineeringSim/1.0'
TIMEOUT = 20  # seconds

# Property headings we surface, in the exact spelling used by PUG View sections.
_TEMP_UNITS = ('°C', 'K', '°F')  # preferred order: °C/K first, °F last resort
_PRESSURE_UNITS = ('mmHg', 'kPa', 'bar', 'atm', 'torr', 'Pa', 'MPa', 'psi')
_DENSITY_UNITS = ('g/cm3', 'g/mL', 'kg/m3')



def _cache_key(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode('utf-8')).hexdigest()[:24]


def _get_json(url, data=None, timeout=TIMEOUT):
    """GET/POST a JSON endpoint with urllib; returns the parsed object."""
    headers = {'User-Agent': UA, 'Accept': 'application/json'}
    body = None
    if data is not None:
        body = data.encode('utf-8')
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _resolve_cid(query):
    """Resolve a name (or SMILES) to a PubChem CID; return (cid, method) or (None, None)."""
    # 1. name lookup
    try:
        d = _get_json(BASE + '/compound/name/' + urllib.parse.quote(query) + '/cids/JSON')
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        d = None
    if d and d.get('IdentifierList', {}).get('CID'):
        return d['IdentifierList']['CID'][0], 'name'
    # 2. SMILES lookup (POST form, so SMILES special characters survive)
    try:
        d = _get_json(BASE + '/compound/smiles/cids/JSON', data='smiles=' + urllib.parse.quote(query))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        d = None
    if d and d.get('IdentifierList', {}).get('CID'):
        return d['IdentifierList']['CID'][0], 'smiles'
    return None, None


def _find_sections(node, heading, out=None):
    """Depth-first search of a PUG View JSON tree for sections with a TOCHeading."""
    if out is None:
        out = []
    if isinstance(node, dict):
        if node.get('TOCHeading') == heading:
            out.append(node)
        for v in node.values():
            _find_sections(v, heading, out)
    elif isinstance(node, list):
        for v in node:
            _find_sections(v, heading, out)
    return out


def _normalize_units(text):
    """Map PubChem's unit spellings onto the canonical tokens the regex knows."""
    for a, b in (('−', '-'), ('° C', '°C'), ('° F', '°F'), ('mm Hg', 'mmHg'),
                 ('g/cu cm', 'g/cm3'), ('g/cu.cm', 'g/cm3'), ('g cm-3', 'g/cm3'),
                 ('g cm3', 'g/cm3'), ('kg m-3', 'kg/m3')):
        text = text.replace(a, b)
    return text


def _parse_measurement(text, units):
    """Return (value, unit) for a number+unit in `text`, or None.

    Prefers the unit listed earliest in `units` (so °C/K beats °F even when °F
    appears first in the string); never converts, so the value is verbatim from
    the source.
    """
    text = _normalize_units(text)
    alt = sorted(set(units), key=len, reverse=True)  # longest first so 'kPa' beats 'K'
    pat = re.compile(r'(-?\d+(?:\.\d+)?)\s*\[?\s*(' + '|'.join(re.escape(u) for u in alt) + r')\s*\]?', re.I)
    matches = pat.findall(text)
    if not matches:
        return None
    pref = {u: i for i, u in enumerate(units)}
    # min() keeps the first (leftmost) match when the preference index ties.
    value, unit = min(matches, key=lambda m: pref.get(m[1], len(units)))
    return float(value), unit


def _best_measurement(section, units):
    """Pick the best-cited (value, unit, raw, note) from a PUG View section.

    Prefers an entry whose unit is higher-preference and that has a reference
    name; ties break toward the earlier entry in PubChem's list.
    """
    candidates = []
    for info in section.get('Information', []):
        val = info.get('Value', {})
        refs = info.get('Reference') or []
        note = refs[0] if refs else None
        for sm in val.get('StringWithMarkup', []):
            if not isinstance(sm, dict):
                continue
            txt = sm.get('String', '')
            if not txt:
                continue
            m = _parse_measurement(txt, units)
            if m is None:
                continue
            value, unit = m
            pref = units.index(unit) if unit in units else len(units)
            candidates.append((pref, 0 if note else 1, len(candidates), value, unit, txt, note))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    _, _, _, value, unit, raw, note = candidates[0]
    return value, unit, raw, note


def _property_record(value, unit, raw, note, url, retrieved):
    """Wrap one parsed PubChem number in a cited value record, keeping `raw`."""
    rec = cite.value(value, unit, 'cited', source=url, retrieved=retrieved, verified=True,
                     note=(note or None))
    rec['raw'] = raw
    return rec


def _extract_properties(pug, url, retrieved):
    """Extract the physical-property records the brief names, when confidently parsed."""
    props = {}
    for heading, key, units in (
        ('Boiling Point', 'boiling_point', _TEMP_UNITS),
        ('Melting Point', 'melting_point', _TEMP_UNITS),
        ('Flash Point', 'flash_point', _TEMP_UNITS),
        ('Autoignition Temperature', 'autoignition_temperature', _TEMP_UNITS),
        ('Vapor Pressure', 'vapor_pressure', _PRESSURE_UNITS),
        ('Density', 'density', _DENSITY_UNITS),
    ):
        sections = _find_sections(pug, heading)
        best = None
        for s in sections:
            r = _best_measurement(s, units)
            if r is not None:
                best = r
                break
        if best is not None:
            value, unit, raw, note = best
            props[key] = _property_record(value, unit, raw, note, url, retrieved)
    # Flammable / explosive limits are a two-ended range, not a single number.
    flam = _extract_flammable(pug, url, retrieved)
    if flam is not None:
        props['flammable_limits'] = flam
    return props


def _extract_flammable(pug, url, retrieved):
    """Extract lower/upper flammability limits as a {raw, lower, upper} dict."""
    sections = _find_sections(pug, 'Flammable Limits')
    for s in sections:
        for info in s.get('Information', []):
            refs = info.get('Reference') or []
            note = refs[0] if refs else None
            for sm in info.get('Value', {}).get('StringWithMarkup', []):
                if not isinstance(sm, dict):
                    continue
                txt = sm.get('String', '')
                lo = re.search(r'[Ll]ower:\s*(\d+(?:\.\d+)?)\s*%', txt)
                hi = re.search(r'[Uu]pper:\s*(\d+(?:\.\d+)?)\s*%', txt)
                if lo and hi:
                    lower = _property_record(float(lo.group(1)), '% vol', txt, note, url, retrieved)
                    upper = _property_record(float(hi.group(1)), '% vol', txt, note, url, retrieved)
                    return {'raw': txt, 'lower': lower, 'upper': upper}
    # Fall back to separate LEL/UEL sections when no combined "Flammable Limits" row exists.
    lel = _first_percent(_find_sections(pug, 'Lower Explosive Limit (LEL)'), url, retrieved)
    uel = _first_percent(_find_sections(pug, 'Upper Explosive Limit (UEL)'), url, retrieved)
    if lel is not None or uel is not None:
        out = {'raw': '', 'lower': lel, 'upper': uel}
        if lel is not None:
            out['raw'] += lel.get('raw', '')
        if uel is not None:
            out['raw'] = (out['raw'] + ' ' + uel.get('raw', '')).strip()
        return out
    return None


def _first_percent(sections, url, retrieved):
    for s in sections:
        for info in s.get('Information', []):
            refs = info.get('Reference') or []
            note = refs[0] if refs else None
            for sm in info.get('Value', {}).get('StringWithMarkup', []):
                if not isinstance(sm, dict):
                    continue
                txt = sm.get('String', '')
                m = re.search(r'(\d+(?:\.\d+)?)\s*%', txt)
                if m:
                    rec = cite.value(float(m.group(1)), '% vol', 'cited', source=url,
                                     retrieved=retrieved, verified=True, note=(note or None))
                    rec['raw'] = txt
                    return rec
    return None


def _extract_ghs(pug, url):
    """Extract GHS hazard statements (H-codes), statements, and pictogram codes."""
    ghs_codes = set()
    statements = set()
    pictograms = set()
    for s in _find_sections(pug, 'GHS Classification'):
        for info in s.get('Information', []):
            name = info.get('Name')
            for sm in info.get('Value', {}).get('StringWithMarkup', []):
                if not isinstance(sm, dict):
                    continue
                if name == 'GHS Hazard Statements':
                    txt = sm.get('String', '').strip()
                    if txt:
                        statements.add(txt)
                        ghs_codes.update(re.findall(r'\bH\d{3}\b', txt))
                elif name == 'Pictogram(s)':
                    for mk in sm.get('Markup', []):
                        m = re.search(r'/ghs/(GHS\d+)\.svg', mk.get('URL', ''))
                        if m:
                            pictograms.add(m.group(1))
    return {
        'ghs_codes': sorted(ghs_codes),
        'statements': sorted(statements),
        'pictograms': sorted(pictograms),
        'source': url,
    }


def _fetch(query):
    """Full live fetch: CID -> properties -> PUG View, assembled into the result dict."""
    cid, method = _resolve_cid(query)
    if cid is None:
        return {'found': False, 'query': query}
    prop_url = BASE + '/compound/cid/%d/property/MolecularFormula,MolecularWeight,IsomericSMILES,IUPACName/JSON' % cid
    prop = _get_json(prop_url)['PropertyTable']['Properties'][0]
    view_url = VIEW_BASE + '/data/compound/%d/JSON' % cid
    pug = _get_json(view_url)
    retrieved = cite.today()

    mw_raw = prop.get('MolecularWeight')
    mw = None
    if mw_raw is not None:
        mw = cite.value(float(mw_raw), 'g/mol', 'cited', source=prop_url, retrieved=retrieved,
                        verified=True, note='PubChem MolecularWeight')
    result = {
        'found': True,
        'cid': cid,
        'name': prop.get('IUPACName') or pug.get('Record', {}).get('RecordTitle'),
        'formula': prop.get('MolecularFormula'),
        'mw': mw,
        'smiles': prop.get('IsomericSMILES') or prop.get('SMILES') or prop.get('CanonicalSMILES'),
        'url': 'https://pubchem.ncbi.nlm.nih.gov/compound/%d' % cid,
        'retrieved': retrieved,
        'properties': _extract_properties(pug, view_url, retrieved),
        'hazards': _extract_ghs(pug, view_url),
    }
    return result



def lookup(query, refresh=False, offline=False):
    """Look up a compound by name (or SMILES) from PubChem, with a local cache.

    Returns {found, cid, name, formula, mw, smiles, url, retrieved, properties,
    hazards} on success; {found: False, offline: True} when offline with no
    cache; {found: False, error: ...} on network failure. Never raises.
    """
    q = (query or '').strip()
    if not q:
        return {'found': False, 'error': 'empty query'}
    cache_path = paths.data_dir() / 'records' / (_cache_key(q) + '.json')
    if not refresh and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass  # corrupt cache -> fall through to a fresh fetch
    if offline:
        return {'found': False, 'offline': True}
    try:
        result = _fetch(q)
    except Exception as e:  # noqa: BLE001 - network/parse errors become a soft failure
        return {'found': False, 'error': '%s: %s' % (type(e).__name__, e)}
    if result.get('found'):
        paths.write_cache(cache_path, result)
    return result
