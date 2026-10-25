# DESCRIPTION: Resolve any way of writing a molecule — formula, name, SMILES, InChI, InChIKey, CAS — to one canonical structure, so every notation starts the same science toolkit run.
"""One molecule, many notations, one canonical answer.

Equivalent formula, SMILES, InChI, common-name and registry notations resolve to a canonical structure.

Resolution order, first hit wins (each step records how it resolved, in `via`):
  1. explicit prefix          smiles:… / formula:… / name:… / inchi:… / inchikey:… / cas:…
  2. InChI / InChIKey         parsed by RDKit (InChI) or looked up (InChIKey)
  3. CAS number               PubChem name lookup (PubChem indexes CAS as a synonym)
  4. common-formula table     H2O, CO2, NH3, … resolved offline, no network, no ambiguity
  5. SMILES                   RDKit parse (so `O` is water and `CO` is methanol, per SMILES rules)
  6. general formula          PubChem name lookup of the formula string
  7. name                     PubChem name lookup

Ambiguity note: a string can be both a formula and a SMILES. `CO` is carbon monoxide as a formula but
methanol as SMILES; `NO` is nitric oxide as a formula but hydroxylamine as SMILES. The common-formula
table (step 4) resolves the frequent inorganic cases the way a chemist reads them, and any input can be
forced with a prefix (`smiles:CO` vs `formula:CO`). Every result says which rule fired.

Network use: only steps 3, 6 and 7, through engineering_sim.science.records.lookup (PubChem, cached on disk).
Pass offline=True to refuse those instead of calling out; the table and SMILES/InChI paths stay offline.
"""
import re
import unicodedata

# Written the way a chemist reads them; SMILES with implicit hydrogens. Offline and unambiguous.
COMMON_FORMULAS = {
    'H2O': 'O', 'D2O': '[2H]O[2H]', 'H2O2': 'OO', 'O2': 'O=O', 'O3': '[O-][O+]=O', 'H2': '[H][H]',
    'N2': 'N#N', 'NH3': 'N', 'N2O': 'N#[N+][O-]', 'NO': '[N]=O', 'NO2': '[N+](=O)[O-]', 'HCN': 'C#N',
    'CO': '[C-]#[O+]', 'CO2': 'O=C=O', 'CH4': 'C', 'C2H2': 'C#C', 'C2H4': 'C=C', 'C2H6': 'CC',
    'C3H8': 'CCC', 'C4H10': 'CCCC', 'CH3OH': 'CO', 'C2H5OH': 'CCO', 'CH2O': 'C=O', 'CH3COOH': 'CC(=O)O',
    'C6H6': 'c1ccccc1', 'C6H12O6': 'OCC1OC(O)C(O)C(O)C1O',
    'HCl': 'Cl', 'HBr': 'Br', 'HI': 'I', 'HF': 'F', 'H2S': 'S', 'SO2': 'O=S=O', 'SO3': 'O=S(=O)=O',
    'H2SO4': 'OS(=O)(=O)O', 'HNO3': 'O[N+](=O)[O-]', 'H3PO4': 'OP(=O)(O)O', 'H2CO3': 'OC(=O)O',
    'NaCl': '[Na+].[Cl-]', 'NaOH': '[Na+].[OH-]', 'KOH': '[K+].[OH-]', 'CaCO3': '[Ca+2].[O-]C([O-])=O',
    'He': '[He]', 'Ne': '[Ne]', 'Ar': '[Ar]', 'Kr': '[Kr]', 'Xe': '[Xe]',
    'Cl2': 'ClCl', 'Br2': 'BrBr', 'I2': 'II', 'F2': 'FF', 'P4': 'P12P3P1P23', 'S8': 'S1SSSSSSS1',
}
PREFIXES = ('smiles', 'formula', 'name', 'inchi', 'inchikey', 'cas')
INCHIKEY_RE = re.compile(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$')
CAS_RE = re.compile(r'^\d{2,7}-\d{2}-\d$')
FORMULA_RE = re.compile(r'^(?:[A-Z][a-z]?\d*)+$')
_SUBSCRIPTS = {0x2080 + i: str(i) for i in range(10)}          # ₀…₉ → 0…9


def normalise_text(text):
    """Strip decorations people type around formulas: Unicode subscripts, LaTeX `_2`/`^{2+}`, spaces, dots."""
    if not isinstance(text, str):
        raise ValueError('Start must be a string.')
    s = unicodedata.normalize('NFKC', text).strip()
    s = s.translate(_SUBSCRIPTS)                                # H₂O -> H2O
    s = re.sub(r'\$|\\mathrm|\\text|\{|\}', '', s)              # $\mathrm{H_2O}$ -> H_2O
    s = re.sub(r'(?<=[A-Za-z])_(\d)', r'\1', s)                 # H_2O -> H2O
    s = re.sub(r'\^\{?([0-9]*[+-])\}?', r'\1', s)               # SO4^{2-} -> SO42-
    return re.sub(r'\s+', '', s)


def _canonical(smiles):
    """Canonical isomeric SMILES for a parsed structure; raises ValueError if RDKit refuses it."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('RDKit could not parse the structure: ' + smiles)
    return Chem.MolToSmiles(mol)


def _pubchem_by_inchikey(key, offline):
    """SMILES for an InChIKey from PubChem's inchikey endpoint (the name endpoint does not accept keys).

    Cached under <data>/records/inchikey_<KEY>.json so a repeat resolve is offline and reproducible.
    """
    import json
    import urllib.error
    import urllib.request
    from engineering_sim.science import paths
    # Cache first: a key resolved once replays offline, which is what makes a recipe reproducible.
    cache = paths.data_dir() / 'records' / ('inchikey_' + key + '.json')
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        if offline:
            raise ValueError('Offline: %r is not cached and needs a PubChem InChIKey lookup.' % key)
        url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/%s/property/'
               'SMILES,IsomericSMILES,MolecularFormula,Title/JSON' % key)
        request = urllib.request.Request(url, headers={'User-Agent': 'EngineeringSim/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise ValueError('PubChem InChIKey lookup failed: %s' % exc)
        except OSError as exc:
            raise ValueError('PubChem InChIKey lookup failed: %s' % exc)
        data = dict(data, url=url)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1, sort_keys=True))
    props = (data.get('PropertyTable') or {}).get('Properties') or []
    if not props:
        return None
    first = props[0]
    smiles = first.get('IsomericSMILES') or first.get('SMILES') or first.get('CanonicalSMILES')
    if not smiles:
        return None
    return dict(smiles=_canonical(smiles), pubchem_cid=first.get('CID'), pubchem_name=first.get('Title'),
                source=data.get('url'))


def _from_pubchem(query, offline, via):
    """PubChem-backed resolution (cached); returns a result dict or None when nothing matched."""
    if offline:
        raise ValueError('Offline: %r needs a PubChem lookup (formula, name, CAS or InChIKey).' % query)
    from engineering_sim.science.records import lookup
    rec = lookup(query)
    if not rec.get('found') or not rec.get('smiles'):
        return None
    return dict(smiles=_canonical(rec['smiles']), via=via, pubchem_cid=rec.get('cid'),
                pubchem_name=rec.get('name'), source=rec.get('url'))


def resolve(text, offline=False):
    """Resolve any notation to {smiles, via, input, normalised, …}; raises ValueError with the reason.

    `smiles` is the canonical isomeric SMILES — the identity every science toolkit run keys on.
    `via` names the rule that fired: prefix:<kind>, inchi, inchikey, cas, common_formula, smiles,
    formula_pubchem or name_pubchem.
    """
    raw = text if isinstance(text, str) else ''
    s = normalise_text(raw)
    if not s:
        raise ValueError('Start is empty.')

    kind = None
    if ':' in s and s.split(':', 1)[0].lower() in PREFIXES and not s.lower().startswith('inchi=1'):
        kind, s = s.split(':', 1)[0].lower(), s.split(':', 1)[1]
        if not s:
            raise ValueError('Start is empty after the %r prefix.' % kind)

    def done(result):
        return dict(result, input=raw, normalised=s)

    # InChI is unambiguous and offline-parsable.
    if (kind == 'inchi') or (kind is None and s.lower().startswith('inchi=')):
        from rdkit import Chem
        mol = Chem.MolFromInchi(s)
        if mol is None:
            raise ValueError('RDKit could not parse that InChI.')
        return done(dict(smiles=Chem.MolToSmiles(mol), via='prefix:inchi' if kind else 'inchi'))

    # InChIKey is a hash: only a lookup can recover the structure.
    if (kind == 'inchikey') or (kind is None and INCHIKEY_RE.match(s)):
        hit = _pubchem_by_inchikey(s.upper(), offline)
        if hit:
            return done(dict(hit, via='prefix:inchikey' if kind else 'inchikey'))
        raise ValueError('No PubChem compound for that InChIKey.')

    # CAS registry numbers are indexed by PubChem as synonyms.
    if (kind == 'cas') or (kind is None and CAS_RE.match(s)):
        hit = _from_pubchem(s, offline, 'prefix:cas' if kind else 'cas')
        if hit:
            return done(hit)
        raise ValueError('No PubChem compound for that CAS number.')

    if kind == 'smiles':
        return done(dict(smiles=_canonical(s), via='prefix:smiles'))

    if kind == 'name':
        hit = _from_pubchem(s, offline, 'prefix:name')
        if hit:
            return done(hit)
        raise ValueError('No PubChem compound named %r.' % s)

    if kind == 'formula' or (kind is None and FORMULA_RE.match(s)):
        # Common inorganic formulas resolve offline the way a chemist reads them (CO = carbon monoxide),
        # which also keeps them out of the SMILES branch below (where CO would be methanol).
        table = COMMON_FORMULAS.get(s) or COMMON_FORMULAS.get(s.capitalize())
        if table:
            return done(dict(smiles=_canonical(table), via='prefix:formula' if kind else 'common_formula'))
        if kind == 'formula':
            hit = _from_pubchem(s, offline, 'prefix:formula')
            if hit:
                return done(hit)
            raise ValueError('No single PubChem compound for formula %r; give a SMILES or a name.' % s)

    # Unprefixed: SMILES first (a valid SMILES is an exact structure), then PubChem as formula/name.
    if kind is None:
        from rdkit import Chem
        if Chem.MolFromSmiles(s) is not None:
            return done(dict(smiles=_canonical(s), via='smiles'))
        if offline:
            # Say why it failed: offline, this input would need a PubChem name/formula lookup.
            raise ValueError('Offline: %r is not a SMILES, InChI or known formula, so it needs a PubChem lookup.' % raw)
        hit = _from_pubchem(s, offline, 'formula_pubchem' if FORMULA_RE.match(s) else 'name_pubchem')
        if hit:
            return done(hit)
        raise ValueError('Could not resolve %r as SMILES, formula, name, InChI, InChIKey or CAS.' % raw)

    raise ValueError('Unknown start notation: ' + raw)
