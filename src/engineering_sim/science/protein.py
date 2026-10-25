# DESCRIPTION: Task E: protein structure lookup (RCSB PDB + AlphaFold DB) and sequence tools.
"""Protein structure lookup and sequence analysis.

Two structure sources, both cached under ``<data>/proteins/<source>_<id>.*``:

  source 'pdb'       experimental entries from the RCSB PDB (mmCIF + entry metadata)
  source 'alphafold' predicted models from the AlphaFold DB (PDB model + UniProt metadata)

``fetch`` classifies a query as a PDB id, a UniProt accession, or a free-text name
(which returns a candidate list for the page to choose from — it never guesses).
Every metadata value (title, method, resolution, deposition date, organism) is a
``cite.value`` record carrying the exact URL it was read from; ``verified`` is True
only because those URLs are fetched at runtime.

``sequence`` wraps Biopython's ``ProteinAnalysis`` (the ExPASy ProtParam
equivalent) and returns every reported quantity as a ``cite`` value record.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

import gemmi

from engineering_sim.science import cite, paths

UA = 'EngineeringSim/1.0'
TIMEOUT = 60  # seconds; structure files are larger than PubChem records

# PDB ids are 4 characters, first character a digit (e.g. 1UBQ, 1HSG).
_PDB_RE = re.compile(r'^[0-9][A-Z0-9]{3}$')
# UniProtKB accessions: 6-char Swiss-Prot ([OPQ]...) or 10-char TrEMBL (A0A...).
_ACC_RE = re.compile(r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{3}[0-9])$')

# The 20 standard amino-acid one-letter codes accepted by sequence().
_AA20 = set('ACDEFGHIKLMNPQRSTVWY')

_MAX_ATOMS_SMALL = 5000  # atoms_small carries every heavy atom only below this cap



# --- HTTP helpers -----------------------------------------------------------
def _get_bytes(url, timeout=TIMEOUT):
    """GET a URL and return raw bytes; raises urllib errors on failure."""
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url, timeout=TIMEOUT):
    """GET a JSON endpoint and return the parsed object."""
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def _post_json(url, payload, timeout=TIMEOUT):
    """POST a JSON body and return the parsed object (RCSB full-text search)."""
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=body,
                                 headers={'User-Agent': UA, 'Content-Type': 'application/json',
                                          'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


# --- mmCIF metadata helpers -------------------------------------------------
def _cif_tag(block, tag):
    """First non-blank value of an mmCIF tag, with quoting stripped, or None."""
    values = block.find_values(tag)
    if values is None:
        return None
    for v in values:
        s = v.strip()
        if s and s not in ('?', '.'):
            return s.strip("'\"")
    return None


def _row_get(row, key):
    """Value of a gemmi cif Row column by name, or None (gemmi .get() is index-only)."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def _chemcomp_names(block):
    """Map 3-letter chemical-component ids to their mmCIF names."""
    names = {}
    cat = block.find_mmcif_category('_chem_comp.')
    if cat is not None:
        for row in cat:
            cid, name = _row_get(row, 'id'), _row_get(row, 'name')
            if cid and name:
                names[cid] = name.strip(';').strip()
    return names


def _chemcomp_bonds(block, code):
    """Heavy-atom bond list [(a1, a2, order, aromatic)] for one chemical component."""
    bonds = []
    cat = block.find_mmcif_category('_chem_comp_bond.')
    if cat is None:
        return bonds
    for row in cat:
        if _row_get(row, 'comp_id') != code:
            continue
        a1, a2 = _row_get(row, 'atom_id_1'), _row_get(row, 'atom_id_2')
        if a1 and a2:
            bonds.append((a1, a2, _row_get(row, 'value_order'), _row_get(row, 'pdbx_aromatic_flag')))
    return bonds


# --- gemmi helpers ----------------------------------------------------------
def _one_letter(res_name):
    """Three-letter residue name -> one-letter code (or 'X' when unknown)."""
    info = gemmi.find_tabulated_residue(res_name)
    code = info.one_letter_code
    return code if code else 'X'


def _classify_residue(res):
    """Residue class: 'polymer', 'water' or 'hetero' (ligand)."""
    if res.is_water():
        return 'water'
    if res.het_flag == 'H':
        return 'hetero'
    return 'polymer'


def _parse_structure(st, source):
    """Turn a gemmi Structure into chains, C-alpha trace and the heavy-atom list."""
    model = st[0]
    chains = []
    trace = []
    heavy_atoms = []
    for chain in model:
        polymer = [r for r in chain if _classify_residue(r) == 'polymer']
        if not polymer:
            continue
        seq = ''.join(_one_letter(r.name) for r in polymer)
        chains.append({'id': chain.name, 'length': len(polymer), 'sequence': seq})
        for r in polymer:
            ca = None
            for atom in r:
                if atom.name == 'CA':
                    ca = atom
                if atom.element.name != 'H':
                    heavy_atoms.append({'el': atom.element.name, 'chain': chain.name,
                                        'resn': r.name, 'resi': r.seqid.num,
                                        'x': round(atom.pos.x, 3), 'y': round(atom.pos.y, 3),
                                        'z': round(atom.pos.z, 3)})
            if ca is not None:
                item = {'chain': chain.name, 'resi': r.seqid.num, 'resn': r.name,
                        'x': round(ca.pos.x, 3), 'y': round(ca.pos.y, 3), 'z': round(ca.pos.z, 3)}
                if source == 'alphafold':
                    item['plddt'] = round(float(ca.b_iso), 2)
                trace.append(item)
    return chains, trace, heavy_atoms


def _collect_ligands(model, block, source):
    """Non-water hetero residues with names, heavy atoms and heavy-atom bonds."""
    names = _chemcomp_names(block)
    ligands = []
    for chain in model:
        for r in chain:
            if _classify_residue(r) != 'hetero':
                continue
            atoms = []
            name2idx = {}
            for atom in r:
                if atom.element.name == 'H':
                    continue
                name2idx[atom.name] = len(atoms)
                atoms.append({'el': atom.element.name,
                              'x': round(atom.pos.x, 3), 'y': round(atom.pos.y, 3),
                              'z': round(atom.pos.z, 3)})
            if not atoms:
                continue
            # Bonds come from the CCD definition carried inside the mmCIF; they are
            # only the heavy-atom bonds whose atoms are actually present here.
            bonds = []
            for a1, a2, _order, _arom in _chemcomp_bonds(block, r.name):
                if a1 in name2idx and a2 in name2idx:
                    bonds.append([name2idx[a1], name2idx[a2]])
            ligands.append({'code': r.name, 'name': names.get(r.name),
                            'chain': chain.name, 'resi': r.seqid.num,
                            'atom_count': len(atoms), 'atoms': atoms, 'bonds': bonds})
    return ligands


def _plddt(trace):
    """Mean pLDDT + AF confidence-band fractions, derived from the C-alpha trace."""
    vals = [t['plddt'] for t in trace if 'plddt' in t]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    # AlphaFold confidence bands: >90 very high, 70-90 confident, 50-70 low, <50 very low.
    n_vh = sum(1 for v in vals if v > 90.0)
    n_cf = sum(1 for v in vals if 70.0 < v <= 90.0)
    n_lo = sum(1 for v in vals if 50.0 < v <= 70.0)
    n_vl = sum(1 for v in vals if v <= 50.0)
    bands = {
        'very_high': cite.value(round(n_vh / n, 4), None, 'derived',
                                method='fraction of residues in AlphaFold confidence band',
                                note='%d of %d residues' % (n_vh, n)),
        'confident': cite.value(round(n_cf / n, 4), None, 'derived',
                                method='fraction of residues in AlphaFold confidence band',
                                note='%d of %d residues' % (n_cf, n)),
        'low': cite.value(round(n_lo / n, 4), None, 'derived',
                          method='fraction of residues in AlphaFold confidence band',
                          note='%d of %d residues' % (n_lo, n)),
        'very_low': cite.value(round(n_vl / n, 4), None, 'derived',
                               method='fraction of residues in AlphaFold confidence band',
                               note='%d of %d residues' % (n_vl, n)),
    }
    mean_rec = cite.value(round(mean, 2), None, 'derived',
                          method='mean of per-residue pLDDT (model B-factor column)')
    return {'mean': mean_rec, 'bands': bands, 'source': None}


def _plddt_from_trace(trace, model_url):
    """Build the plddt sub-dict, attaching the model URL as the source of the raw values."""
    out = _plddt(trace)
    if out is not None:
        out['source'] = model_url
    return out


# --- source-specific fetchers ----------------------------------------------
def _fetch_pdb(pid, cache_dir):
    """Fetch and parse an experimental RCSB PDB entry."""
    cif_url = 'https://files.rcsb.org/download/%s.cif' % pid
    entry_url = 'https://data.rcsb.org/rest/v1/core/entry/%s' % pid
    cif_path = cache_dir / ('pdb_%s.cif' % pid)
    cif_path.write_bytes(_get_bytes(cif_url))
    entry = _get_json(entry_url)
    retrieved = cite.today()
    st = gemmi.read_structure(str(cif_path))
    block = gemmi.cif.read(str(cif_path))[0]

    info = entry.get('rcsb_entry_info', {})
    resolution = None
    rcs = info.get('resolution_combined')
    if isinstance(rcs, list) and rcs:
        resolution = float(rcs[0])
    elif info.get('diffrn_resolution_high') is not None:
        resolution = float(info['diffrn_resolution_high'].get('value'))

    deposit = entry.get('rcsb_accession_info', {}).get('deposit_date')
    title = entry.get('struct', {}).get('title')
    method = info.get('experimental_method')
    organism = _cif_tag(block, '_entity_src_gen.pdbx_gene_src_scientific_name')

    model = st[0]
    chains, trace, heavy_atoms = _parse_structure(st, 'pdb')
    ligands = _collect_ligands(model, block, 'pdb')

    result = {
        'found': True,
        'ref': 'pdb_' + pid,
        'source': 'pdb',
        'id': pid,
        'title': cite.value(title, None, 'cited', source=entry_url, retrieved=retrieved,
                            verified=True) if title else None,
        'kind': 'experimental',
        'method': cite.value(method, None, 'cited', source=entry_url, retrieved=retrieved,
                             verified=True) if method else None,
        'resolution_A': cite.value(resolution, 'Å', 'cited', source=entry_url,
                                   retrieved=retrieved, verified=True) if resolution is not None else None,
        'deposition_date': cite.value(deposit, None, 'cited', source=entry_url,
                                      retrieved=retrieved, verified=True) if deposit else None,
        'organism': cite.value(organism, None, 'cited', source=cif_url, retrieved=retrieved,
                               verified=True) if organism else None,
        'chains': chains,
        'ligands': ligands,
        'plddt': None,
        'url': 'https://www.rcsb.org/structure/' + pid,
        'retrieved': retrieved,
        'trace': trace,
        'atoms_small': heavy_atoms if 0 < len(heavy_atoms) <= _MAX_ATOMS_SMALL else None,
        'atom_count': len(heavy_atoms),
        'notes': ['Structure from the RCSB Protein Data Bank (experimental). '
                  'Coordinates, resolution and metadata read from files.rcsb.org and data.rcsb.org.'],
    }
    return result


def _fetch_alphafold(acc, cache_dir):
    """Fetch and parse a predicted AlphaFold DB model + UniProt entry."""
    af_url = 'https://alphafold.ebi.ac.uk/api/prediction/%s' % acc
    preds = _get_json(af_url)
    if not preds:
        return {'found': False, 'error': 'AlphaFold DB has no model for %s' % acc}
    pred = preds[0]
    model_url = pred.get('pdbUrl') or pred.get('cifUrl')
    if not model_url:
        return {'found': False, 'error': 'AlphaFold DB returned no model URL for %s' % acc}
    model_path = cache_dir / ('alphafold_%s.pdb' % acc)
    model_path.write_bytes(_get_bytes(model_url))
    retrieved = cite.today()

    up = None
    up_url = 'https://rest.uniprot.org/uniprotkb/%s.json' % acc
    try:
        up = _get_json(up_url)
    except urllib.error.HTTPError:
        up = None  # model is still usable without the UniProt entry

    st = gemmi.read_structure(str(model_path))
    chains, trace, heavy_atoms = _parse_structure(st, 'alphafold')

    title = (up or {}).get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value')
    organism = (up or {}).get('organism', {}).get('scientificName')
    method = pred.get('toolUsed')

    result = {
        'found': True,
        'ref': 'alphafold_' + acc,
        'source': 'alphafold',
        'id': acc,
        'title': cite.value(title, None, 'cited', source=up_url, retrieved=retrieved,
                            verified=True) if title else None,
        'kind': 'predicted',
        'method': cite.value(method, None, 'cited', source=af_url, retrieved=retrieved,
                             verified=True) if method else None,
        'resolution_A': None,
        'deposition_date': cite.value(pred.get('modelCreatedDate'), None, 'cited', source=af_url,
                                      retrieved=retrieved, verified=True) if pred.get('modelCreatedDate') else None,
        'organism': cite.value(organism, None, 'cited', source=up_url, retrieved=retrieved,
                               verified=True) if organism else None,
        'chains': chains,
        'ligands': [],
        'plddt': _plddt_from_trace(trace, model_url),
        'url': 'https://alphafold.ebi.ac.uk/entry/' + acc,
        'retrieved': retrieved,
        'trace': trace,
        'atoms_small': heavy_atoms if 0 < len(heavy_atoms) <= _MAX_ATOMS_SMALL else None,
        'atom_count': len(heavy_atoms),
        'notes': ['Predicted structure from the AlphaFold DB (AlphaFold2); it is a model, '
                  'not an experimentally determined structure. Per-residue confidence (pLDDT) '
                  'is read from the model B-factor column.'],
    }
    return result


# --- name search ------------------------------------------------------------
def _search(name, offline):
    """Free-text name search: RCSB full-text + UniProt, merged into candidates."""
    if offline:
        return {'found': False, 'offline': True}
    candidates = []
    # RCSB experimental structures (full-text search, top 5).
    try:
        payload = {
            'query': {'type': 'terminal', 'service': 'full_text',
                      'parameters': {'value': name}},
            'return_type': 'entry',
            'request_options': {'paginate': {'start': 0, 'rows': 5},
                                'results_content_type': ['experimental']},
        }
        hits = _post_json('https://search.rcsb.org/rcsbsearch/v2/query', payload)
        for hit in hits.get('result_set', [])[:5]:
            pid = hit.get('identifier')
            if not pid:
                continue
            title = organism = None
            try:
                entry = _get_json('https://data.rcsb.org/rest/v1/core/entry/%s' % pid)
                title = entry.get('struct', {}).get('title')
                info = entry.get('rcsb_entry_info', {})
                if info.get('experimental_method') in ('X-ray', 'NMR', 'EM', 'ELECTRON MICROSCOPY', 'NEUTRON'):
                    pass
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
                pass
            candidates.append({'id': pid, 'source': 'pdb', 'title': title, 'organism': organism})
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
        pass  # search failing must not fail the op; the other source may still answer

    # UniProtKB search (top 5) -> predicted AlphaFold models.
    try:
        up = _get_json('https://rest.uniprot.org/uniprotkb/search?query=%s&size=5&format=json'
                       % urllib.parse.quote(name))
        for row in up.get('results', []):
            acc = row.get('primaryAccession')
            if not acc:
                continue
            title = row.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value')
            organism = row.get('organism', {}).get('scientificName')
            candidates.append({'id': acc, 'source': 'alphafold', 'title': title, 'organism': organism})
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
        pass

    if not candidates:
        return {'found': False, 'error': 'No structures found for "%s"' % name}
    return {'found': False, 'candidates': candidates, 'query': name}


# --- public API -------------------------------------------------------------
def _classify(query, source):
    """Decide how to interpret a query: PDB id, UniProt accession, or name."""
    upper = query.upper().strip()
    if source in ('auto', 'pdb') and _PDB_RE.match(upper):
        return 'pdb', upper
    if source in ('auto', 'alphafold') and _ACC_RE.match(upper):
        return 'alphafold', upper
    if source == 'pdb':
        return 'pdb', upper
    if source == 'alphafold':
        return 'alphafold', upper
    return 'search', query.strip()



def fetch(query, source='auto', refresh=False, offline=False):
    """Fetch a protein structure by PDB id, UniProt accession, or name.

    Returns the parsed structure record (``found: True``), a ``candidates`` list
    when a free-text name was given, or ``{found: False, ...}``. Network errors
    never raise; cached results are re-read when ``offline`` is set.
    """
    q = (query or '').strip()
    if not q:
        return {'found': False, 'error': 'empty query'}
    kind, key = _classify(q, source)
    if kind == 'search':
        return _search(key, offline)

    cache_dir = paths.data_dir() / 'proteins'
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / ('%s_%s.json' % (kind, key))
    if not refresh and json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass  # corrupt cache -> fetch fresh
    if offline:
        return {'found': False, 'offline': True}
    try:
        result = _fetch_pdb(key, cache_dir) if kind == 'pdb' else _fetch_alphafold(key, cache_dir)
    except Exception as exc:  # noqa: BLE001 - network/parse errors become a soft failure
        return {'found': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
    if result.get('found'):
        paths.write_cache(json_path, result)
    return result


def resolve_structure(ref):
    """Load a cached structure for a fetch ``ref`` (e.g. 'pdb_1UBQ').

    Returns (gemmi.Structure, source) or (None, None). Used by md.py and docking.py.
    """
    if not ref:
        return None, None
    d = paths.data_dir() / 'proteins'
    for ext in ('cif', 'pdb'):
        p = d / ('%s.%s' % (ref, ext))
        if p.exists():
            return gemmi.read_structure(str(p)), ref.split('_', 1)[0]
    return None, None


def structure_file_path(ref):
    """Path of the cached structure file for a fetch ``ref``, or None."""
    if not ref:
        return None
    d = paths.data_dir() / 'proteins'
    for ext in ('cif', 'pdb'):
        p = d / ('%s.%s' % (ref, ext))
        if p.exists():
            return p
    return None


def _clean_sequence(seq):
    """Strip whitespace/digits and validate one-letter codes; raise ValueError on bad input."""
    cleaned = re.sub(r'[\s0-9]', '', seq or '')
    invalid = sorted({c for c in cleaned if c not in _AA20})
    if invalid:
        raise ValueError('Invalid residue letter(s): %s — only the 20 standard amino-acid '
                         'one-letter codes are supported.' % ', '.join(invalid))
    if not cleaned:
        raise ValueError('Empty sequence after stripping whitespace and numbers.')
    return cleaned


def sequence(seq):
    """Compute ProtParam-style properties for an amino-acid sequence.

    All quantities are ``cite`` records: exact arithmetic (length, molecular
    weight, extinction coefficient, GRAVY, aromaticity, composition) is
    ``derived``; pKa/statistics-model outputs (pI, charge, instability,
    secondary-structure fraction) are ``estimated``.
    """
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    cleaned = _clean_sequence(seq)
    pa = ProteinAnalysis(cleaned)
    pa_mono = ProteinAnalysis(cleaned, monoisotopic=True)
    n = len(cleaned)
    method = 'Biopython ProtParam (ExPASy-equivalent)'

    mw_avg = pa.molecular_weight()
    mw_mono = pa_mono.molecular_weight()
    ext = pa.molar_extinction_coefficient()  # (reduced, cystine)
    composition = pa.count_amino_acids()
    percent = pa.amino_acids_percent  # cached property in Biopython 1.88 (a dict)
    ss = pa.secondary_structure_fraction()  # (helix, turn, sheet)

    charge_curve_points = []
    for ph_i in range(0, 29):  # pH 0..14 in 0.5 steps
        ph = ph_i * 0.5
        charge_curve_points.append([ph, round(float(pa.charge_at_pH(ph)), 4)])

    result = {
        'found': True,
        'sequence': cleaned,
        'length': cite.value(n, 'residues', 'derived', method='residue count'),
        'mw_avg': cite.value(round(mw_avg, 3), 'Da', 'derived', method=method),
        'mw_monoisotopic': cite.value(round(mw_mono, 3), 'Da', 'derived', method=method),
        'pi': cite.value(round(float(pa.isoelectric_point()), 3), None, 'estimated',
                         method=method + ' pKa model'),
        'charge_pH7': cite.value(round(float(pa.charge_at_pH(7.0)), 3), 'e', 'estimated',
                                 method=method + ' charge_at_pH'),
        'charge_curve': {'kind': 'estimated', 'method': method + ' charge_at_pH',
                         'points': charge_curve_points},
        'extinction_280_reduced': cite.value(round(float(ext[0]), 1), 'M⁻¹cm⁻¹', 'derived',
                                             method=method),
        'extinction_280_cystine': cite.value(round(float(ext[1]), 1), 'M⁻¹cm⁻¹', 'derived',
                                             method=method),
        'gravy': cite.value(round(float(pa.gravy()), 4), None, 'derived', method=method),
        'instability_index': cite.value(round(float(pa.instability_index()), 2), None, 'estimated',
                                        method=method),
        'aromaticity': cite.value(round(float(pa.aromaticity()), 4), None, 'derived', method=method),
        'aa_composition': {
            'kind': 'derived',
            'method': method,
            'counts': composition,
            'percent': {aa: round(pct, 3) for aa, pct in percent.items()},
        },
        'ss_fraction': {
            'helix': cite.value(round(float(ss[0]), 4), None, 'estimated',
                                method=method + ' Chou-Fasman'),
            'turn': cite.value(round(float(ss[1]), 4), None, 'estimated',
                               method=method + ' Chou-Fasman'),
            'sheet': cite.value(round(float(ss[2]), 4), None, 'estimated',
                                method=method + ' Chou-Fasman'),
        },
        'notes': [
            'Computed with Biopython ProtParam (the same algorithms as the ExPASy ProtParam tool); '
            'these are sequence-derived estimates, not measured values.',
            'Instability index: a value above 40 predicts an unstable protein in vitro '
            '(Guruprasad et al. 1990).',
        ],
    }
    if result['instability_index']['value'] > 40:
        result['notes'].append('Instability index > 40: predicted unstable.')
    return result


def from_structure_sequence(result, chain=None):
    """Run sequence() on one chain of a fetch result (default: the first chain)."""
    chains = (result or {}).get('chains') or []
    if not chains:
        return {'found': False, 'error': 'the structure has no chains'}
    target = chains[0]
    if chain is not None:
        target = next((c for c in chains if c.get('id') == chain), None)
    if target is None:
        return {'found': False, 'error': 'chain not found: %s' % chain}
    try:
        out = sequence(target.get('sequence', ''))
    except ValueError as exc:
        return {'found': False, 'error': str(exc)}
    out['chain'] = target.get('id')
    out['structure'] = (result or {}).get('id')
    return out
