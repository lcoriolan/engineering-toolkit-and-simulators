# DESCRIPTION: Concoctions calculator — one concoction (compounded formulation, thermoset
#              system or molecule) plus a target batch mass -> the materials shopping list
#              and the steps to create it. Deterministic recipe_id; every cited number keeps
#              its provenance record; on-hand status is checked against the shared inventory
#              and the science toolkit stock file.
"""Concoctions extension area.

Imported by ``engineering_sim/science/api.py`` under the standard-library studio interpreter to
read timeouts (standard library only at module level) and by ``engineering_sim/science/worker.py``
under configured science environment to run handlers. Every op name starts with ``concoct_``.

The calculator is a materials-and-procedure view of what science toolkit already models:
formulations screen through the compounding engine, thermosets through the cure
screen, molecules through the bench, PubChem vendor listings and — when a stability
ladder job exists for the molecule — the AiZynthFinder retrosynthesis route. All
masses are parametric in the batch mass. Nothing here claims kinetics, prices beyond
the library's recalled bands, or process qualification.
"""

import hashlib
import json
import os
from pathlib import Path
import re

TIMEOUTS = {
    'concoct_plan': 90,
    'concoct_stock': 30,
}

VERSION = '1.1.0'

_LIMITS = [
    'Masses are wt% of the total batch; the resin / part-A balance absorbs rounding.',
    'Price bands are the library\'s recalled bands, not quotes; catalogue keys are generated '
    'placeholders until a vendor catalogue is attached.',
    'On-hand status reads an explicitly configured inventory and the '
    'science toolkit stock file; "not tracked" means no stock record exists, not that the material is absent.',
    'Formulation and thermoset steps are a compounding/cure procedure from cited library processing '
    'fields, not a process qualification: residence, shear and dispersion are tool-dependent.',
    'Molecule shopping lists name the materials; vendor sourcing (PubChem listings) is only looked up '
    'when "Check vendor listings" is ticked.',
    'Molecule steps are gas-phase thermodynamics only unless a stability-ladder retrosynthesis job '
    'exists for the molecule; no kinetics is predicted. AiZynthFinder routes give connectivity, not masses.',
]


# ---- shared helpers -------------------------------------------------------------
def _stock():
    """Stock from an explicitly configured inventory plus the caller-local science stock file."""
    from engineering_sim.science import paths

    shared = {}
    configured_inventory = os.environ.get('ENGINEERING_SIM_INVENTORY')
    inv = Path(configured_inventory).expanduser() if configured_inventory else None
    if inv is not None and inv.is_file():
        try:
            shared = json.loads(inv.read_text()).get('chemicals') or {}
        except Exception:
            shared = {}
    local_file = paths.data_dir() / 'stock.json'
    local = {}
    if local_file.is_file():
        try:
            local = json.loads(local_file.read_text()).get('rows') or {}
        except Exception:
            local = {}
    else:

        try:
            local_file.write_text(json.dumps({
                '_note': 'science toolkit bench stock (owner-maintained). Rows: {id: {label, qty, unit, note}}.',
                'rows': {},
            }, indent=1))
        except Exception:
            pass
    return {'shared': shared, 'local': local}


def _on_hand(rid, label, stock):
    for rows in (stock['local'], stock['shared']):
        for key, row in rows.items():
            if not isinstance(row, dict):
                continue
            if key == rid or (row.get('label') or '').lower() == (label or '').lower():
                qty = row.get('qty')
                detail = ('%s %s' % (qty, row['unit']) if qty is not None and row.get('unit')
                          else str(qty) if qty is not None else 'recorded, quantity not stated')
                return {'status': 'on hand', 'detail': detail, 'stock': key}
    return {'status': 'not tracked', 'detail': 'no stock record', 'stock': None}


def _shopping_row(rid, label, role, wt_pct, qty_g, price, stock, note=None):
    hand = _on_hand(rid, label, stock)
    key_role = re.sub(r'[^A-Z0-9]+', '_', role.upper()).strip('_')
    row = {'key': 'TL.%s.%s' % (key_role, str(rid).upper().replace('-', '_')),
           'material': label, 'role': role, 'wt_pct': wt_pct, 'qty_g': qty_g,
           'price': price, 'on_hand': hand['status'], 'on_hand_detail': hand['detail']}
    if note:
        row['note'] = note
    return row


def _num(rec):
    from engineering_sim.science import cite
    return cite.num(rec) if rec else None


def _field(record, key):
    v = record.get(key)
    return v if isinstance(v, dict) and 'value' in v else None


def _find(records, rid):
    for r in records:
        if r['id'] == rid:
            return r
    return None


# ---- formulation ----------------------------------------------------------------
def _plan_formulation(spec, batch_g, stock):
    from engineering_sim.science import compound, documents

    resin_id = (spec.get('resin') or '').strip()
    if not resin_id:
        raise ValueError('Formulation: choose a resin.')
    resin = _find(compound.library('resins'), resin_id)
    if resin is None:
        raise ValueError('Formulation: unknown resin id: ' + resin_id)
    comps, fills = [], []
    for c in spec.get('components') or []:
        rec = _find(compound.library('additives'), c.get('id'))
        if rec is not None:
            comps.append({'record': rec, 'wt': float(c.get('wt') or 0)})
    for f in spec.get('fillers') or []:
        rec = _find(compound.library('fillers'), f.get('id'))
        if rec is not None:
            fills.append({'record': rec, 'wt': float(f.get('wt') or 0), 'length_mm': f.get('length_mm')})
    other = sum(x['wt'] for x in comps) + sum(x['wt'] for x in fills)
    if other >= 100.0:
        raise ValueError('Formulation: component wt% must total less than 100 (resin takes the balance).')
    resin_wt = 100.0 - other

    shopping = [_shopping_row(resin['id'], resin['name'], 'resin', round(resin_wt, 2),
                              round(batch_g * resin_wt / 100.0, 3), resin.get('price_band'), stock)]
    for x in comps:
        shopping.append(_shopping_row(x['record']['id'], x['record']['name'], 'additive', x['wt'],
                                      round(batch_g * x['wt'] / 100.0, 3),
                                      x['record'].get('price_band'), stock))
    for x in fills:
        note = ('length %g mm' % x['length_mm']) if x.get('length_mm') else None
        shopping.append(_shopping_row(x['record']['id'], x['record']['name'], 'filler', x['wt'],
                                      round(batch_g * x['wt'] / 100.0, 3),
                                      x['record'].get('price_band'), stock, note=note))

    steps, cautions = [], []
    n = [0]
    def step(title, detail, cite=None):
        n[0] += 1
        steps.append({'n': n[0], 'title': title, 'detail': detail, 'cite': cite})

    dry = (_field(resin, 'drying_temp_c'), _field(resin, 'drying_time_h'),
           _field(resin, 'drying_dewpoint_c'), _field(resin, 'drying_max_moisture_pct'))
    if any(v is not None for v in dry):
        t, h, dew, m = dry
        detail = 'Dry %s' % resin['name']
        if t is not None:
            detail += ' at %.0f °C' % t['value']
        if h is not None:
            detail += ' for %.1f h' % h['value']
        bits = []
        if dew is not None:
            bits.append('dew point ≤ %.0f °C' % dew['value'])
        if m is not None:
            bits.append('moisture ≤ %s wt%%' % m['value'])
        if bits:
            detail += ' (' + '; '.join(bits) + ')'
        detail += '.'
        step('Dry the resin', detail, resin.get('drying_temp_c'))
    elif _field(resin, 'moisture_sensitive'):
        step('Dry the resin', 'Follow the supplier drying guide (%s is moisture sensitive); the library '
             'records no schedule for it.' % resin['name'], resin.get('moisture_sensitive'))

    weigh = 'Weigh %s %.1f g (%.1f wt%%)' % (resin['name'], shopping[0]['qty_g'], resin_wt)
    for row in shopping[1:]:
        weigh += ' · %s %.1f g (%.1f wt%%)' % (row['material'], row['qty_g'], row['wt_pct'])
    step('Weigh the batch', weigh + '.', {'value': 'qty = batch_g × wt% / 100', 'unit': 'g',
                                          'kind': 'derived', 'method': 'mass balance over the batch'})
    lo, hi = _field(resin, 'melt_temp_min_c'), _field(resin, 'melt_temp_max_c')
    if lo or hi:
        rng = '%.0f–%.0f °C' % (lo['value'] if lo else hi['value'], hi['value'] if hi else lo['value'])
        step('Heat to melt temperature', 'Bring %s to its melt range %s.' % (resin['name'], rng),
             lo or hi or resin.get('melt_temp_min_c'))
    for x in comps + fills:
        rec, wt = x['record'], x['wt']
        note = ''
        lo_l, hi_l = _field(rec, 'loading_min_wt_pct'), _field(rec, 'loading_max_wt_pct')
        lo_v = lo_l.get('value') if lo_l is not None else None
        hi_v = hi_l.get('value') if hi_l is not None else None
        if lo_v is not None and wt < lo_v or hi_v is not None and wt > hi_v:
            note = ' outside the library loading range %s–%s wt%%.' % (lo_v if lo_v is not None else '–',
                                                                     hi_v if hi_v is not None else '–')
        tga = _field(rec, 'tga_1pct_loss_c')
        # Library records can carry a field with no value (e.g. ADR-4468 TGA): skip the comparison then.
        if tga and hi and tga.get('value') is not None and hi.get('value') is not None \
                and tga['value'] < hi['value']:
            note += ' Add late: TGA 1%% loss at %.0f °C is below the melt temperature.' % tga['value']
        step('Incorporate %s' % rec['name'], 'Add %.1f g at %.1f wt%% and mix.%s' % (
            batch_g * wt / 100.0, wt, note), rec.get('loading_max_wt_pct'))
    step('Mix to dispersion', 'Mix to visual dispersion. See limits: no mixing model (shear/residence are tool-dependent).')
    mold = _field(spec.get('process') or {}, 'mold_temp_c')
    if mold is not None:
        step('Shape', 'Mold at %.0f °C per the process inputs.' % mold['value'],
             {'value': mold['value'], 'unit': 'degC', 'kind': 'user', 'note': 'process input'})
    else:
        step('Shape', 'Mold or press to shape per the tooling; no mold temperature was given.')

    try:
        screen = compound.screen(spec)
        for f in screen.get('findings') or []:
            cautions.append(f if isinstance(f, str) else (f.get('text') or f.get('message') or str(f)))
        for f in screen.get('notes') or []:
            cautions.append(f)
    except Exception as exc:
        cautions.append('Screening unavailable for this formulation: %s' % exc)

    body = {'kind': 'formulation', 'name': '%s compound' % resin['name'],
            'resin': {'id': resin['id'], 'name': resin['name'], 'wt_pct': round(resin_wt, 2)},
            'shopping': shopping, 'steps': steps, 'cautions': cautions}
    try:
        body['qc_plan'] = documents.qc_plan(spec)
    except Exception:
        pass
    return body


# ---- thermoset ------------------------------------------------------------------
def _plan_thermoset(spec, batch_g, stock):
    from pathlib import Path
    from engineering_sim.science import thermoset

    system_id = (spec.get('system') or '').strip()
    if not system_id:
        raise ValueError('Thermoset: choose a system.')
    sys = {'system': system_id}
    if spec.get('nco_index') is not None:
        sys['nco_index'] = float(spec['nco_index'])
    screen = thermoset.screen(sys)
    phr = _num(screen.get('mix_ratio_phr')) if screen.get('mix_ratio_phr') else None
    if phr is None:
        raise ValueError('Thermoset: the library has no stoichiometric mix data for %s.' % system_id)
    records = json.loads((Path(__file__).resolve().parent / 'data' / 'thermosets.json').read_text())['records']
    rec = _find(records, system_id)
    full = rec.get('name') or system_id
    # "PU (MDI + polyether polyol)" -> parts "MDI" / "polyether polyol"; "DGEBA + DETA" -> "DGEBA" / "DETA".
    m = re.match(r'^(.*?)\s*\((.+\+.+)\)\s*$', full)
    names = (m.group(2) if m else full).split(' + ')
    name_a = names[0].strip() or 'part A'
    name_b = names[1].strip() if len(names) > 1 else 'hardener (part B)'

    a_g = batch_g * 100.0 / (100.0 + phr)
    b_g = batch_g * phr / (100.0 + phr)
    shopping = [
        _shopping_row(system_id + '_a', name_a, 'resin (part A)', round(100.0 * 100.0 / (100.0 + phr), 2),
                      round(a_g, 3), rec.get('price_band'), stock),
        _shopping_row(system_id + '_b', name_b, 'hardener (part B)', round(100.0 * phr / (100.0 + phr), 2),
                      round(b_g, 3), rec.get('price_band'), stock),
    ]

    steps, cautions = [], []
    n = [0]
    def step(title, detail, cite=None):
        n[0] += 1
        steps.append({'n': n[0], 'title': title, 'detail': detail, 'cite': cite})

    step('Weigh part A', 'Weigh %s: %.1f g.' % (name_a, a_g),
         {'value': a_g, 'unit': 'g', 'kind': 'derived', 'method': 'part A = batch × 100 / (100 + phr)'})
    step('Weigh part B', 'Weigh %s: %.1f g at %.2f phr.' % (name_b, b_g, phr), screen.get('mix_ratio_phr'))
    for f in screen.get('flags') or []:
        cautions.append(f)
    for h in screen.get('hazards') or []:
        cautions.append('hazard: ' + str(h))
    pot = rec.get('pot_life')
    step('Mix', 'Combine and mix thoroughly. Respect the pot life: %s.'
         % (pot['value'] if isinstance(pot, dict) and pot.get('value') else 'see library'),
         pot if isinstance(pot, dict) and pot.get('value') else None)
    step('Degas', 'Degas the mixture (vacuum or slow pour) before molding; the library records no schedule.')
    schedule = rec.get('cure_schedule')
    schedule = schedule.get('value') if isinstance(schedule, dict) else schedule
    schedule = schedule or 'not recorded'
    step('Mold and cure', 'Pour into the mold and cure per the library schedule: %s.' % schedule,
         {'value': schedule, 'unit': '-', 'kind': 'unverified',
          'source': 'thermoset library record (%s)' % system_id})
    gel = screen.get('gel_conversion')
    if gel is not None and _num(gel) is not None:
        cautions.append('Gel conversion %.2f (Flory-Stockmayer); work before the gel point.' % _num(gel))
    elif gel is not None:
        cautions.append('No gel point: %s' % gel.get('method'))
    return {'kind': 'thermoset', 'name': rec.get('name') or system_id,
            'system': system_id, 'mix_ratio_phr': screen.get('mix_ratio_phr'),
            'shopping': shopping, 'steps': steps, 'cautions': cautions}


# ---- molecule -------------------------------------------------------------------
def _molecule_element_counts(mol):
    counts = {}
    for a in mol.get('atoms') or []:
        el = a.get('el')
        if el:
            counts[el] = counts.get(el, 0) + 1
    return counts


def _thermo_key_for(elements):
    """Species key whose element composition equals ``elements`` (e.g. an atom or the molecule)."""
    from engineering_sim.science import thermo
    for entry in thermo.species():
        sp = thermo.species(entry['key'])
        if sp.get('elements') == elements:
            return entry['key']
    return None


def _find_stability_job(smiles):
    """The most recent done stability-ladder job for this SMILES, or None."""
    from engineering_sim.science import paths
    jobs = paths.data_dir() / 'stability'
    if not jobs.is_dir():
        return None
    best = None
    for d in jobs.iterdir():
        if not (d / 'request.json').is_file() or not (d / 'state.json').is_file():
            continue
        try:
            req = json.loads((d / 'request.json').read_text())
            state = json.loads((d / 'state.json').read_text())
        except Exception:
            continue
        if req.get('smiles') == smiles and state.get('state') == 'done' and state.get('rungs', {}).get('5'):
            if best is None or d.stat().st_mtime > best.stat().st_mtime:
                best = d
    return best


def _retro_steps(tree):
    """Flatten an AiZynthFinder route tree into creation steps (leaves first).

    Tree shape (stability_retro.tree_of): molecule node {smiles, in_stock, children:
    [reaction node {reaction, template_code, policy_probability, children: [molecule
    nodes]}]}. A reaction's children are its reactants; it produces the parent molecule.
    """
    out = []
    def walk(mol_node):
        for ch in mol_node.get('children') or []:
            if ch.get('reaction'):
                for reactant in ch.get('children') or []:
                    walk(reactant)
                out.append({'reactants': [c.get('smiles') for c in ch.get('children') or []],
                            'product': mol_node.get('smiles'),
                            'template': ch.get('template_code'), 'p': ch.get('policy_probability')})
            else:
                walk(ch)
    walk(tree)
    steps = []
    for i, r in enumerate(out, 1):
        code = r['template'] or 'unnamed USPTO template'
        p = (' (policy p=%.2f)' % r['p']) if isinstance(r['p'], (int, float)) else ''
        steps.append({'n': i, 'title': 'Step %d' % i,
                      'detail': 'React %s via %s%s to form %s.'
                                % (' + '.join(r['reactants']) or 'purchased stock', code, p, r['product']),
                      'cite': {'value': code, 'unit': '-', 'kind': 'estimated',
                               'method': 'AiZynthFinder policy (USPTO-trained)',
                               'note': 'connectivity only; no masses or conditions'}})
    return steps


def _material_name(smiles):
    """Common name for a SMILES: PubChem compound Title, resolved by SMILES only (a name-first
    lookup misreads short SMILES such as "CCO" as synonyms of other compounds). Cached; no vendor
    search. Falls back to the RDKit formula, then the SMILES itself."""
    import urllib.parse
    from engineering_sim.science import paths, records
    cache = paths.data_dir() / 'concoct' / ('name_%s.json' % hashlib.sha256(smiles.encode()).hexdigest()[:16])
    try:
        if cache.is_file():
            return json.loads(cache.read_text())['name']
        d = records._get_json(records.BASE + '/compound/smiles/cids/JSON',
                              data='smiles=' + urllib.parse.quote(smiles))
        cid = (d.get('IdentifierList') or {}).get('CID', [0])[0]
        if cid:
            t = records._get_json(records.BASE + '/compound/cid/%d/property/Title/JSON' % cid)
            title = t['PropertyTable']['Properties'][0].get('Title')
            if title:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({'smiles': smiles, 'cid': cid, 'name': title}))
                return title
    except Exception:
        pass
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        pass
    return smiles


def _plan_molecule(query, batch_g, stock, check_vendors=False):
    from engineering_sim.science import chem, thermo, paths

    if not (query or '').strip():
        raise ValueError('Molecule: enter a name, SMILES, InChI or CAS.')
    mol = chem.molecule(query)
    if mol.get('error'):
        raise ValueError('Molecule: %s' % mol['error'])
    smiles = mol.get('smiles') or ''
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.error')
    typed_smiles = Chem.MolFromSmiles(query.strip()) is not None
    RDLogger.EnableLog('rdApp.error')
    # Shopping lists show the name the user typed; a typed SMILES is resolved to its common name.
    name = (_material_name(smiles) if typed_smiles and smiles else query.strip()) or mol.get('name') or query
    counts = _molecule_element_counts(mol)

    job = _find_stability_job(smiles)
    retro, pv = {}, {}
    if job is not None:
        state = json.loads((job / 'state.json').read_text())
        r5 = state.get('rungs', {}).get('5') or {}
        ev = r5.get('evidence') or {}
        retro = ev.get('retrosynthesis') or {}
        pv = ev.get('pubchem_vendors') or {}
    if not pv and check_vendors:
        from engineering_sim.science import stability
        from types import SimpleNamespace
        ctx = SimpleNamespace(cache_dir=paths.data_dir() / 'concoct')
        pv = stability.pubchem_vendors(ctx, smiles)

    shopping, steps, cautions = [], [], []
    n = [0]
    def step(title, detail, cite=None):
        n[0] += 1
        steps.append({'n': n[0], 'title': title, 'detail': detail, 'cite': cite})

    note_buy = ''
    if isinstance(pv, dict) and (pv.get('vendors') or 0) > 0:
        note_buy = 'PubChem lists %d chemical-vendor source(s) (secondary flag: a listing, not a stock or price check).' % pv['vendors']
        if pv.get('cid'):
            note_buy += ' CID %s, retrieved %s.' % (pv['cid'], pv.get('retrieved'))
    elif isinstance(pv, dict) and pv.get('error'):
        cautions.append('PubChem vendor check unavailable: %s' % pv['error'])

    route = (retro.get('routes') or [None])[0] if retro.get('routes') else None
    in_stock_target = bool(retro.get('target_in_stock'))
    solved = bool(retro.get('solved'))

    if route is not None and route.get('tree') and solved:
        # Real synthesis route: shopping list = the in-stock starting materials.
        seen = set()
        for m in route.get('starting_materials') or []:
            s = m.get('smiles', '')
            if s in seen:
                continue
            seen.add(s)
            label = _material_name(s)
            row = _shopping_row('mol_' + hashlib.sha256(s.encode()).hexdigest()[:8], label, 'starting material',
                                None, None, None, stock,
                                note='SMILES %s. AiZynthFinder route gives connectivity, not masses; '
                                     'scale per the balanced template.' % s)
            row['smiles'] = s
            shopping.append(row)
        steps.extend(_retro_steps(route['tree']))
        if note_buy:
            cautions.append(note_buy)
        cautions.append('Route from AiZynthFinder (USPTO-trained policy, ZINC stock): %d step(s). Masses and '
                        'conditions are not part of the route model.' % (retro.get('steps') or len(steps)))
    elif not check_vendors or in_stock_target or (isinstance(pv, dict) and (pv.get('vendors') or 0) > 0):
        # Buyable: the shopping list is the compound itself.
        buy_note = note_buy or ('The molecule is in the ZINC stock.' if in_stock_target else '')
        shopping.append(_shopping_row('mol', name, 'compound', None, round(batch_g, 3),
                                      None, stock, note=buy_note or None))
        step('Procure', ('Obtain %s (%.1f g). %s' % (name, batch_g, buy_note)).strip())
        cite = {'value': smiles, 'unit': '-', 'kind': 'cited', 'source': 'PubChem (CID %s)' % pv.get('cid'),
                'retrieved': pv.get('retrieved'), 'verified': False} \
            if isinstance(pv, dict) and pv.get('cid') else None
        step('Verify identity', 'Confirm identity (NMR, melting point or mass spec) before use.', cite)
        if not check_vendors and not note_buy and not in_stock_target:
            cautions.append('Sourcing not checked: tick "Check vendor listings" to look the compound up in '
                            'PubChem vendor listings (and fall back to a formation route if none exist).')
        elif not in_stock_target:
            cautions.append('Buyable by vendor listings only (not in the checked stock); confirm availability before ordering.')
    else:
        # Not buyable and no solved route: fall back to formation from elements (bench thermo).
        cautions.append('Not buyable: no PubChem vendor listings and no solved retrosynthesis route. '
                        'Formation from elements below is the remaining creation route.')
        prod_key = _thermo_key_for(counts)
        atom_keys = {}
        for el, c in counts.items():
            k = _thermo_key_for({el: 1})
            if k is not None:
                atom_keys[el] = k
        if prod_key is not None and len(atom_keys) == len(counts):
            try:
                rx = thermo.reaction({atom_keys[el]: c for el, c in counts.items()}, {prod_key: 1})
            except ValueError as exc:
                rx = None
                cautions.append('Formation reaction could not be balanced: %s' % exc)
            if rx is not None:
                from rdkit.Chem import GetPeriodicTable
                pt = GetPeriodicTable()
                total_mw = 0.0
                mws = {}
                for el, c in counts.items():
                    mw = c * pt.GetAtomicWeight(el)
                    mws[el] = mw
                    total_mw += mw
                for el, c in counts.items():
                    sp = thermo.species(atom_keys[el])
                    shopping.append(_shopping_row('el_' + el.lower(), sp.get('name') or el,
                                                  'element precursor', None, round(batch_g * mws[el] / total_mw, 3),
                                                  None, stock,
                                                  note='scaled by atomic mass from the bench element balance'))
                eq = ' + '.join('%g %s' % (c, el) for el, c in sorted(counts.items())) + ' → ' + name
                step('Form from elements', 'Combine the elements as %s (gas phase).' % eq, rx.get('dG'))
                step('Conditions note', 'Thermodynamics only: the bench reports what is favored at equilibrium, '
                     'not whether or how fast it happens (no kinetics). Two gas-phase atoms cannot bond without '
                     'a third body to carry the bond energy.')
        else:
            cautions.append('The molecule is not in the bench thermochemistry database; no formation step is reported.')

    return {'kind': 'molecule', 'name': name, 'formula': mol.get('formula'), 'smiles': smiles,
            'shopping': shopping, 'steps': steps, 'cautions': cautions,
            'job_id': job.name if job is not None else None}


# ---- handlers -------------------------------------------------------------------
def _plan(args):
    kind = args.get('kind') or 'formulation'
    batch_g = float(args.get('batch_g') or 0)
    if batch_g <= 0:
        raise ValueError('Enter the target batch mass in grams (batch_g).')
    if kind not in ('formulation', 'thermoset', 'molecule'):
        raise ValueError('kind must be formulation, thermoset or molecule.')
    stock = _stock()
    if kind == 'formulation':
        body = _plan_formulation(args.get('formulation') or {}, batch_g, stock)
    elif kind == 'thermoset':
        body = _plan_thermoset(args.get('thermoset') or {}, batch_g, stock)
    else:
        m = args.get('molecule') or {}
        body = _plan_molecule(m.get('query') or '', batch_g, stock, check_vendors=bool(m.get('check_vendors')))

    spec = {k: args.get(k) for k in ('formulation', 'thermoset', 'molecule') if args.get(k)}
    recipe = json.dumps({'concoct': VERSION, 'kind': kind, 'batch_g': batch_g,
                         'spec': json.dumps(spec, sort_keys=True)}, sort_keys=True)
    body['recipe_id'] = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    body['batch_g'] = batch_g
    body['limits'] = _LIMITS
    body['stock_summary'] = {'shared_rows': len(stock['shared']), 'science_bundle_rows': len(stock['local'])}
    return body


def _stock_op(args):
    from engineering_sim.science import paths
    stock = _stock()
    return {'shared': stock['shared'], 'science_bundle': stock['local'],
            'contract': {
                'shared': 'Set ENGINEERING_SIM_INVENTORY to a JSON file with a "chemicals" section: {id: {label, qty, unit, note}}.',
                'science_bundle': 'Maintain %s: rows {id: {label, qty, unit, note}}.' % (paths.data_dir() / 'stock.json')}}


OPS = {
    'concoct_plan': _plan,
    'concoct_stock': _stock_op,
}
