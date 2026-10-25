# DESCRIPTION: Markdown documents for science toolkit: a compounding sheet (feed order, screw zones, drying, temperatures, vent, QC hold points) and a QC plan (ASTM/ISO methods with action limits). Standard library only.
"""Markdown documents for a screened formulation.

compounding_sheet(screen_result) -- turns a compound.screen() result into a
    compounding run sheet (drying, feed order, screw zones, temperatures, vent,
    residence, QC hold points), using the resin's library drying/window records.
qc_plan(formulation) -- incoming and release test plan with ASTM/ISO method numbers
    and action limits per resin family, reusing the rPET QC methods (IV, COOH, DSC,
    rheometry, ISO 527 / ISO 75).

Both are deterministic text; the numbers are either the caller's cite records or the
library records, never new measurements.
"""
from . import compound


def _num(x):
    from .cite import num
    return num(x)


def _resin_record(resin_id):
    for r in compound.library('resins'):
        if r['id'] == resin_id:
            return r
    raise KeyError('unknown resin id: ' + str(resin_id))


def _fmt(record, digits=0):
    """Render a cite record's value with a unit, tolerating None values."""
    if record is None:
        return '-'
    v = _num(record)
    if v is None:
        return '-'
    s = ('%.*f' % (digits, v)) if isinstance(v, float) else str(v)
    unit = record.get('unit') if isinstance(record, dict) else ''
    return (s + ' ' + unit).strip() if unit and unit != '-' else s


# Resin-family rules: (needs_drying, feed order note, screw zones, vent, mould).
def _family(resin):
    cls = resin.get('class', '')
    if cls in ('polyester', 'polyamide', 'polycarbonate'):
        return dict(drying=True,
                    feed='resin + powder additives (AO, lubricant) at the main feed; glass/filler via side feeder into the melt (zone 5-6); volatile-laden or reactive additives via downstream feed ports',
                    zones=['Z1 feed (resin + powders)', 'Z2-Z3 melting/compression', 'Z4 side-feeder (filler)',
                           'Z5-Z6 distributive mixing', 'Z7 atmospheric vent', 'Z8-Z9 metering + melt seal'],
                    vent='atmospheric (or %s mbar per process)', mould='mould temperature from process')
    if cls in ('polyolefin', 'styrenic', 'acrylic'):
        return dict(drying=False,
                    feed='all components at the main feed; glass via side feeder when used',
                    zones=['Z1 feed (all)', 'Z2-Z3 melting', 'Z4 mixing', 'Z5 vent (if foaming/by-products)', 'Z6-Z7 metering'],
                    vent='vent only if a blowing agent or moisture is present', mould='mould temperature from process')
    if cls in ('polyarylene sulfide', 'polyaryletherketone'):
        return dict(drying=True,
                    feed='resin + powders at the main feed; carbon fibre via side feeder; high-temperature barrel',
                    zones=['Z1 feed (resin + powders)', 'Z2-Z4 high-temperature melting', 'Z5 side-feeder (fibre)',
                           'Z6 mixing', 'Z7 metering'],
                    vent='no vent below 340 C (or vacuum vent for volatiles)', mould='hot mould (crystallisation)')
    if cls == 'polyoxymethylene':
        return dict(drying=False,
                    feed='all at the main feed; keep below 230 C (formaldehyde evolution)',
                    zones=['Z1 feed', 'Z2-Z3 melting (max 220 C)', 'Z4 mixing', 'Z5 metering'],
                    vent='no vent (formaldehyde handling); short residence', mould='mould from process')
    return dict(drying=False, feed='all at the main feed', zones=['Z1 feed', 'Z2 melt', 'Z3 metering'],
                vent='none', mould='mould from process')


def compounding_sheet(screen_result):
    """Markdown run sheet from a compound.screen() result."""
    resin = _resin_record(screen_result['resin']['id'])
    fam = _family(resin)
    proc = screen_result.get('process', {})
    mb = screen_result.get('mass_balance', {})
    hdt = screen_result.get('hdt', {})
    findings = screen_result.get('findings', [])
    fillers = screen_result.get('composite', {}).get('fillers', [])

    lines = ['# Compounding sheet — %s' % resin['name'], '']
    lines.append('## Formulation')
    lines.append('| Component | Loading (wt%) |')
    lines.append('| --- | --- |')
    lines.append('| %s (resin) | %s |' % (resin['name'], _fmt(mb.get('resin_wt_pct'), 1)))
    for c in screen_result.get('components', []):
        lines.append('| %s | %.2f |' % (c['name'], c['wt']))
    for f in screen_result.get('fillers', []):
        lines.append('| %s (filler) | %.2f |' % (f['name'], f['wt']))
    lines.append('')

    lines.append('## Drying (%s)' % ('required' if fam['drying'] else 'not required'))
    if fam['drying']:
        lines.append('- %s at %s for %s, dew point %s, target moisture below %s.'
                    % (resin['name'], _fmt(resin['drying_temp_c']), _fmt(resin['drying_time_h']),
                       _fmt(resin['drying_dewpoint_c']), _fmt(resin['drying_max_moisture_pct'], 3)))
    else:
        lines.append('- %s: no hydrolysable bonds; dry only if condensate is a concern.' % resin['name'])
    lines.append('')

    lines.append('## Temperatures, vent, residence')
    lines.append('- Melt window: %s to %s.' % (_fmt(resin['melt_temp_min_c']), _fmt(resin['melt_temp_max_c'])))
    lines.append('- Melt temperature: %s.' % _fmt(proc.get('melt_temp_c'), 0))
    lines.append('- Mould temperature: %s.' % _fmt(proc.get('mold_temp_c'), 0))
    lines.append('- Vent: %s.' % (fam['vent'] % proc.get('vent_mbar', 100.0)))
    lines.append('- Residence time: %s s.' % _fmt(proc.get('residence_s'), 0))
    lines.append('')

    lines.append('## Feed order and screw zones')
    lines.append('- ' + fam['feed'])
    for z in fam['zones']:
        lines.append('- ' + z)
    lines.append('')

    lines.append('## HDT mechanism')
    lines.append('- %s (ceiling %s).' % (hdt.get('mechanism', {}).get('value', '-'), _fmt(hdt.get('ceiling_c'), 0)))
    lines.append('')

    lines.append('## QC hold points')
    lines.append('- Incoming moisture, IV and end groups before drying (see qc_plan).')
    lines.append('- Melt IV / melt-flow index at the die; hold the lot if below the action limit.')
    lines.append('- Filler content (ash) and tensile/HDT on the first shot of each lot.')
    if findings:
        lines.append('- Screen findings to disposition before release:')
        for f in findings:
            lines.append('  - ' + f)
    lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def qc_plan(formulation):
    """Markdown incoming + release QC plan with ASTM/ISO methods and action limits."""
    resin = _resin_record(formulation['resin'])
    cls = resin.get('class', '')
    components = formulation.get('components', [])
    condensation = cls in ('polyester', 'polyamide')
    has_extender = any(_num(compound._by_id('additives', c['id'])['functionality']) is not None
                       and _num(compound._by_id('additives', c['id'])['functionality']) > 2.0
                       for c in components) if components else False
    lines = ['# QC plan — %s' % resin['name'], '']

    lines.append('## Incoming (per lot)')
    lines.append('| Test | Method | Action limit |')
    lines.append('| --- | --- | --- |')
    lines.append('| Moisture | ASTM D6869 (Karl Fischer) | below %s before drying' % _fmt(resin['drying_max_moisture_pct'], 3))
    if condensation:
        lines.append('| Intrinsic viscosity | ASTM D4603 | IV >= 0.55 dL/g (embrittlement screen)')
        lines.append('| Carboxyl end groups | ASTM D7409 | COOH within the feedstock specification')
    lines.append('| Comonomer / regrind ratio | DSC (ASTM D3418) or NMR | confirm incoming grade')
    lines.append('| Melt flow index | ISO 1133 | within resin grade specification')
    lines.append('')

    lines.append('## Release (per production lot)')
    lines.append('| Test | Method | Action limit |')
    lines.append('| --- | --- | --- |')
    lines.append('| Tensile strength / modulus | ISO 527-2 | >= %s MPa target' % formulation.get('targets', {}).get('tensile_mpa', 0.0))
    lines.append('| Flexural modulus / HDT | ISO 75-2 (1.8 MPa) | >= %s C target' % formulation.get('targets', {}).get('hdt_c', 0.0))
    lines.append('| Filler content (ash) | ISO 3451 | within +/- 1 wt% of the nominal loading')
    lines.append('| Density | ISO 1183 | within grade specification')
    lines.append('| Impact (notched Charpy) | ISO 179 | >= grade minimum (report)')
    if condensation:
        lines.append('| Melt viscosity / IV | rheometry (plateau G′) | no gel plateau; IV retained')
    if has_extender:
        lines.append('| Chain-extender response | rheometry ladder (G′ vs dose) | start <= 0.3 wt%; treat higher doses as gel-risk')
    lines.append('')

    lines.append('## Notes')
    lines.append('- The rPET QC methods carry over: IV (ASTM D4603), COOH (ASTM D7409), DSC, rheometry, ISO 527 and ISO 75.')
    lines.append('- These are action limits, not acceptance criteria: set final limits from a capability study on production equipment.')
    lines.append('')
    return '\n'.join(lines).rstrip() + '\n'
