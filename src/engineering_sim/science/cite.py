# DESCRIPTION: Value/citation records shared by every science toolkit module (standard library only).
"""Every reported quantity is a small dict so provenance travels with the number.

kind:
  cited      taken from a named source; `source` is a URL or bibliographic reference,
             `retrieved` the date it was read, `verified` True only when the value was
             read from the fetched source text during this build (not recalled)
  estimated  produced by a model (Joback, GFN2-xTB, mean-field gel point, ...); `method` names it
  derived    computed exactly from other records (stoichiometry, unit conversion); `method` names it
  user       entered by the user
  unverified a recalled typical value whose source could not be fetched and checked
"""

KINDS = ('cited', 'estimated', 'derived', 'user', 'unverified')


def today():
    """UTC date (YYYY-MM-DD) for a citation's `retrieved` field."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def value(v, unit, kind, source=None, retrieved=None, verified=False, method=None, note=None, uncertainty=None):
    """Build one value record; raises on an unknown kind or a cited value without a source."""
    if kind not in KINDS:
        raise ValueError('Unknown value kind: ' + str(kind))
    if kind == 'cited' and not source:
        raise ValueError('A cited value needs a source.')
    if kind in ('estimated', 'derived') and not method:
        raise ValueError(kind + ' values need a method.')
    rec = dict(value=v, unit=unit, kind=kind)
    # Optional fields are omitted when empty so records stay compact and hash-stable.
    for k, x in (('source', source), ('retrieved', retrieved), ('method', method), ('note', note), ('uncertainty', uncertainty)):
        if x is not None:
            rec[k] = x
    if kind == 'cited':
        rec['verified'] = bool(verified)
    return rec


def num(rec):
    """Plain number from a value record (or pass a bare number through)."""
    return rec['value'] if isinstance(rec, dict) else rec


def is_record(x):
    return isinstance(x, dict) and 'value' in x and 'kind' in x and x['kind'] in KINDS
