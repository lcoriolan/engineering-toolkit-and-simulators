# DESCRIPTION: Stability ladder rung 5 — AiZynthFinder retrosynthesis (public USPTO expansion/filter models +


import json
import sys
import time


def tree_of(node):
    """Compact route tree: {smiles, in_stock, children:[...]} from an AiZynth route dict."""
    out = {'smiles': node.get('smiles'), 'in_stock': bool(node.get('in_stock'))}
    kids = []
    for ch in node.get('children', []) or []:
        if ch.get('type') == 'reaction':
            meta = ch.get('metadata', {}) or {}
            rx = {'reaction': True, 'template_code': meta.get('template_code'),
                  'policy_probability': meta.get('policy_probability'), 'children': []}
            for g in ch.get('children', []) or []:
                rx['children'].append(tree_of(g))
            kids.append(rx)
    if kids:
        out['children'] = kids
    return out


def leaves(t, acc):
    if not t.get('children'):
        acc.append({'smiles': t['smiles'], 'in_stock': t['in_stock']})
    for c in t.get('children', []):
        for g in c.get('children', []):
            leaves(g, acc)
    return acc


def depth(t):
    if not t.get('children'):
        return 0
    return 1 + max(depth(g) for c in t['children'] for g in c['children'])


def main():
    smiles, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    res = {'tool': 'AiZynthFinder', 'target': smiles}
    try:
        import aizynthfinder
        from aizynthfinder.aizynthfinder import AiZynthFinder
        res['version'] = getattr(aizynthfinder, '__version__', None)
        if res['version'] is None:
            import importlib.metadata as md
            res['version'] = md.version('aizynthfinder')
        f = AiZynthFinder(configfile=cfg)
        f.stock.select('zinc')
        f.expansion_policy.select('uspto')
        try:
            f.filter_policy.select('uspto')
        except Exception:
            pass
        f.config.search.time_limit = 120
        f.config.search.iteration_limit = 100
        f.target_smiles = smiles
        res['target_in_stock'] = bool(f.stock.__contains__(f.target_mol)) if hasattr(f, 'target_mol') else None
        f.tree_search()
        f.build_routes()
        stats = f.extract_statistics()
        res['solved'] = bool(stats.get('is_solved'))
        res['search_time_s'] = round(time.time() - t0, 1)
        res['statistics'] = {k: stats.get(k) for k in ('is_solved', 'number_of_routes', 'number_of_solved_routes',
                                                       'top_score', 'number_of_steps', 'number_of_precursors',
                                                       'number_of_precursors_in_stock', 'precursors_in_stock',
                                                       'precursors_not_in_stock', 'search_time')}
        if res['target_in_stock'] is None:
            res['target_in_stock'] = bool(stats.get('number_of_steps') == 0 and stats.get('is_solved'))
        routes = []
        for d in f.routes.dicts[:3]:
            t = tree_of(d)
            routes.append({'tree': t, 'steps': depth(t), 'starting_materials': leaves(t, [])})
        res['routes'] = routes
        res['steps'] = routes[0]['steps'] if routes else None
        res['settings'] = {'stock': 'zinc', 'expansion': 'uspto', 'filter': 'uspto', 'time_limit_s': 120,
                           'iteration_limit': 100}
    except Exception as ex:  # report, never crash the ladder
        import traceback
        res['error'] = '%s: %s' % (type(ex).__name__, ex)
        res['trace'] = traceback.format_exc()[-2000:]
    with open(out, 'w') as fh:
        json.dump(res, fh, default=str)


if __name__ == '__main__':
    main()
