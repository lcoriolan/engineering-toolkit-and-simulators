"""Caller-owned runtime data stays opt-in; exported synthetic work remains portable."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_ancestor_inventory_is_not_read(tmp_path, monkeypatch):
    from engineering_sim.science import ops_concoct, paths
    installed = tmp_path / 'installed' / 'engineering_sim'
    installed.mkdir(parents=True)
    legacy = tmp_path / '_inventory'
    legacy.mkdir()
    (legacy / 'inventory.json').write_text(json.dumps({'chemicals': {'synthetic_sentinel': {'qty': 1}}}))
    monkeypatch.setattr(paths, 'ROOT', installed)
    monkeypatch.setenv('ENGINEERING_SIM_DATA', str(tmp_path / 'science'))
    monkeypatch.delenv('ENGINEERING_SIM_INVENTORY', raising=False)
    assert ops_concoct._stock()['shared'] == {}


def test_explicit_inventory_is_read(tmp_path, monkeypatch):
    from engineering_sim.science import ops_concoct
    inventory = tmp_path / 'caller-inventory.json'
    row = {'synthetic_water': {'label': 'water', 'qty': 2, 'unit': 'g'}}
    inventory.write_text(json.dumps({'chemicals': row}))
    monkeypatch.setenv('ENGINEERING_SIM_INVENTORY', str(inventory))
    monkeypatch.setenv('ENGINEERING_SIM_DATA', str(tmp_path / 'science'))
    assert ops_concoct._stock()['shared'] == row


def test_local_stock_stays_caller_controlled(tmp_path, monkeypatch):
    from engineering_sim.science import ops_concoct, paths
    monkeypatch.delenv('ENGINEERING_SIM_INVENTORY', raising=False)
    monkeypatch.setenv('ENGINEERING_SIM_DATA', str(tmp_path / 'science'))
    row = {'synthetic_water': {'label': 'water', 'qty': 3, 'unit': 'g'}}
    (paths.data_dir() / 'stock.json').write_text(json.dumps({'rows': row}))
    assert ops_concoct._stock()['local'] == row


def test_retrosynthesis_paths_are_explicit(tmp_path):
    data = tmp_path / 'retro'
    env = dict(os.environ, ENGINEERING_SIM_RETRO_DATA=str(data),
               ENGINEERING_SIM_RETRO_PYTHON=sys.executable, ENGINEERING_SIM_DATA=str(tmp_path / 'science'))
    code = 'from engineering_sim.science import stability as s; import json; print(json.dumps([str(s.AZ_DIR),str(s.AZ_PY)]))'
    result = subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [str(data), sys.executable]


def test_synthetic_offline_export_survives_isolated_replay(tmp_path, monkeypatch):
    from engineering_sim.science import chemcalc
    monkeypatch.setenv('ENGINEERING_SIM_DATA', str(tmp_path / 'science'))
    result = chemcalc.run({'start': 'H2O', 'k': 3, 'elements': ['H', 'O'],
                          'max_heavy_atoms': 2, 'known_check': None}, replay=True)
    exported = chemcalc.export(result['recipe']['recipe_id'])
    with zipfile.ZipFile(exported['zip_path']) as archive:
        assert 'science_bundle/thermo.py' in archive.namelist()
    assert chemcalc.export(result['recipe']['recipe_id'])['sha256'] == exported['sha256']
