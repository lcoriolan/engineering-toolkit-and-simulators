# DESCRIPTION: Task A: filesystem paths for science toolkit runtime data (standard library only).
"""Where science toolkit keeps its runtime data (cached PubChem records, etc.).

Standard library only: this module is imported both by engineering_sim/science/api.py
(which runs under the studio's stdlib interpreter) and by the worker's RDKit
modules under configured science environment.
"""
import json
import os
import uuid
from pathlib import Path

# Use the installed interpreter and caller-selectable data directory.
import sys as _sys
ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = Path(os.environ.get('ENGINEERING_SIM_PYTHON') or _sys.executable)


def data_dir() -> Path:
    """Return the science toolkit data directory, creating it on demand.

    Overridable with the ENGINEERING_SIM_DATA environment variable (tests point it at a
    temp directory so they never touch real runtime state). Defaults to
    <current directory>/.engineering-sim/science.
    """
    p = Path(os.environ.get('ENGINEERING_SIM_DATA') or (Path.cwd() / '.engineering-sim' / 'science'))
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_cache(path, result) -> None:
    """Best-effort JSON cache write (sorted keys); atomic replace so a concurrent reader never sees half a file.

    A failed write is swallowed: a cache miss next time is acceptable, breaking the lookup is not."""
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(result, sort_keys=True))
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
