# DESCRIPTION: Dependency-free dispatch for the world and trusted scientific solver adapters.
"""One execution boundary; numerical methods remain owned by domain solvers.

This module does not replace a solver's time step, units, random generator, device,
or convergence rules. It owns operation registration, nested execution context,
and completion/failure accounting. No parameters, results, exception messages, or
personal paths are retained in execution records. An observer is opt-in and local
to its thread/async context; ordinary calculations have no filesystem side effects.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import re
import threading
import time
import uuid

API_VERSION = 1
VERSION = "1.0.0"
KINDS = frozenset({"static", "transient", "study", "orchestration"})
_active = ContextVar("unified_sim_execution", default=None)
_observers = ContextVar("unified_sim_observers", default=())


@dataclass(frozen=True)
class Operation:
    """A trusted Python callback and its explicit numerical-contract category."""
    name: str
    kind: str
    callback: object
    source_sha256: str


def _source_identity(callback):
    """Fingerprint available Python source without storing its contents or path."""
    source = inspect.getsourcefile(callback)
    if source and Path(source).is_file():
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    # Synthetic/test callbacks may have no source file; absence is explicit.
    return "unavailable"


class Engine:
    """Dispatch registered domain operations while preserving return values/errors.

    Registration is synchronized, but the lock is never held during numerical work.
    Independent solvers therefore keep their existing parallelism. Re-loading a
    trusted adapter updates the catalogue; an already-bound wrapper retains its
    original callable so a reload cannot silently substitute a different solver.
    """
    def __init__(self):
        self._operations = {}
        self._counts = {}
        self._lock = threading.RLock()

    def register(self, name, callback, kind="static"):
        """Register a code-selected callable; names never resolve arbitrary imports."""
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,159}", name):
            raise ValueError("Invalid simulation operation name.")
        if kind not in KINDS or not callable(callback):
            raise ValueError("A simulation operation needs a supported kind and callable.")
        op = Operation(name, kind, callback, _source_identity(callback))
        with self._lock:
            self._operations[name] = op
        return op

    def catalog(self):
        """Return source identities and contracts without exposing input designs."""
        with self._lock:
            return [dict(operation=o.name, kind=o.kind, source_sha256=o.source_sha256)
                    for _, o in sorted(self._operations.items())]

    def statistics(self):
        """Return bounded per-operation execution counts, including worker threads."""
        with self._lock:
            return {name: dict(counts) for name, counts in sorted(self._counts.items())}

    def call(self, name, *args, **kwargs):
        """Dispatch by an already registered identifier; unknown IDs fail closed."""
        with self._lock:
            try:
                op = self._operations[name]
            except KeyError:
                raise ValueError("Unregistered simulation operation: " + str(name)) from None
        return self.execute(op, *args, **kwargs)

    def execute(self, op, *args, **kwargs):
        """Run the actual solver under one context, propagating its exact exception."""
        if not isinstance(op, Operation):
            raise TypeError("Execution requires a registered Operation.")
        parent = _active.get()
        record = dict(engine="unified_sim", engine_version=VERSION, api_version=API_VERSION,
                      run_id=uuid.uuid4().hex, parent_run_id=parent["run_id"] if parent else None,
                      operation=op.name, kind=op.kind, source_sha256=op.source_sha256,
                      state="running")
        token = _active.set(record)
        started = time.perf_counter()
        try:
            result = op.callback(*args, **kwargs)
        except BaseException as exc:
            # Preserve interruption as well as ordinary solver failures. Never
            # turn a numerical failure into a successful-looking result payload.
            record.update(state="failed", error_type=type(exc).__name__)
            raise
        else:
            # Some asynchronous adapters report failure in a job record instead
            # of raising. Their explicit report_failure() must remain a failure.
            if record["state"] == "running":
                record["state"] = "complete"
            return result
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            with self._lock:
                counts = self._counts.setdefault(op.name, dict(complete=0, failed=0))
                counts[record["state"]] += 1
            _active.reset(token)
            for records in _observers.get():
                records.append(dict(record))


ENGINE = Engine()


def operation(name, kind="static"):
    """Bind a public solver entry point to the shared dispatcher, without fallback."""
    def decorate(callback):
        op = ENGINE.register(name, callback, kind)
        @wraps(callback)
        def routed(*args, **kwargs):
            return ENGINE.execute(op, *args, **kwargs)
        routed.simulation_operation = op
        return routed
    return decorate


def current_execution():
    """Return a copy of the current execution identity, or None outside a solver."""
    record = _active.get()
    return dict(record) if record is not None else None


def report_failure(error):
    """Account for a failure already represented by a solver's existing job API."""
    record = _active.get()
    if record is None:
        raise RuntimeError("No simulation execution is active.")
    record.update(state="failed", error_type=type(error).__name__)


@contextmanager
def capture():
    """Collect metadata for actual nested calls in this context; never input data."""
    records = []
    token = _observers.set((*_observers.get(), records))
    try:
        yield records
    finally:
        _observers.reset(token)

# Browser backends keep their numerical kernels and graphics context locally.
# This coordinator owns their lifecycle and bounded execution permits. Results
# are explicitly worker-reported; receiving them is not independent validation.
class BrowserWorkers:
    """Central control of same-origin numerical workers, with one in-flight grant."""
    OPERATIONS = frozenset({'flow.lbm2d', 'flow.lbm3d'})
    RESULT_KEYS = frozenset({'finite','steps','min_density','max_density','max_lattice_speed',
        'max_mach','mean_velocity_departure','mean_transverse_speed','convective_times',
        'fluid_cells','solid_cells'})

    def __init__(self):
        self._sessions = {}
        self._lock = threading.RLock()
        self._ttl = 1800.0

    def _status(self, row):
        return {key: row[key] for key in ('id','operation','state','steps','result','error_type')}

    def _get(self, identifier):
        row = self._sessions.get(identifier)
        if row is None:
            raise ValueError('Unknown or expired browser simulation.')
        row['updated'] = time.monotonic()
        return row

    def _permit(self, row, steps, phase):
        permit = dict(token=uuid.uuid4().hex, steps=steps, phase=phase)
        row['pending'] = permit
        return dict(self._status(row), permit=dict(permit))

    def request(self, payload):
        """Grant work, accept bounded diagnostics, or control a live worker session."""
        import math
        if not isinstance(payload, dict):
            raise ValueError('A browser engine request must be an object.')
        action = payload.get('action')
        with self._lock:
            now = time.monotonic()
            for key, row in list(self._sessions.items()):
                if now-row['updated'] > self._ttl:
                    del self._sessions[key]
            if action == 'create':
                op, cells = payload.get('operation'), payload.get('cells')
                if op not in self.OPERATIONS:
                    raise ValueError('Unknown browser simulation operation.')
                if type(cells) is not int or not 1 <= cells <= 4194304:
                    raise ValueError('Browser simulation cell budget is invalid.')
                if len(self._sessions) >= 128:
                    raise ValueError('Browser engine session limit reached.')
                identifier = uuid.uuid4().hex
                row = dict(id=identifier, operation=op, state='initializing', cells=cells,
                    steps=0, result=None, error_type=None, pending=None, updated=now)
                self._sessions[identifier] = row
                return self._permit(row, 0, 'initialize')
            row = self._get(payload.get('id'))
            if action == 'status':
                return self._status(row)
            if action == 'stop':
                row.update(state='stopped', pending=None)
                return self._status(row)
            if action == 'pause':
                if row['state'] == 'running':
                    row['state'] = 'paused'
                return self._status(row)
            if action == 'resume':
                if row['state'] == 'paused':
                    row['state'] = 'running'
                return self._status(row)
            if action == 'fail':
                if row['state'] != 'stopped':
                    error = payload.get('error_type', 'Error')
                    if not isinstance(error, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]{0,99}', error):
                        error = 'Error'
                    row.update(state='failed', pending=None, error_type=error)
                return self._status(row)
            if action == 'grant':
                n = payload.get('steps')
                if type(n) is not int or not 1 <= n <= 64:
                    raise ValueError('A browser step grant must be between 1 and 64.')
                if row['state'] != 'running':
                    return dict(self._status(row), permit=None)
                if row['pending'] is not None:
                    raise ValueError('A browser execution grant is already outstanding.')
                return self._permit(row, n, 'advance')
            if action in ('complete', 'error'):
                permit = row['pending']
                if permit is None or payload.get('token') != permit['token']:
                    raise ValueError('Stale or unknown browser execution permit.')
                if action == 'error':
                    error = payload.get('error_type', 'Error')
                    if not isinstance(error, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]{0,99}', error):
                        error = 'Error'
                    row.update(state='failed', pending=None, error_type=error)
                    return self._status(row)
                result = payload.get('result')
                if not isinstance(result, dict) or not set(result) <= self.RESULT_KEYS:
                    raise ValueError('Only bounded numerical diagnostics may be reported.')
                if type(result.get('finite')) is not bool or type(result.get('steps')) is not int:
                    raise ValueError('Browser diagnostics need finite and integer step fields.')
                for key, value in result.items():
                    if key != 'finite' and (type(value) not in (int, float) or not math.isfinite(value)):
                        raise ValueError('Browser diagnostics must be finite scalar values.')
                expected = 0 if permit['phase'] == 'initialize' else row['steps']+permit['steps']
                if result['steps'] != expected:
                    raise ValueError('Browser result does not match its granted step count.')
                row.update(result=dict(result), steps=expected, pending=None)
                if not result['finite']:
                    row.update(state='failed', error_type='NonFiniteResult')
                elif row['state'] == 'initializing':
                    row['state'] = 'running'
                return self._status(row)
            raise ValueError('Unknown browser engine action.')


BROWSER_WORKERS = BrowserWorkers()


@operation('browser.worker.control', kind='orchestration')
def browser_request(payload):
    """Use the same engine for server-controlled, browser-computed numerical work."""
    return BROWSER_WORKERS.request(payload)


def source_files():
    """Files for an explicitly requested flat scientific export."""
    from pathlib import Path
    return [Path(__file__).with_name('portable_bridge.py'),Path(__file__).resolve()]
