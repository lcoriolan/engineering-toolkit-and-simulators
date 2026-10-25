# DESCRIPTION: Check dispatch, failure ownership, nested context, concurrency, and privacy.
"""Standard-library tests; no CAD, GPU, physical input, or project recipe is used."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
from engineering_sim import execution as engine


class ExecutionTests(unittest.TestCase):
    """The dispatcher executes numerical callbacks without changing their contract."""
    def test_actual_dispatch_and_return_identity(self):
        calls = []
        result = object()
        @engine.operation('test.execution.identity')
        def solve(value):
            calls.append((value, engine.current_execution()['operation']))
            return result
        with engine.capture() as records:
            self.assertIs(solve(3), result)
            self.assertIs(engine.ENGINE.call('test.execution.identity', 4), result)
        self.assertEqual(calls, [(3,'test.execution.identity'),(4,'test.execution.identity')])
        self.assertEqual([r['state'] for r in records], ['complete','complete'])
        self.assertIsNone(engine.current_execution())

    def test_unknown_operation_never_imports_or_executes(self):
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            engine.ENGINE.call('module.not_registered')

    def test_exact_exception_is_preserved(self):
        failure = ValueError('private input must not enter metadata')
        @engine.operation('test.execution.failure')
        def solve():
            raise failure
        with engine.capture() as records:
            with self.assertRaises(ValueError) as caught:
                solve()
        self.assertIs(caught.exception, failure)
        self.assertEqual(records[0]['state'], 'failed')
        self.assertEqual(records[0]['error_type'], 'ValueError')
        self.assertNotIn('private input', repr(records))
        self.assertIsNone(engine.current_execution())

    def test_existing_async_failure_contract(self):
        @engine.operation('test.execution.job', kind='orchestration')
        def job():
            engine.report_failure(RuntimeError('solver process failed'))
            return {'state':'failed'}
        with engine.capture() as records:
            self.assertEqual(job(), {'state':'failed'})
        self.assertEqual(records[0]['state'], 'failed')

    def test_nested_calls_have_one_parent_chain(self):
        @engine.operation('test.execution.inner')
        def inner():
            return 7
        @engine.operation('test.execution.outer', kind='study')
        def outer():
            return inner()
        with engine.capture() as records:
            self.assertEqual(outer(),7)
        child, parent = records
        self.assertEqual(child['parent_run_id'],parent['run_id'])
        self.assertIsNone(parent['parent_run_id'])

    def test_thread_contexts_do_not_cross(self):
        @engine.operation('test.execution.thread')
        def solve(value):
            return value * value
        def worker(value):
            with engine.capture() as records:
                result=solve(value)
            return result,records
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(worker,range(12)))
        self.assertEqual([r for r,_ in rows],[n*n for n in range(12)])
        ids=[]
        for _,records in rows:
            self.assertEqual(len(records),1)
            self.assertIsNone(records[0]['parent_run_id'])
            ids.append(records[0]['run_id'])
        self.assertEqual(len(set(ids)),12)

    def test_records_exclude_inputs_outputs_and_paths(self):
        secret={'a_private_dimension':12345,'source_path':'/private/example'}
        @engine.operation('test.execution.no_payload')
        def solve(value):
            return value
        with engine.capture() as records:
            self.assertIs(solve(secret),secret)
        for forbidden in ['a_private_dimension','12345','/private/example','callback','args','kwargs']:
            self.assertNotIn(forbidden,repr(records))

    def test_bound_revision_is_not_replaced_by_reload(self):
        @engine.operation('test.execution.reload')
        def before():
            return 'before'
        @engine.operation('test.execution.reload')
        def after():
            return 'after'
        self.assertEqual(before(),'before')
        self.assertEqual(after(),'after')
        self.assertEqual(engine.ENGINE.call('test.execution.reload'),'after')


if __name__=='__main__':
    unittest.main()
