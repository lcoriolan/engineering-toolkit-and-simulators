# DESCRIPTION: Real control-state transitions and rejection of ungranted numerical results.
"""No graphics dependency: lifecycle protocol tests complement real WebGL parity."""
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
import unittest
from engineering_sim import execution
def engine(): return execution


class BrowserControlTests(unittest.TestCase):
    def setUp(self):
        self.controller=engine().BrowserWorkers()

    def ready(self, operation='flow.lbm2d'):
        grant=self.controller.request(dict(action='create',operation=operation,cells=512))
        self.controller.request(dict(action='complete',id=grant['id'],token=grant['permit']['token'],result=dict(finite=True,steps=0)))
        return grant['id']

    def test_initialization_needs_a_permit_and_matching_result(self):
        row=self.controller.request(dict(action='create',operation='flow.lbm3d',cells=1024))
        self.assertEqual(row['state'],'initializing')
        self.assertIsNone(self.controller.request(dict(action='grant',id=row['id'],steps=1))['permit'])
        with self.assertRaisesRegex(ValueError,'step count'):
            self.controller.request(dict(action='complete',id=row['id'],token=row['permit']['token'],result=dict(finite=True,steps=1)))

    def test_pause_resume_stop_and_late_result(self):
        identifier=self.ready()
        self.controller.request(dict(action='pause',id=identifier))
        self.assertIsNone(self.controller.request(dict(action='grant',id=identifier,steps=3))['permit'])
        self.controller.request(dict(action='resume',id=identifier))
        grant=self.controller.request(dict(action='grant',id=identifier,steps=3))
        self.controller.request(dict(action='stop',id=identifier))
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=dict(finite=True,steps=3)))
        self.assertIsNone(self.controller.request(dict(action='grant',id=identifier,steps=3))['permit'])

    def test_only_one_concurrent_chunk_is_granted(self):
        identifier=self.ready()
        def attempt(_):
            try:return self.controller.request(dict(action='grant',id=identifier,steps=1))['permit']
            except ValueError:return None
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(attempt,range(4)))
        self.assertEqual(sum(row is not None for row in rows),1)

    def test_results_are_received_without_design_payloads(self):
        identifier=self.ready();grant=self.controller.request(dict(action='grant',id=identifier,steps=4))
        result=dict(finite=True,steps=4,min_density=1.0,max_density=1.0)
        reply=self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=result))
        self.assertEqual(reply['result'],result)
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=result))
        grant=self.controller.request(dict(action='grant',id=identifier,steps=1))
        with self.assertRaisesRegex(ValueError,'bounded numerical'):
            self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=dict(result,geometry=[1,2,3])))

    def test_numerical_failure_stops_future_grants(self):
        identifier=self.ready();grant=self.controller.request(dict(action='grant',id=identifier,steps=1))
        row=self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=dict(finite=False,steps=1)))
        self.assertEqual(row['state'],'failed')
        self.assertIsNone(self.controller.request(dict(action='grant',id=identifier,steps=1))['permit'])

    def test_explicit_worker_error_is_retained(self):
        identifier=self.ready()
        row=self.controller.request(dict(action='fail',id=identifier,error_type='TypeError'))
        self.assertEqual((row['state'],row['error_type']),('failed','TypeError'))

    def test_resource_and_scalar_budgets(self):
        for cells in (0,-1,4194305,True,1.5):
            with self.assertRaises(ValueError):self.controller.request(dict(action='create',operation='flow.lbm2d',cells=cells))
        identifier=self.ready()
        for count in (0,65,True,1.5):
            with self.assertRaises(ValueError):self.controller.request(dict(action='grant',id=identifier,steps=count))
        grant=self.controller.request(dict(action='grant',id=identifier,steps=1))
        with self.assertRaisesRegex(ValueError,'finite scalar'):
            self.controller.request(dict(action='complete',id=identifier,token=grant['permit']['token'],result=dict(finite=True,steps=1,max_mach=float('nan'))))

    def test_server_boundary_really_enters_shared_engine(self):
        with engine().capture() as records:
            row=engine().browser_request(dict(action='create',operation='flow.lbm2d',cells=4))
        self.assertEqual(records[0]['operation'],'browser.worker.control')
        self.assertEqual(records[0]['state'],'complete')
        engine().browser_request(dict(action='stop',id=row['id']))


if __name__=='__main__':unittest.main()
