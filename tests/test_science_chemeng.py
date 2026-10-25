# DESCRIPTION: Tests for the Chemical Engineering (chemeng) area: Cantera kinetics,
#              CoolProp properties and the Rachford-Rice / reactor-sizing balances,
#              against independent benchmark numbers typed below with their sources.
"""Chemical Engineering (chemeng) tests.

Every expected number is typed in from a published source (NIST WebBook / GRI-Mech 3.0
literature) and is independent of the code under test. The registry test round-trips
an op through the real worker subprocess.
"""
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# science toolkit worker-side test: needs configured science environment (RDKit/SciPy/Cantera/CoolProp). Under the
# standard-library studio interpreter the whole module is skipped instead of erroring.
try:
    import rdkit, scipy, cantera, CoolProp  # noqa: F401
except ImportError:
    raise unittest.SkipTest("science toolkit interpreter only: run with python")

from engineering_sim.science import api, chemeng_balance as balance
from engineering_sim.science import chemeng_kinetics as kinetics
from engineering_sim.science import chemeng_props as props
from engineering_sim.science import cite


def _num(rec):
    return cite.num(rec)


# Independent benchmarks ----------------------------------------------------------
# NIST Chemistry WebBook, water, "Isothermal Properties" page (fetched 2026-09-15):
#   https://webbook.nist.gov/cgi/fluid.cgi?ID=C7732185&Action=Load&Type=IsoTherm&T=298.15&TUnit=K&PUnit=kPa&DUnit=kg%2Fm3&HUnit=J%2Fmol&WUnit=m%2Fs&VisUnit=uPa*s&STUnit=N%2Fm&RefState=&PLow=101.325&PHigh=101.325&PInc=1&Digits=5
# The fetched page states, for water:
#   "Isothermal Data for T = 298.15 K ... 101.33 kPa -> Density 997.05 kg/m3"
#   "Normal boiling point 373.1243 K"
# These are the two independent CoolProp cross-checks below.

# GRI-Mech 3.0 (Smith et al., http://combustion.berkeley.edu/gri-mech/version30/text30.html)
# literature: stoichiometric CH4/air adiabatic flame temperature ~2225 K (range 2220-2240 K)
# and laminar flame speed ~36-40 cm/s at 300 K, 1 atm.


class FluidPropertyTests(unittest.TestCase):
    def test_water_saturation_temperature(self):
        # NIST WebBook normal boiling point 373.1243 K; require 373.12 +/- 0.1 K.
        Tsat = _num(props.saturation('Water', P_Pa=101325.0)['T_sat'])
        self.assertAlmostEqual(Tsat, 373.12, delta=0.1)

    def test_water_liquid_density(self):
        # NIST WebBook 997.05 kg/m3 at 298.15 K, 101.325 kPa; require 997.0 +/- 0.5.
        rho = _num(props.state('Water', {'T_K': 298.15, 'P_Pa': 101325.0})['rho_kg_m3'])
        self.assertAlmostEqual(rho, 997.0, delta=0.5)

    def test_state_reports_phase(self):
        st = props.state('Water', {'T_K': 298.15, 'P_Pa': 101325.0})
        self.assertEqual(st['phase'], 'liquid')
        self.assertEqual(st['quality'], 0.0)

    def test_phase_envelope_reaches_critical_point(self):
        env = props.phase_envelope('Water')
        self.assertLessEqual(len(env['points']), 200)
        self.assertAlmostEqual(_num(env['T_crit']), 647.096, delta=0.2)
        self.assertAlmostEqual(env['points'][-1][0], _num(env['T_crit']), delta=0.5)

    def test_bubble_dew_methane_ethane(self):
        # HEOS mixture: 50/50 methane/ethane at 1 atm bubbles well below either pure
        # boiling point (methane ~111.7 K, ethane ~184.6 K); require a sane ordering.
        r = props.mixture_bubble_dew(['Methane', 'Ethane'], [0.5, 0.5], 101325.0)
        self.assertLess(_num(r['T_bubble']), _num(r['T_dew']))


class KineticsTests(unittest.TestCase):
    def test_stoichiometric_ch4_air_adiabatic_flame_temperature(self):
        # GRI-Mech 3.0 adiabatic flame temperature, CH4 + 2(O2 + 3.76 N2) at 300 K, 1 atm:
        # 2220-2240 K (literature; GRI-Mech 3.0 predicts ~2225 K).
        r = kinetics.equilibrate('gri30', {'CH4': 1, 'O2': 2, 'N2': 7.52}, 300.0, 1.0, 'HP')
        Tad = _num(r['T'])
        self.assertGreater(Tad, 2220.0)
        self.assertLess(Tad, 2240.0)

    def test_stoichiometric_ch4_air_flame_speed(self):
        # GRI-Mech 3.0 freely-propagating flame, CH4/air phi=1, 300 K, 1 atm:
        # literature range 36-40 cm/s.
        r = kinetics.flame_speed('gri30', 'CH4', 'O2:1, N2:3.76', 1.0, 300.0, 1.0)
        SL = _num(r['S_L'])
        self.assertGreater(SL, 36.0)
        self.assertLess(SL, 40.0)

    def test_h2_o2_frozen_at_300K_for_an_hour(self):
        # 2 H2 + O2 at 300 K, 1 bar: kinetically frozen, so no 50 % equilibrium
        # conversion within an hour (happens == False).
        r = kinetics.bench_kinetics(['2 H2', 'O2'], 300.0, 1.0, 3600.0)
        self.assertTrue(r['covered'])
        self.assertFalse(r['happens'])

    def test_h2_o2_ignites_below_1s_at_1100K(self):
        # The same mixture at 1100 K ignites in milliseconds (ignition delay < 1 s).
        r = kinetics.bench_kinetics(['2 H2', 'O2'], 1100.0, 1.0, 1.0)
        self.assertTrue(r['covered'])
        self.assertTrue(r['happens'])
        self.assertLess(_num(r['time_to_50pct']), 1.0)

    def test_bench_reports_uncovered_species(self):
        # n-octane is not in GRI-Mech 3.0; the result must say so, not guess.
        r = kinetics.bench_kinetics(['C8H18', 'O2'], 300.0, 1.0, 1.0)
        self.assertFalse(r['covered'])
        self.assertIn('C8H18', r['missing'])

    def test_mechanisms_list(self):
        mechs = {m['name']: m for m in kinetics.mechanisms()}
        self.assertIn('gri30', mechs)
        self.assertIn('h2o2', mechs)
        self.assertEqual(mechs['gri30']['species'], 53)
        self.assertEqual(mechs['gri30']['reactions'], 325)
        self.assertIn('http://combustion.berkeley.edu/gri-mech/version30/text30.html',
                      mechs['gri30']['reference'])

    def test_rate_screen_returns_top_ten(self):
        r = kinetics.rate_screen({'CH4': 1, 'O2': 2, 'N2': 7.52}, 1400.0, 1.0)
        self.assertEqual(len(r['reactions']), 10)


class BalanceTests(unittest.TestCase):
    def test_rachford_rice_three_component(self):
        # Rachford-Rice equation (Seader, Henley & Roper, "Separation Process
        # Principles", 3rd ed., Wiley 2011, Eq. 4.12): for z = [0.4, 0.3, 0.3] with
        # K = [3.0, 1.0, 0.2] the equation has the exact root V = 0.5
        # (x = [0.2, 0.3, 0.5], y = [0.6, 0.3, 0.1], sum = 0).
        r = balance.flash_rachford_rice({'A': 0.4, 'B': 0.3, 'C': 0.3},
                                        K={'A': 3.0, 'B': 1.0, 'C': 0.2})
        self.assertAlmostEqual(_num(r['V']), 0.5, delta=1e-9)
        self.assertEqual(r['phase'], 'two-phase')
        self.assertAlmostEqual(_num(r['x']['A']), 0.2, delta=1e-9)
        self.assertAlmostEqual(_num(r['y']['A']), 0.6, delta=1e-9)

    def test_cstr_pfr_first_order_ratio(self):
        # First-order design equations: V_CSTR/V_PFR = [X/(1-X)] / [-ln(1-X)] exactly.
        X = 0.9
        c = balance.reactor_sizing('CSTR', k=0.1, order=1, X=X, F_A0=1.0, C_A0=100.0)
        p = balance.reactor_sizing('PFR', k=0.1, order=1, X=X, F_A0=1.0, C_A0=100.0)
        ratio = _num(c['V']) / _num(p['V'])
        self.assertAlmostEqual(ratio, (X / (1 - X)) / (-math.log(1 - X)), delta=1e-9)

    def test_stream_mix_constant_cp(self):
        # 1 kg/s at 350 K + 2 kg/s at 290 K (same cp) -> (1*350 + 2*290)/3 = 310 K.
        r = balance.stream_mix([
            {'name': 'hot', 'm_dot': 1.0, 'T_K': 350.0, 'cp': 4180.0},
            {'name': 'cold', 'm_dot': 2.0, 'T_K': 290.0, 'cp': 4180.0}])
        self.assertAlmostEqual(_num(r['T_mix']), 310.0, delta=1e-9)

    def test_lmtd_counter_flow(self):
        # counter: dT1 = 100-50 = 50, dT2 = 60-30 = 30 -> LMTD = 20/ln(50/30).
        r = balance.lmtd(100.0, 60.0, 30.0, 50.0, flow='counter')
        self.assertAlmostEqual(_num(r['LMTD']), 20.0 / math.log(50.0 / 30.0), delta=1e-9)


class RegistryTests(unittest.TestCase):
    """Ops are discoverable and round-trip through the real worker subprocess."""

    OPS = ('chemeng_mechanisms', 'chemeng_ignition', 'chemeng_equilibrate', 'chemeng_flame',
           'chemeng_rates', 'chemeng_bench', 'chemeng_fluids', 'chemeng_state',
           'chemeng_saturation', 'chemeng_envelope', 'chemeng_bubble_dew', 'chemeng_mix',
           'chemeng_flash', 'chemeng_reactor', 'chemeng_lmtd')

    def test_extension_timeouts_contains_all_ops(self):
        timeouts = api.extension_timeouts()
        for op in self.OPS:
            self.assertIn(op, timeouts)
            self.assertLessEqual(timeouts[op], 900)

    def test_state_round_trip_through_worker(self):
        if api.interpreter() is None:
            self.skipTest('science toolkit interpreter (configured science environment) is missing.')
        with tempfile.TemporaryDirectory() as tmp:
            result = api.handle('chemeng_state',
                                {'fluid': 'Water', 'inputs': {'T_K': 298.15, 'P_Pa': 101325.0}},
                                Path(tmp))
        self.assertIn('rho_kg_m3', result)
        self.assertAlmostEqual(_num(result['rho_kg_m3']), 997.0, delta=0.5)

    def test_equilibrate_round_trip_through_worker(self):
        if api.interpreter() is None:
            self.skipTest('science toolkit interpreter (configured science environment) is missing.')
        with tempfile.TemporaryDirectory() as tmp:
            result = api.handle('chemeng_equilibrate',
                                {'mech': 'gri30', 'mixture': {'CH4': 1, 'O2': 2, 'N2': 7.52},
                                 'T': 300.0, 'P': 1.0, 'constraint': 'HP'}, Path(tmp))
        self.assertGreater(_num(result['T']), 2220.0)
        self.assertLess(_num(result['T']), 2240.0)




if __name__ == '__main__':
    unittest.main()
