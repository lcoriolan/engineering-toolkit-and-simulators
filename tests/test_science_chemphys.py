# DESCRIPTION: Task G: tests for the Chemical Physics area against independent benchmark
#              numbers (typed below with their sources; every number is independent of the code).
"""Chemical Physics tests.

Every expected number is typed in from a published source (Szabo & Ostlund,
NIST CCCBDB, NIST WebBook, CODATA) and is independent of the code under test.
"""
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

# science toolkit worker-side test: needs configured science environment (RDKit/SciPy/PySCF). Under the standard-library
# studio interpreter the whole module is skipped instead of erroring the shared suite.
try:
    import rdkit, scipy, pyscf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("science toolkit interpreter only: run with python")

from engineering_sim.science import api, cite, chemphys_qm, chemphys_spectra, chemphys_statmech, thermo



class ScfBenchmarkTests(unittest.TestCase):
    def test_h2_rhf_sto3g_energy(self):
        # Szabo & Ostlund, "Modern Quantum Chemistry", H2 RHF/STO-3G at R = 1.4 bohr
        # (0.7408 Angstrom): E = -1.1167 hartree (the tabulated value at 1.4 bohr).
        r = chemphys_qm.run_scf([{'el': 'H', 'x': 0.0, 'y': 0.0, 'z': 0.0},
                                 {'el': 'H', 'x': 0.0, 'y': 0.0, 'z': 0.74}], 0, 1, 'RHF', 'sto-3g')
        self.assertTrue(r['converged'])
        self.assertAlmostEqual(cite.num(r['energy_hartree']), -1.1167, delta=0.001)

    def test_h2o_b3lyp_dipole_and_gap(self):
        # Water B3LYP/6-31G*: dipole moment 1.9-2.3 D (experimental gas-phase value is
        # 1.855 D; the calculated value is slightly higher), HOMO-LUMO gap positive.
        r = chemphys_qm.scf('water', method='B3LYP', basis='6-31g*')
        self.assertTrue(r['converged'])
        self.assertGreater(cite.num(r['dipole']), 1.9)
        self.assertLess(cite.num(r['dipole']), 2.3)
        self.assertGreater(cite.num(r['gap_eV']), 0.0)

    def test_size_guard_refuses_oversize_molecule(self):
        # A 40-carbon chain exceeds the 30 heavy-atom guard and must be refused before SCF.
        with self.assertRaises(ValueError) as ctx:
            chemphys_qm.scf('C' * 40, method='RHF', basis='sto-3g')
        self.assertIn('heavy-atom', str(ctx.exception))


class SpectraBenchmarkTests(unittest.TestCase):
    def test_h2o_scaled_harmonic_frequencies(self):
        # Experimental gas-phase fundamentals of H2O: 1595, 3657, 3756 cm^-1
        # (NIST CCCBDB / Shimanouchi's tables). B3LYP/6-31G* frequencies scaled by the
        # 0.9613 factor must be within 5 % of each.
        ir = chemphys_spectra.ir('water', method='B3LYP', basis='6-31g*', intensities=False)
        freqs = sorted(p['frequency']['value'] for p in ir['frequencies'])
        self.assertEqual(len(freqs), 3)
        for got, exp in zip(freqs, [1595.0, 3657.0, 3756.0]):
            self.assertLess(abs(got - exp) / exp, 0.05,
                            'mode %.1f vs experimental %.1f cm^-1' % (got, exp))

    def test_co_rotational_constant(self):
        # CO rotational constant B_e = 1.9313 cm^-1 (NIST WebBook, constants of diatomic
        # molecules, X1Sigma+). Computed from the RDKit geometry it must be within 2 %.
        rot = chemphys_spectra.rotational('[C-]#[O+]')
        b = cite.num(rot['rotational_constants_cm1']['B'])
        self.assertLess(abs(b - 1.9313) / 1.9313, 0.02)


class StatmechBenchmarkTests(unittest.TestCase):
    def test_n2_rrho_entropy_matches_nist(self):
        # NIST (Shomate, via thermo.props) gives S(N2, g, 298.15 K) = 191.61 J/mol*K.
        # The RRHO value from the cited NIST WebBook diatomic constants must be within
        # 1 J/mol*K (the harmonic/anharmonic offset is small for N2 at 298 K).
        cmp = chemphys_statmech.compare_nist('N2', 298.15)
        nist = cite.num(thermo.props('N2', 298.15)['s'])
        self.assertAlmostEqual(nist, 191.61, delta=0.2)
        self.assertLess(abs(cite.num(cmp['delta_s'])), 1.0)

    def test_particle_in_box_n2_scaling(self):
        # Particle in a box: E_n scales exactly as n^2 (energies are returned rounded to
        # 6 decimal places, hence the 1e-3 tolerance rather than an exact-integer check).
        pib = chemphys_statmech.particle_in_box(n_max=4, L_nm=1.0)
        e = [cite.num(l['energy_eV']) for l in pib['levels']]
        for n in (2, 3, 4):
            self.assertAlmostEqual(e[n - 1] / e[0], n * n, delta=1e-3)

    def test_hydrogen_atom_levels(self):
        # Hydrogen levels: E_n = -Rydberg / n^2 with Rydberg = 13.6057 eV (CODATA).
        ha = chemphys_statmech.hydrogen_atom(n_max=3)
        e = [cite.num(l['energy_eV']) for l in ha['levels']]
        self.assertAlmostEqual(e[0], -13.6057, delta=1e-3)
        self.assertAlmostEqual(e[0] / 4.0, e[1], delta=1e-6)
        self.assertAlmostEqual(e[0] / 9.0, e[2], delta=1e-6)


class RegistryTests(unittest.TestCase):
    def test_extension_timeouts_contains_chemphys_ops(self):
        timeouts = api.extension_timeouts()
        for op in ('chemphys_scf', 'chemphys_orbital', 'chemphys_ir', 'chemphys_uvvis',
                   'chemphys_rotational', 'chemphys_statmech', 'chemphys_compare_nist',
                   'chemphys_boltzmann', 'chemphys_models', 'chemphys_constants'):
            self.assertIn(op, timeouts)
        self.assertLessEqual(max(timeouts.values()), 900)

    def test_constants_roundtrip_through_worker(self):
        # End-to-end: the constants op round-trips through the real worker subprocess.
        with tempfile.TemporaryDirectory() as tmp:
            result = api.handle('chemphys_constants', {}, Path(tmp))
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 8)
        rec = result[0]['record']
        self.assertIn('value', rec)
        self.assertIn('source', rec)
        self.assertTrue(rec['verified'])



if __name__ == '__main__':
    unittest.main()
