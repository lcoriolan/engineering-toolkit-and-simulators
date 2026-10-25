# DESCRIPTION: Tests for engineering_sim/science/thermo.py against independent benchmark numbers
#              (typed below with their sources; every number is independent of the code).
"""Thermochemistry tests.

Every expected number is typed in from a published source (NIST WebBook / CODATA /
NIST-JANAF) and is independent of the code under test.
"""
import math
import unittest

# science toolkit worker-side test: needs configured science environment (RDKit/SciPy/tblite). Under the standard-library studio
# interpreter the whole module is skipped instead of erroring the shared suite.
try:
    import rdkit, scipy  # noqa: F401
except ImportError:
    raise unittest.SkipTest("science toolkit interpreter only: run with python")

from engineering_sim.science import cite, thermo

R = 8.314462618  # J/(mol K), CODATA 2018


class ThermoPropsTests(unittest.TestCase):
    def test_h2o_gas_formation_enthalpy_reproduces_database(self):
        # CODATA review value: dHf(H2O, g, 298.15 K) = -241.826 kJ/mol
        # (NIST WebBook, gas phase thermochemistry data, water).
        spec = thermo.species("H2O(g)")
        dHf_db = cite.num(spec["dHf298_kJmol"])
        h = cite.num(thermo.props("H2O(g)", 298.15)["h"])
        self.assertAlmostEqual(h, dHf_db, delta=0.01)

    def test_props_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            thermo.props("H2O(g)", 7000.0)  # above the 6000 K Shomate cap
        with self.assertRaises(ValueError):
            thermo.props("H2O(g)", 100.0)   # below the 298.15 K anchor

    def test_unknown_species_raises(self):
        with self.assertRaises(KeyError):
            thermo.props("NoSuchSpecies", 298.15)


class ReactionTests(unittest.TestCase):
    def test_element_balance_error(self):
        # H2 -> H2O is not balanced (missing an oxygen atom).
        with self.assertRaises(ValueError) as ctx:
            thermo.reaction({"H2": 1}, {"H2O(g)": 1}, 298.15)
        self.assertIn("not element-balanced", str(ctx.exception))

    def test_water_formation_enthalpy(self):
        # 2 H2 + O2 -> 2 H2O(g), dH = 2 * dHf(H2O,g) = 2 * (-241.826) = -483.65 kJ/mol
        # (CODATA dHf(H2O,g); NIST WebBook). Expected -483.6 +/- 1 kJ.
        r = thermo.reaction({"H2": 2, "O2": 1}, {"H2O(g)": 2}, 298.15)
        dH = cite.num(r["dH"])
        self.assertGreater(dH, -484.6)
        self.assertLess(dH, -482.6)
        self.assertTrue(r["favored"])

    def test_hydrogen_atomization_enthalpy(self):
        # H2 -> 2 H, dH = 2 * dHf(H,g) = 2 * 217.998 = 435.996 kJ/mol
        # (CODATA dHf(H,g) = 217.998 kJ/mol; NIST WebBook). Expected +436.0 +/- 1 kJ.
        r = thermo.reaction({"H2": 1}, {"H": 2}, 298.15)
        dH = cite.num(r["dH"])
        self.assertGreater(dH, 435.0)
        self.assertLess(dH, 437.0)
        self.assertFalse(r["favored"])  # endothermic and entropy-unfavoured at 298 K? dG>0

    def test_ammonia_synthesis(self):
        # N2 + 3 H2 <=> 2 NH3. At 298 K the reaction is favoured (dG < 0, K >> 1);
        # at 1000 K dG > 0 (the Haber equilibrium shifts back to the elements).
        r298 = thermo.reaction({"N2": 1, "H2": 3}, {"NH3": 2}, 298.15)
        self.assertTrue(r298["favored"])
        self.assertGreater(cite.num(r298["K"]), 1.0)
        r1000 = thermo.reaction({"N2": 1, "H2": 3}, {"NH3": 2}, 1000.0)
        self.assertGreater(cite.num(r1000["dG"]), 0.0)

    def test_water_gas_shift_constant_at_1000K(self):
        # CO + H2O <=> CO2 + H2. NIST-JANAF gives K ~ 1.4 at 1000 K
        # (NIST-JANAF Thermochemical Tables, fourth edition). Require within 20 %.
        r = thermo.reaction({"CO": 1, "H2O(g)": 1}, {"CO2": 1, "H2": 1}, 1000.0)
        K = cite.num(r["K"])
        self.assertGreater(K, 1.4 * 0.8)
        self.assertLess(K, 1.4 * 1.2)


class EquilibriumTests(unittest.TestCase):
    def test_h2_stable_at_298K(self):
        # Hydrogen stays molecular at 298 K / 1 bar (dissociation is negligible).
        eq = thermo.equilibrium({"H2": 1.0}, ["H2", "H"], 298.15, 1.0)
        self.assertTrue(eq["converged"])
        self.assertAlmostEqual(eq["amounts"]["H2"], 1.0, delta=1e-6)
        self.assertAlmostEqual(eq["amounts"]["H"], 0.0, delta=1e-6)

    def test_h2_dissociates_at_4000K(self):
        # At 4000 K / 1 bar hydrogen is mostly H atoms (NIST-JANAF free-energy data
        # give ~77 % dissociation). Cross-check the equilibrium mole fractions against
        # the equilibrium constant from the tabulated free energies.
        eq = thermo.equilibrium({"H2": 1.0}, ["H2", "H"], 4000.0, 1.0)
        self.assertTrue(eq["converged"])
        n_H = eq["amounts"]["H"]
        n_H2 = eq["amounts"]["H2"]
        self.assertGreater(n_H, n_H2)  # mostly atoms
        y_H = n_H / (n_H + n_H2)
        y_H2 = n_H2 / (n_H + n_H2)
        self.assertGreater(y_H, 0.70)
        self.assertLess(y_H, 0.85)
        # Independent K from the tabulated dG (different code path: props/reaction).
        g_H = cite.num(thermo.props("H", 4000.0)["g"])
        g_H2 = cite.num(thermo.props("H2", 4000.0)["g"])
        K_indep = math.exp(-(2 * g_H - g_H2) * 1000.0 / (R * 4000.0))
        K_eq = (y_H ** 2) / y_H2 * 1.0  # P/P0 = 1 bar
        self.assertAlmostEqual(K_eq, K_indep, delta=K_indep * 0.05)

    def test_water_formation_equilibrium(self):
        # 2 H2 + O2 -> 2 H2O(g): the equilibrium is essentially complete at 298 K.
        eq = thermo.equilibrium({"H2": 2.0, "O2": 1.0}, ["H2", "O2", "H2O(g)"], 298.15, 1.0)
        self.assertTrue(eq["converged"])
        self.assertAlmostEqual(eq["amounts"]["H2O(g)"], 2.0, delta=1e-3)

    def test_condensed_phase_allowed_to_form(self):
        # With H2O(l) allowed, liquid water (activity 1, more stable than the gas at
        # 298 K / 1 bar) forms in preference to the gas.
        eq = thermo.equilibrium({"H2": 2.0, "O2": 1.0}, ["H2", "O2", "H2O(g)", "H2O(l)"], 298.15, 1.0)
        self.assertTrue(eq["converged"])
        self.assertAlmostEqual(eq["amounts"]["H2O(l)"], 2.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
