# DESCRIPTION: Tests for the Biochemistry area (biochem_nucleic / biochem_enzyme / biochem_buffers /
#              ops_biochem) against independent benchmark numbers typed below with their sources.
"""Biochemistry area tests.

Every expected number is typed in from a published source or a hand-checked construction and is
independent of the code under test. The module is skipped under the standard-library studio
interpreter (the science imports Biopython/primer3/scipy only in configured science environment).
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# science toolkit worker-side test: needs configured science environment (Biopython/primer3/scipy/numpy). Under the
# standard-library studio interpreter the whole module is skipped instead of erroring the suite.
try:
    import Bio, primer3, scipy, numpy  # noqa: F401
except ImportError:
    raise unittest.SkipTest("science toolkit interpreter only: run with python")

from engineering_sim.science import api, cite
from engineering_sim.science import biochem_buffers, biochem_enzyme, biochem_nucleic

ROOT = Path(__file__).resolve().parents[1]


class NucleicAcidTests(unittest.TestCase):
    def test_m13_forward_primer_gc_and_primer3_tm(self):
        # M13(-20) forward primer GTAAAACGACGGCCAGT: 9 of 17 bases are G/C -> 52.9 % GC.
        # primer3's calc_tm at its defaults returns 54.7 C (within the 54-56 C design band).
        a = biochem_nucleic.analyze("GTAAAACGACGGCCAGT")
        self.assertAlmostEqual(cite.num(a["gc_percent"]), 52.9, delta=0.1)
        self.assertGreaterEqual(cite.num(a["tm_primer3"]), 54.0)
        self.assertLessEqual(cite.num(a["tm_primer3"]), 56.0)

    def test_translation_and_reverse_complement(self):
        # ATG TTT TAA -> Met Phe Stop = "MF*" (standard code, NCBI table 1).
        # Reverse complement of ATGTTTTAA is TTAAAACAT (hand complement of the reverse).
        a = biochem_nucleic.analyze("ATGTTTTAA")
        self.assertEqual(a["reverse_complement"], "TTAAAACAT")
        frame1 = next(f for f in a["frames"] if f["strand"] == 1 and f["frame"] == 1)
        self.assertEqual(frame1["aa"], "MF*")

    def test_ecori_two_sites_three_fragments(self):
        # A 1 kb sequence with GAATTC at positions 200 and 506. Bio.Restriction reports the EcoRI
        # cuts at 202 and 508, so the three fragments are 202, 306 (=508-202) and 492 (=1000-508) bp.
        seq = "A" * 200 + "GAATTC" + "A" * 300 + "GAATTC" + "A" * 488
        r = biochem_nucleic.restriction(seq, enzymes=["EcoRI"])
        site = r["sites"][0]
        self.assertEqual(site["cuts"], [202, 508])
        self.assertEqual(sorted(site["fragments"], reverse=True), [492, 306, 202])

    def test_analyze_rejects_invalid_characters(self):
        with self.assertRaises(ValueError):
            biochem_nucleic.analyze("ATGXCC")


class EnzymeKineticsTests(unittest.TestCase):
    def test_mm_fit_recovers_true_parameters(self):
        # Noise-free Michaelis-Menten data: v = 10*S/(2.5+S). The nonlinear fit must recover
        # Vmax = 10 and Km = 2.5 to within 1 %.
        import numpy as np
        S = np.linspace(0.1, 20.0, 40).tolist()
        v = (10.0 * np.asarray(S) / (2.5 + np.asarray(S))).tolist()
        r = biochem_enzyme.fit_michaelis_menten(S, v, model="mm")
        self.assertAlmostEqual(cite.num(r["params"]["Vmax"]), 10.0, delta=0.1)
        self.assertAlmostEqual(cite.num(r["params"]["Km"]), 2.5, delta=0.025)

    def test_competitive_inhibition_selected_by_aic(self):
        # Synthetic competitive-inhibition data (Ki = 5) at two inhibitor levels. The competitive
        # model must be the AIC winner against mm/uncompetitive/noncompetitive (and hill/substrate).
        import numpy as np
        S = np.concatenate([np.linspace(1.0, 40.0, 12), np.linspace(1.0, 40.0, 12)]).tolist()
        I = [0.0] * 12 + [10.0] * 12
        v = (10.0 * np.asarray(S) / (2.5 * (1.0 + np.asarray(I) / 5.0) + np.asarray(S))).tolist()
        fits = biochem_enzyme.compare_models(S, v, I=I)
        self.assertEqual(fits[0]["model"], "competitive")
        self.assertAlmostEqual(cite.num(fits[0]["params"]["Ki"]), 5.0, delta=0.25)


class BufferTests(unittest.TestCase):
    def _pka2(self):
        with open(Path(api.__file__).parent / "data" / "biochem_pka.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        pka2 = next(p["pka"] for p in data["buffers"]["phosphate"]["pkas"] if p["label"] == "pKa2")
        return cite.num(pka2)

    def test_phosphate_hh_ratio_uses_data_pka(self):
        # At I = 0 (no activity correction) the HH ratio at pH 7.20 is 10^(7.20 - pKa2), with pKa2
        # taken from the data file (7.199). This is the classical equimolar phosphate recipe.
        pka2 = self._pka2()
        hh = biochem_buffers.henderson_hasselbalch("phosphate", pH=7.20, I=0.0)
        self.assertAlmostEqual(cite.num(hh["ratio"]), 10 ** (7.20 - pka2), places=4)

    def test_davies_lowers_apparent_pka_at_0_1M(self):
        # The Davies correction lowers the apparent pKa2 of phosphate at I = 0.1 M (the HPO4(2-)/
        # H2PO4(-) pair: the divalent base is stabilised more than the monovalent acid).
        pka2 = self._pka2()
        hh = biochem_buffers.henderson_hasselbalch("phosphate", pH=7.20, I=0.1)
        self.assertLess(cite.num(hh["pka_apparent"]), pka2)

    def test_glycine_pi(self):
        # Glycine pI = (pKa1 + pKa2)/2 = (2.34 + 9.60)/2 = 5.97 (from the data-file values).
        pi = biochem_buffers.pI([{"pka": 2.34, "type": "acid"}, {"pka": 9.60, "type": "base"}])
        self.assertAlmostEqual(cite.num(pi["pI"]), (2.34 + 9.60) / 2.0, delta=0.01)

    def test_atp_dg_prime_at_q1_equals_dg0(self):
        # dG' = dG'0 + RT ln Q; at Q = 1 the log term vanishes, so dG' equals dG'0 exactly.
        dg0 = biochem_buffers.dG0prime("atp_hydrolysis")
        r = biochem_buffers.biochemical_dG(dg0, 1.0)
        self.assertEqual(cite.num(r["dg_kjmol"]), cite.num(dg0))


class RegistryAndPageTests(unittest.TestCase):
    def test_extension_timeouts_contains_biochem_ops(self):
        timeouts = api.extension_timeouts()
        expected = {"biochem_nucleic", "biochem_primers", "biochem_restriction", "biochem_align",
                    "biochem_mutate", "biochem_fit", "biochem_simulate", "biochem_efficiency",
                    "biochem_eyring", "biochem_buffers", "biochem_hh", "biochem_prepare_buffer",
                    "biochem_titration", "biochem_ionic_strength", "biochem_pi", "biochem_dg"}
        self.assertTrue(expected.issubset(timeouts.keys()),
                        "missing: %s" % (expected - set(timeouts)))

    def test_ops_module_declares_consistent_ops(self):
        from engineering_sim.science import ops_biochem
        self.assertEqual(set(ops_biochem.OPS.keys()), set(ops_biochem.TIMEOUTS.keys()))
        for op in ops_biochem.OPS:
            self.assertTrue(op.startswith("biochem_"), op)
            self.assertLessEqual(ops_biochem.TIMEOUTS[op], 900, op)

    def test_biochem_nucleic_roundtrip_through_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = api.handle("biochem_nucleic", {"seq": "ATGGCC"}, Path(tmp))
        self.assertEqual(cite.num(result["length"]), 6)
        self.assertEqual(result["kind"], "dna")



if __name__ == "__main__":
    unittest.main(verbosity=2)
