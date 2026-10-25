# Third-party notices and licensing boundaries

The [PolyForm license](LICENSE) and any separately negotiated commercial agreement cover only rights held by this project's licensor. They do not relicense dependencies, third-party reference data, publications, public-domain material or upstream code. Preserve upstream attribution, copyright and license notices when applicable.

## Scope of this review

Runtime libraries, native solvers and their binaries are installed separately. Their license terms and distribution conditions apply to the versions and combinations you use. The NIMS database notice and scientific source citations below must be retained. This notice is not a complete transitive dependency or commercial redistribution clearance.

The table records the observed direct dependencies in the validation environment. Its labels summarize upstream metadata; the full license accompanying the exact component/version controls. Transitive libraries, data files, GPU runtimes and compiled wheels can add further terms. [Machine-readable review](evidence/dependency-license-review.json).

| Component | Observed version | Upstream licensing summary | Primary project/source |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (package metadata) | [Upstream](https://numpy.org) |
| scipy | 1.18.1 | BSD-3-Clause; bundled component terms also apply | [Upstream](https://scipy.org/) |
| pint | 0.26.2 | BSD-3-Clause | [Upstream](https://github.com/hgrecco/pint) |
| sympy | 1.14.0 | BSD-3-Clause | [Upstream](https://github.com/sympy/sympy) |
| warp-lang | 1.18.0 | Apache-2.0 | [Upstream](https://developer.nvidia.com/warp-python) |
| rdkit | 2026.9.1 | BSD-3-Clause | [Upstream](http://rdkit.org/) |
| biopython | 1.88 | Biopython License Agreement; component-specific terms | [Upstream](https://biopython.org/) |
| primer3-py | 2.4.0 | GPL-2.0-or-later | [Upstream](https://github.com/libnano/primer3-py) |
| CoolProp | 8.0.0 | MIT | [Upstream](http://www.coolprop.org) |
| cantera | 3.2.0 | BSD-3-Clause | [Upstream](https://cantera.org) |
| iapws | 1.5.5 | GPL v3 (package metadata) | [Upstream](https://github.com/jjgomera/iapws) |
| pyscf | 2.14.0 | Apache-2.0 | [Upstream](http://www.pyscf.org) |
| tblite | 0.7.0 | LGPL-3.0-or-later | [Upstream](https://github.com/tblite/tblite) |
| ase | 3.29.0 | LGPL-2.1-or-later | [Upstream](https://ase-lib.org/) |
| openmm | 8.6.1 | Python wrapper: PSF-style; native library and bundled components have separate terms | [Upstream](https://openmm.org) |
| pdbfixer | 1.12.0 | MIT | [Upstream](https://github.com/openmm/pdbfixer) |
| gemmi | 0.7.5 | MPL-2.0 | [Upstream](https://github.com/project-gemmi/gemmi) |
| meeko | 0.8.0 | LGPL-2.1 in license field; classifier says LGPLv2+; consult installed license | [Upstream](https://github.com/forlilab/meeko) |
| vina | 1.2.7 | Apache-2.0 | [Upstream](https://ccsb.scripps.edu/) |
| pymatgen | 2026.10.24 | MIT | [Upstream](https://pymatgen.org) |
| pycalphad | 0.11.2 | MIT | [Upstream](https://pycalphad.org/) |
| ngsolve | 6.2.2607 | LGPL-2.1-only | [Upstream](https://github.com/NGSolve/ngsolve/blob/master/LICENSE) |
| netgen-mesher | 6.2.2607 | LGPL-2.1-only | [Upstream](https://github.com/NGSolve/netgen/blob/master/LICENSE) |
| scikit-fem | 12.0.2 | BSD license (package classifier) | [Upstream](https://github.com/kinnala/scikit-fem) |
| gmsh | 4.15.2 | GPLv2+; upstream commercial licensing option | [Upstream](https://gmsh.info) |
| sgp4 | 2.27 | MIT | [Upstream](https://github.com/brandon-rhodes/python-sgp4) |
| pymsis | 0.13.0 | MIT for Python package; underlying MSIS model terms also apply | [Upstream](https://swxtrec.github.io/pymsis/) |
| matplotlib | 3.11.2 | Matplotlib/PSF-style license; bundled component terms also apply | [Upstream](https://matplotlib.org) |
| PyYAML | 6.0.3 | MIT | [Upstream](https://github.com/yaml/pyyaml/issues) |
| pytest | 9.1.1 | MIT | [Upstream](https://docs.pytest.org/en/stable/changelog.html) |
| build | 1.6.1 | MIT | [Upstream](https://build.pypa.io/en/stable/changelog.html) |

External native tools also retain their own terms: [FreeCAD](https://github.com/FreeCAD/FreeCAD/blob/main/LICENSE) uses LGPL licensing with additional component licenses; [SU2](https://github.com/su2code/SU2/blob/master/COPYING) supplies LGPL terms. Neither tool is redistributed in this source release. The separately tested Warp 1.9.1 CUDA 12 environment remains subject to [Warp's upstream license](https://github.com/NVIDIA/warp/blob/v1.9.1/LICENSE.md) and applicable runtime-component terms.

## Copyleft and commercial distribution

Optional dependencies include [primer3-py (GPL)](https://github.com/libnano/primer3-py/blob/master/LICENSE), [iapws (GPL)](https://github.com/jjgomera/iapws/blob/master/LICENSE) and [Gmsh (GPL with a separate upstream licensing option)](https://gmsh.info/doc/texinfo/gmsh.html#Copying-conditions). Other components use LGPL or MPL terms. PolyForm's noncommercial restriction is not a replacement for GPL permissions or obligations. Do not assume that importing or installing a GPL dependency separately automatically makes redistribution of a combined program compatible. A redistribution involving GPL code may require a GPL-compatible overall license or separate permission from its copyright holders; a paid license from this project alone cannot provide that. LGPL/MPL components also retain their applicable linking, modification, notice and source obligations.

This repository does not impose a noncommercial fee or restriction on independent use of upstream software obtained under its own license. It does not grant permission to redistribute a combined proprietary/noncommercial binary bundle under PolyForm alone.

## Scientific reference data and citations

- `science/data/thermo_species.json` includes selected thermochemical values and Shomate coefficients with record-level [NIST Chemistry WebBook, SRD 69](https://webbook.nist.gov/chemistry/) citations. The WebBook states: © 2026 by the U.S. Secretary of Commerce on behalf of the United States of America. All rights reserved. NIST distinguishes copyrighted Standard Reference Data from other government works in its [SRD and data policy](https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications) and [SRD copyright notice](https://www.nist.gov/srd/public-law). These cited source materials are excluded from this project's relicensing claim. No blanket redistribution or commercial data license from NIST is supplied here; retain citations and obtain any permissions required for the intended reuse.
- `science/data/biochem_pka.json`, other property tables and benchmark references preserve source names, dates and confidence labels. [NCBI policy](https://www.ncbi.nlm.nih.gov/home/about/policies/) specifically cautions that PubChem and other resources may include protected third-party material. A public database URL is not itself a license grant. The project licenses only its own expression/arrangement where it holds those rights, not upstream publications or proprietary supplier material.
- `science/data/tdb/alzn_mey.json` is metadata only. Its retained notice identifies **Copyright (C) NIMS 2009**, with TDB authors T. Abe, K. Hashimoto and Y. Sawada, National Institute for Materials Science, and the Mey (1993) reassessment. The referenced `.tdb` payload is **not included**. Availability in an upstream test suite is not a new license grant from this project.
- Equations, ideas, physical facts and public-domain material are not claimed as proprietary inventions through this license. Standards, papers, supplier names and trademarks remain attributable to their respective sources. No institutional endorsement is implied.

## License-text provenance

The root `LICENSE` is an unmodified copy of the [official PolyForm Noncommercial 1.0.0 text](https://polyformproject.org/licenses/noncommercial/1.0.0.txt). The [PolyForm source repository](https://github.com/polyformproject/polyform-licenses) permits use of its license texts; its naming conditions for modified texts do not apply because no license clauses were changed. The SHA-256 of the downloaded text is recorded in the dependency review. Project attribution is supplied separately in [NOTICE](NOTICE).
