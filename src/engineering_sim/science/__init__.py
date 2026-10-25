# DESCRIPTION: science toolkit chemistry package: molecules, cited records, My Lab reaction bench, thermochemistry, formulations, thermosets, volatility and documents.
"""science toolkit is ENGToolkit's chemistry page.

Two interpreters:
  engineering_sim/science/api.py      standard library; imported by the calling application; never imports RDKit/numpy
  engineering_sim/science/worker.py   runs under configured science environment (RDKit, numpy, scipy, tblite); one JSON request per process

Every number that leaves science toolkit is a engineering_sim.science.cite value record (cited / estimated / derived / user).
See proof/science_bundle_01/briefs/00_common.md for the module contract.
"""
