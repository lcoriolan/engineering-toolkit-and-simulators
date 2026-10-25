# Package contents and verification

The package provides engineering algorithms, scientific workers, CAD operations, reference data, synthetic examples, tests and benchmark reports. Supply the geometry, materials, dimensions and operating conditions required by each API. See [coverage](coverage.md) for implementation and validation limits and [installation](dependencies.md) for optional runtime dependencies.

`PUBLIC_FILES.json` lists package paths, byte sizes and SHA-256 hashes. Run `python tools/check_release.py` on a clean checkout to verify the manifest. The checker also rejects unexpected files, symlinks and unsupported artifact types.

Repository-owned material is source-available under [PolyForm Noncommercial 1.0.0](../LICENSE). Personal hobby use and the license's other expressly permitted uses are free; commercial uses outside that grant require a separately negotiated paid written agreement. Preserve [NOTICE](../NOTICE), [licensing scope](../LICENSE_STATUS.md) and [third-party exceptions](../THIRD_PARTY_NOTICES.md) in distributions. A project commercial agreement does not change upstream terms or grant rights to caller-supplied data.
