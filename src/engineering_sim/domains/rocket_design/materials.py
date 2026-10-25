#!/usr/bin/env python3
# DESCRIPTION: Bulk density [kg/m^3], in-plane shear modulus G [Pa] (fin flutter), and surface roughness [m] by
#              finish. Every value carries its provenance. Override any entry from a design spec
#              (materials: {name: {rho: .., G: ..}}) when a measured value exists (MEASURED beats everything here).
#              OpenRocket = OpenRocket 23.09 built-in component database (core/src/main/resources/datafiles/presets
#              and the Material defaults in net.sf.openrocket.database.Databases), the de-facto hobby reference.

MATERIALS = {
    # name: rho [kg/m3], G [Pa] in-plane shear (None = not used for fins), provenance
    "fiberglass":   dict(rho=1850.0, G=2.93e9, src="rho: OpenRocket 'Fiberglass' 1850; G: 425 kpsi typical G10, "
                                                    "Apogee Peak of Flight #291 (DATASHEET-typical, ASSUMED for a given layup)"),
    "g10":          dict(rho=1850.0, G=2.93e9, src="as fiberglass (G10/FR4 laminate, MatWeb 1.80-1.90 g/cc)"),
    "carbon_fiber": dict(rho=1780.0, G=5.0e9, src="rho: OpenRocket 'Carbon fiber' 1780; G: 0/90 woven laminate "
                                                   "~4-6 GPa (ASSUMED 5 GPa; quasi-isotropic would be ~15-19 GPa)"),
    "cardboard":    dict(rho=680.0, G=None, src="OpenRocket 'Cardboard' 680"),
    "phenolic":     dict(rho=950.0, G=None, src="OpenRocket 'Phenolic' 950"),
    "blue_tube":    dict(rho=1300.0, G=None, src="ASSUMED 1300 (vulcanised-fibre tube; no OpenRocket default)"),
    "aluminum":     dict(rho=2700.0, G=26.0e9, src="6061-T6, ASM Aerospace Specification Metals: 2.70 g/cc, G 26 GPa"),
    "plywood":      dict(rho=630.0, G=0.6e9, src="rho: OpenRocket 'Plywood (birch)' 630; G: USDA Wood Handbook "
                                                  "FPL-GTR-282 order of magnitude 0.5-0.7 GPa (ASSUMED 0.6)"),
    "balsa":        dict(rho=170.0, G=0.15e9, src="rho: OpenRocket 'Balsa' 170; G ASSUMED (USDA Wood Handbook range)"),
    "pla":          dict(rho=1240.0, G=1.2e9, src="NatureWorks Ingeo datasheet 1.24 g/cc; G = E/2(1+nu) with E 3.5 GPa, "
                                                   "nu 0.36 (bulk, not printed-part; ASSUMED)"),
    "nylon":        dict(rho=1140.0, G=None, src="PA6 bulk, typical datasheet 1.13-1.15 g/cc"),
}
SURFACE = {   # areal density [kg/m2] of recovery fabrics, linear density [kg/m] of cords
    "ripstop_nylon": dict(areal=0.067, src="OpenRocket 'Ripstop nylon' 67 g/m2"),
    "shock_cord":    dict(linear=0.013, src="OpenRocket 'Tubular nylon (11 mm, 7/16 in)' 13 g/m"),
}
FINISH_ROUGHNESS_M = {   # OpenRocket finish table (Niskanen 2013, Table 3.2, after Barrowman 1967)
    "polished": 2e-6, "smooth_paint": 20e-6, "regular_paint": 60e-6, "unfinished": 150e-6, "rough": 500e-6,
}


def material(name, overrides=None):
    m = dict(MATERIALS[name.lower()])
    if overrides and name in overrides:
        m.update(overrides[name], src="spec override (%s)" % overrides[name].get("src", "MEASURED?"))
    return m
