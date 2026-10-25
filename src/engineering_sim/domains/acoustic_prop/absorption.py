import math
AIR_TEMP_C=20.0
AIR_RH_PCT=50.0
AIR_PRESSURE_KPA=101.325
P_REF = 101.325

T0 = 293.15

T01 = 273.16

def absorption_db_per_km(f, temp_c=None, rh=None, pressure_kpa=None):
    """ISO 9613-1 atmospheric absorption, dB/km; inputs Hz, Celsius, percent RH, kPa."""
    temp_c = AIR_TEMP_C if temp_c is None else temp_c        # None = the active site's air (core)
    rh = AIR_RH_PCT if rh is None else rh
    pressure_kpa = AIR_PRESSURE_KPA if pressure_kpa is None else pressure_kpa
    T = temp_c + 273.15
    pa = pressure_kpa
    # saturation vapour pressure ratio psat/pref, then molar water-vapour concentration h [%]
    psat_ratio = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = rh * psat_ratio * (P_REF / pa)
    # relaxation frequencies of oxygen (frO) and nitrogen (frN)
    frO = (pa / P_REF) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / P_REF) * (T / T0) ** (-0.5) * (9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    # absorption [dB/m] = 8.686 * (classical + O2 + N2 relaxation terms), then -> dB/km
    a = 8.686 * f * f * (
        1.84e-11 * (P_REF / pa) * (T / T0) ** 0.5
        + (T / T0) ** (-2.5) * (
            0.01275 * math.exp(-2239.1 / T) / (frO + f * f / frO)
            + 0.1068 * math.exp(-3352.0 / T) / (frN + f * f / frN)
        )
    )
    return a * 1000.0
