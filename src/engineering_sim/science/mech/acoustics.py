# DESCRIPTION: Acoustics and noise section: sound levels (pressure/power/intensity), decibel addition,
#              wave speed, transmission loss, noise reduction, reverberation (Sabine), A-weighting and
#              noise-exposure dose.
"""Acoustics and noise calculators.

Levels are in decibels (declared dimensionless, ``signed`` because a pressure below the reference is a
negative level). Reference quantities are the standard ones: p_ref = 20 µPa, W_ref = 1 pW, I_ref = 1 pW/m².
The A-weighting is the IEC 61672 analytic transfer function with the rounded pole frequencies
20.6 / 107.7 / 737.9 / 12194 Hz (stated recalled, not re-read). Relations are cited by name.
The ideal-gas sound speed c = √(γRT) lives in ``compressible.speed_of_sound`` (not duplicated here).
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('acoustics', 'Acoustics and noise', order=450,
                  summary='Sound levels, decibel addition, wave speed, transmission loss, noise reduction, '
                          'reverberation (Sabine), A-weighting and exposure dose.')

# IEC 61672-1 A-weighting pole frequencies (rounded; recalled, not re-read).
_F1, _F2, _F3, _F4 = 20.6, 107.7, 737.9, 12194.0


def _number(x, what):
    """Coerce a JSON scalar to a finite float, or raise MechError (never a bare ValueError)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise MechError(f'{what}: {x!r} is not a finite number')
    return float(x)


def _add_levels(values):
    """Total level of incoherent sources: L_total = 10 log10(sum 10^(L_i/10)) dB."""
    levels = values['levels']
    if not isinstance(levels, list) or not levels:
        raise MechError('levels: give a non-empty list of sound levels in dB')
    total = 10.0 * math.log10(sum(10.0 ** (_number(L, 'levels') / 10.0) for L in levels))
    return {'L_total': total}


def _total_absorption(values):
    """Total absorption A = sum(alpha_i S_i) and the mean absorption coefficient of the room surfaces."""
    surfaces = values['surfaces']
    if not isinstance(surfaces, list) or not surfaces:
        raise MechError('surfaces: give a non-empty list of [alpha (0..1), area (m^2)] rows')
    for row in surfaces:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise MechError('surfaces: each row must be [alpha, area]')
        a = _number(row[0], 'surfaces alpha')
        S = _number(row[1], 'surfaces area')
        if not 0.0 <= a <= 1.0:
            raise MechError('surfaces: absorption coefficient alpha must be in 0..1')
        if S < 0:
            raise MechError('surfaces: areas must be non-negative')
    A = sum(float(a) * float(S) for a, S in surfaces)
    S_tot = sum(float(S) for _, S in surfaces)
    return {'A': A, 'mean_alpha': A / S_tot if S_tot > 0 else 0.0}


def _a_weighting(values):
    """A-weighting A(f) in dB from the IEC 61672 analytic transfer function (rounded poles)."""
    f = values['f']
    if f <= 0:
        raise MechError('f: frequency must be positive')
    r = (_F4 ** 2 * f ** 4) / ((f ** 2 + _F1 ** 2) * (f ** 2 + _F4 ** 2)
                               * math.sqrt(f ** 2 + _F2 ** 2) * math.sqrt(f ** 2 + _F3 ** 2))
    return {'A': 20.0 * math.log10(r) + 2.0}


def _noise_dose(values):
    """OSHA or NIOSH/ISO noise-exposure dose and time-weighted average.

    Reference duration for a level L: T = 8 h / 2^((L - criterion)/exchange_rate). Dose = 100 sum(C/T).
    TWA = criterion + exchange_rate * log2(dose/100). OSHA: 5 dB exchange, 90 dBA criterion;
    NIOSH/ISO: 3 dB exchange, 85 dBA criterion. Every entered level is counted (no threshold is applied).
    """
    standard = values['standard']
    er, crit = (5.0, 90.0) if standard == 'OSHA' else (3.0, 85.0)
    levels = values['levels']
    durations = values['durations']
    if not isinstance(levels, list) or not isinstance(durations, list) or not levels \
            or len(levels) != len(durations):
        raise MechError('levels/durations: give matching non-empty lists of level [dB] and duration [h]')
    for L in levels:
        _number(L, 'levels')  # reject non-numeric level entries
    for C in durations:
        C = _number(C, 'durations')
        if C < 0:
            raise MechError('durations: negative durations are not allowed')
    dose = 0.0
    for L, C in zip(levels, durations):
        T = 8.0 / (2.0 ** ((float(L) - crit) / er))
        dose += float(C) / T
    dose *= 100.0
    if dose <= 0:
        raise MechError('total exposure is zero (dose = 0); TWA is undefined')
    twa = crit + er * math.log2(dose / 100.0)
    return {'exchange_rate': er, 'criterion': crit, 'dose': dose, 'twa': twa}


CALCS = [
    Calc('spl', 'Sound pressure level (re 20 µPa)',
         vars=[V('spl', '', 'sound pressure level', signed=True, label='Lp'),
               V('p', 'Pa', 'RMS sound pressure'), V('p_ref', 'Pa', 'reference pressure (20 µPa)')],
         equations=['spl = 20*log(p/p_ref)/log(10)'], outputs=['spl'],
         example={'p': 0.02, 'p_ref': 20e-6},
         keywords=['dB', 'sound level', 'noise', 'spl'],
         assumptions=['p is the RMS sound pressure; p_ref = 20 µPa is the standard hearing-threshold reference.'],
         refs=['Lp = 20 log10(p/p_ref), p_ref = 20 µPa (standard)']),

    Calc('power_level', 'Sound power level (re 1 pW)',
         vars=[V('Lw', '', 'sound power level', signed=True, label='Lw'),
               V('W', 'W', 'sound power'), V('W_ref', 'W', 'reference power (1 pW)')],
         equations=['Lw = 10*log(W/W_ref)/log(10)'], outputs=['Lw'],
         example={'W': 0.01, 'W_ref': 1e-12},
         keywords=['dB', 'power level', 'acoustic power'],
         assumptions=['W_ref = 1 pW is the standard reference for sound power.'],
         refs=['Lw = 10 log10(W/W_ref), W_ref = 1 pW (standard)']),

    Calc('intensity_level', 'Sound intensity level (re 1 pW/m²)',
         vars=[V('Li', '', 'sound intensity level', signed=True, label='Li'),
               V('I', 'W/m^2', 'sound intensity'), V('I_ref', 'W/m^2', 'reference intensity (1 pW/m²)')],
         equations=['Li = 10*log(I/I_ref)/log(10)'], outputs=['Li'],
         example={'I': 1e-6, 'I_ref': 1e-12},
         keywords=['dB', 'intensity level'],
         assumptions=['I_ref = 1 pW/m² is the standard reference intensity.'],
         refs=['Li = 10 log10(I/I_ref), I_ref = 1 pW/m² (standard)']),

    Calc('pressure_intensity', 'Plane-wave relation between RMS sound pressure and intensity (I = p²/ρc)',
         vars=[V('I', 'W/m^2', 'sound intensity'), V('p', 'Pa', 'RMS sound pressure'),
               V('rho', 'kg/m^3', 'medium density (air ≈ 1.2 kg/m³)', label='ρ'),
               V('c', 'm/s', 'speed of sound in the medium')],
         equations=['I = p**2/(rho*c)'], outputs=['I'],
         example={'p': 0.1, 'rho': 1.2, 'c': 343.0},
         keywords=['plane wave', 'impedance', 'characteristic impedance', 'intensity'],
         assumptions=['Plane progressive wave (free field). In air at standard conditions ρc ≈ 400 rayl, so the '
                      'intensity level and sound pressure level are numerically nearly equal.'],
         refs=['I = p_rms²/(ρc) for a plane progressive wave']),

    Calc('add_levels', 'Add (combine) sound levels in dB',
         vars=[V('levels', '', 'list of sound levels in dB', kind='json'),
               V('L_total', '', 'combined level', signed=True)],
         procedure=_add_levels, outputs=['L_total'],
         example={'levels': [80.0, 80.0, 75.0]},
         keywords=['sum', 'combine', 'incoherent', 'decibel addition'],
         assumptions=['Incoherent sources (energies add). Two equal levels add 3 dB; a level 10 dB below '
                      'another adds only ~0.4 dB.'],
         refs=['L_total = 10 log10(Σ 10^(Li/10))']),

    Calc('wave_speed', 'Wave speed, frequency and wavelength (c = f·λ)',
         vars=[V('c', 'm/s', 'wave speed'), V('f', 'Hz', 'frequency'), V('lam', 'm', 'wavelength', label='λ')],
         equations=['c = f*lam'], outputs=['c'],
         example={'f': 500.0, 'lam': 0.686},
         keywords=['wavelength', 'frequency', 'speed of sound'],
         refs=['c = f λ']),

    Calc('mass_law', 'Sound transmission loss of a limp panel (normal-incidence mass law)',
         vars=[V('TL', '', 'transmission loss', signed=True),
               V('rho_s', 'kg/m^2', 'surface mass density of the panel', label='ρs'),
               V('f', 'Hz', 'frequency'), V('rho', 'kg/m^3', 'air density (≈ 1.2 kg/m³)', label='ρ'),
               V('c', 'm/s', 'speed of sound in air (≈ 343 m/s)')],
         equations=['TL = 10*log(1 + (pi*f*rho_s/(rho*c))**2)/log(10)'], outputs=['TL'],
         example={'rho_s': 20.0, 'f': 1000.0, 'rho': 1.2, 'c': 343.0},
         keywords=['sound insulation', 'mass law', 'partition', 'transmission loss'],
         assumptions=['Limp, thin, uniform panel; normal incidence; below the coincidence frequency. The '
                      'field-incidence empirical form is TL ≈ 20 log10(f·ρs) − 47 dB.'],
         refs=['Normal-incidence mass law: TL = 10 log10(1 + (π f ρs/ρc)²)']),

    Calc('noise_reduction', 'Noise reduction between two rooms (level difference)',
         vars=[V('NR', '', 'noise reduction', signed=True),
               V('L1', '', 'source-side sound level', signed=True),
               V('L2', '', 'receiver-side sound level', signed=True)],
         equations=['NR = L1 - L2'], outputs=['NR'],
         example={'L1': 95.0, 'L2': 62.0},
         keywords=['insertion loss', 'level difference', 'attenuation'],
         refs=['NR = L1 − L2']),

    Calc('room_noise_reduction', 'Room-to-room noise reduction from transmission loss and absorption',
         vars=[V('NR', '', 'noise reduction', signed=True), V('TL', '', 'transmission loss of the partition', signed=True),
               V('A', 'm^2', 'total absorption in the receiving room (sabins)'),
               V('S', 'm^2', 'area of the common partition')],
         equations=['NR = TL + 10*log(A/S)/log(10)'], outputs=['NR'],
         example={'TL': 40.0, 'A': 100.0, 'S': 20.0},
         keywords=['room acoustics', 'partition', 'transmission loss'],
         assumptions=['Diffuse field in the receiving room; A in metric sabins (m²).'],
         refs=['NR = TL + 10 log10(A/S)']),

    Calc('reverberation_sabine', 'Reverberation time (Sabine)',
         vars=[V('T60', 's', 'reverberation time (time for sound to decay 60 dB)'),
               V('k_sabine', 's/m', 'Sabine constant (0.161 s/m in SI)', lo=0.1, hi=0.2),
               V('V', 'm^3', 'room volume'), V('A', 'm^2', 'total absorption (metric sabins)')],
         equations=['T60 = k_sabine*V/A'], outputs=['T60'],
         example={'k_sabine': 0.161, 'V': 500.0, 'A': 50.0},
         keywords=['sabine', 'reverberation', 'room', 'acoustics'],
         assumptions=['Diffuse field, uniform absorption. The constant is 0.161 s/m in SI (0.049 s/ft imperial).'],
         refs=['Sabine reverberation time: T60 = 0.161 V/A (metric)']),

    Calc('total_absorption', 'Total absorption of a room (Σ α·S)',
         vars=[V('surfaces', '', 'rows [alpha (0..1), area (m^2)] for each surface', kind='json'),
               V('A', 'm^2', 'total absorption (metric sabins)'),
               V('mean_alpha', '', 'mean absorption coefficient (area-weighted)', hi=1.0, label='ᾱ')],
         procedure=_total_absorption, outputs=['A', 'mean_alpha'],
         example={'surfaces': [[0.03, 200.0], [0.5, 40.0], [0.8, 20.0]]},
         keywords=['sabine', 'absorption', 'alpha'],
         refs=['A = Σ α_i S_i']),

    Calc('a_weighting', 'A-weighting of a frequency (IEC 61672 analytic form)',
         vars=[V('f', 'Hz', 'frequency', lo=0.5, hi=20000.0),
               V('A', '', 'A-weighting at this frequency in dB', signed=True)],
         procedure=_a_weighting, outputs=['A'],
         example={'f': 1000.0},
         keywords=['A-weighted', 'dBA', 'frequency weighting', 'iec 61672'],
         assumptions=['Analytic IEC 61672 transfer function with rounded pole frequencies 20.6, 107.7, 737.9 '
                      'and 12194 Hz (recalled). A(1000 Hz) = 0 dB by definition.'],
         refs=['IEC 61672-1 A-weighting transfer function (poles recalled, not re-read)']),

    Calc('noise_dose', 'Noise-exposure dose and time-weighted average (OSHA or NIOSH/ISO)',
         vars=[V('standard', '', 'exchange-rate convention', kind='text', choices=['OSHA', 'NIOSH/ISO']),
               V('levels', '', 'list of exposure levels in dB(A)', kind='json'),
               V('durations', '', 'list of exposure durations in hours (same order as levels)', kind='json'),
               V('exchange_rate', '', 'exchange rate (5 dB OSHA, 3 dB NIOSH/ISO)', label='Q'),
               V('criterion', '', 'criterion level (90 dBA OSHA, 85 dBA NIOSH/ISO)', label='Lc'),
               V('dose', '', 'noise dose in percent (100 % = full allowed exposure)'),
               V('twa', '', 'time-weighted average level', signed=True)],
         procedure=_noise_dose, outputs=['exchange_rate', 'criterion', 'dose', 'twa'],
         example={'standard': 'OSHA', 'levels': [90.0], 'durations': [8.0]},
         keywords=['osha', 'iso', 'niosh', 'exposure', 'dose', 'twa', 'hearing'],
         assumptions=['OSHA: 5 dB exchange rate, 90 dBA criterion, 8 h reference. NIOSH/ISO: 3 dB exchange '
                      'rate, 85 dBA criterion. Reference duration T = 8 h / 2^((L − criterion)/exchange_rate). '
                      'Every entered level is counted — no threshold (OSHA 80/90 dBA) is applied.',
                      'The "3 dB / 85 dBA" convention is the NIOSH/ISO criterion.'],
         refs=['Dose = 100 Σ(Ci/Ti); TWA = criterion + Q·log2(dose/100) (OSHA/NIOSH/ISO conventions)']),
]
