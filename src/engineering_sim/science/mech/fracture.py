# DESCRIPTION: Fracture mechanics section: stress intensity factor, critical crack/stress, energy release
#              rate and J, plastic zone size, Paris-law crack growth (closed form and Y(a) table) and
#              leak-before-break.
"""Fracture mechanics calculators (linear-elastic fracture mechanics, LEFM).

K = Y sigma sqrt(pi a) is the shared relation; the geometry factor Y is a user input with common
closed-form values given in the descriptions (centre crack in an infinite plate Y = 1; edge crack in a
semi-infinite plate Y = 1.12). Relations are the standard LEFM results (e.g. Anderson, *Fracture
Mechanics: Fundamentals and Applications*; Dowling, *Mechanical Behavior of Materials*); cited by
relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

import numpy as np
from scipy import integrate

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('fracture', 'Fracture mechanics', order=210,
                  summary='Stress intensity factor, fracture toughness and critical sizes, energy release '
                          'rate and J, plastic zone, Paris-law growth and leak-before-break.')

PI = math.pi

_Y_DESC = ('dimensionless geometry factor (centre crack in an infinite plate Y = 1; edge crack in a '
           'semi-infinite plate Y = 1.12; embedded penny crack Y = 2/pi)')


# ---- procedures ---------------------------------------------------------------------------------
def _paris_growth(v):
    """Paris-law crack growth rate: da/dN = C dK^m, with dK = Y dsigma sqrt(pi a)."""
    DK = v['Y'] * v['dsigma'] * math.sqrt(PI * v['a'])
    return {'dadN': v['C'] * DK ** v['m'], 'DK': DK}


def _paris_cycles(v):
    """Cycles to failure under the Paris law with a constant geometry factor Y.

    Closed form of N_f = integral_{a0}^{ac} da / (C (Y dsigma sqrt(pi a))^m) for m != 2 and m = 2.
    """
    C, m, dsigma, Y = v['C'], v['m'], v['dsigma'], v['Y']
    a0, ac = v['a0'], v['ac']
    if ac <= a0:
        raise MechError('ac (final crack length) must be greater than a0 (initial crack length)')
    DKbase = Y * dsigma * math.sqrt(PI)
    if abs(m - 2.0) < 1e-9:
        Nf = math.log(ac / a0) / (C * DKbase ** 2)
    else:
        Nf = (ac ** (1 - m / 2) - a0 ** (1 - m / 2)) / (C * DKbase ** m * (1 - m / 2))
    return {'N_f': Nf}


def _paris_cycles_ya(v):
    """Cycles to failure with a crack-length-dependent geometry factor Y(a) given as a table.

    N_f = integral_{a0}^{ac} da / (C (Y(a) dsigma sqrt(pi a))^m), with Y(a) linearly interpolated from
    Y_table. The integral is evaluated per table segment with scipy.integrate.quad (the integrand is
    ~a^(-m/2), so a uniform trapezoid is inaccurate; integrating segment-by-segment across the table
    breakpoints keeps Y piecewise-linear and the quadrature convergent).
    """
    C, m, dsigma = v['C'], v['m'], v['dsigma']
    a0, ac = v['a0'], v['ac']
    Ytab = v['Y_table']
    if ac <= a0:
        raise MechError('ac (final crack length) must be greater than a0 (initial crack length)')
    if not isinstance(Ytab, list) or len(Ytab) < 2:
        raise MechError('Y_table: [[a (m), Y], ...] with strictly increasing crack lengths')
    Ya = np.array([float(r[0]) for r in Ytab])
    Yv = np.array([float(r[1]) for r in Ytab])
    if np.any(np.diff(Ya) <= 0):
        raise MechError('Y_table crack lengths must be strictly increasing')
    if a0 < Ya[0] or ac > Ya[-1]:
        raise MechError('Y_table must span [a0, ac]')

    def integrand(a):
        Y = np.interp(a, Ya, Yv)
        return 1.0 / (C * (Y * dsigma * np.sqrt(PI * a)) ** m)

    breaks = [a0] + [x for x in Ya if a0 < x < ac] + [ac]
    total = 0.0
    for x0, x1 in zip(breaks[:-1], breaks[1:]):
        seg, _ = integrate.quad(integrand, x0, x1, limit=200)
        total += seg
    return {'N_f': float(total)}


def _leak_before_break(v):
    """Leak-before-break: critical through-wall crack half-length vs wall thickness.

    a_c = (K_Ic/(Y sigma))^2 / pi is the critical half-length. A through-wall crack of total length ~2t
    (half-length ~t) leaks detectably, so leak-before-break holds when the crack half-length reaches t
    before the critical half-length a_c, i.e. margin = a_c/t > 1.
    """
    K_Ic, Y, sigma, t = v['K_Ic'], v['Y'], v['sigma'], v['t']
    ac = (K_Ic / (Y * sigma)) ** 2 / PI
    return {'margin': ac / t, 'a_c': ac}


CALCS = [
    Calc('stress_intensity', 'Stress intensity factor (K = Y σ √(π a))',
         vars=[V('K', 'Pa*m^0.5', 'mode-I stress intensity factor'),
               V('Y', '', _Y_DESC), V('sigma', 'Pa', 'remote (applied) stress', label='σ'),
               V('a', 'm', 'crack length (half-length for a centre crack)')],
         equations=['K = Y*sigma*sqrt(pi*a)'], outputs=['K'],
         example={'Y': 1.12, 'sigma': 100e6, 'a': 0.01},
         keywords=['sif', 'stress intensity', 'mode i', 'lefm'],
         assumptions=['Linear-elastic mode-I loading; small-scale yielding. Y accounts for geometry and finite '
                      'width and is a user input.'],
         refs=['Stress intensity factor definition']),

    Calc('critical_crack_length', 'Critical crack length for fracture',
         vars=[V('a_c', 'm', 'critical crack length'), V('K_Ic', 'Pa*m^0.5', 'plane-strain fracture toughness'),
               V('Y', '', _Y_DESC), V('sigma', 'Pa', 'applied stress', label='σ')],
         equations=['a_c = (K_Ic/(Y*sigma))**2/pi'], outputs=['a_c'],
         example={'K_Ic': 50e6, 'Y': 1.0, 'sigma': 200e6},
         keywords=['fracture toughness', 'critical crack', 'lefm'],
         assumptions=['Fracture occurs when K = Y sigma sqrt(pi a) reaches K_Ic; solving for a gives a_c.'],
         refs=['Critical crack length from K_Ic']),

    Calc('critical_stress', 'Critical (fracture) stress for a given crack',
         vars=[V('sigma_c', 'Pa', 'critical (fracture) stress', label='σc'),
               V('K_Ic', 'Pa*m^0.5', 'plane-strain fracture toughness'),
               V('Y', '', _Y_DESC), V('a', 'm', 'crack length')],
         equations=['sigma_c = K_Ic/(Y*sqrt(pi*a))'], outputs=['sigma_c'],
         example={'K_Ic': 50e6, 'Y': 1.0, 'a': 0.01},
         keywords=['fracture', 'critical stress', 'lefm'],
         assumptions=['Fracture stress at which K reaches K_Ic for the given crack length.'],
         refs=['Critical stress from K_Ic']),

    Calc('energy_release_rate', 'Energy release rate G and J-integral (LEFM equivalence)',
         vars=[V('G_plane_strain', 'J/m^2', 'energy release rate G in plane strain'),
               V('G_plane_stress', 'J/m^2', 'energy release rate G in plane stress'),
               V('K', 'Pa*m^0.5', 'stress intensity factor'),
               V('E', 'Pa', "Young's modulus"), V('nu', '', "Poisson's ratio", hi=0.5, label='ν')],
         equations=['G_plane_strain = K**2*(1 - nu**2)/E', 'G_plane_stress = K**2/E'],
         outputs=['G_plane_strain', 'G_plane_stress'],
         example={'K': 50e6, 'E': 200e9, 'nu': 0.3},
         keywords=['j integral', 'energy release rate', 'g', 'lefm'],
         assumptions=['G = K^2/E\' with E\' = E (plane stress) or E/(1 - nu^2) (plane strain). In LEFM the J-integral '
                      'equals G, so J = G here.'],
         refs=['Irwin relation G = K^2/E\' ; J = G in LEFM']),

    Calc('plastic_zone_size', 'Irwin plastic zone radius',
         vars=[V('r_y_stress', 'm', 'Irwin plastic zone radius ahead of the crack tip (plane stress)'),
               V('r_y_strain', 'm', 'Irwin plastic zone radius ahead of the crack tip (plane strain)'),
               V('K', 'Pa*m^0.5', 'stress intensity factor'),
               V('sigma_ys', 'Pa', 'yield strength', label='σys')],
         equations=['r_y_stress = K**2/(2*pi*sigma_ys**2)', 'r_y_strain = K**2/(6*pi*sigma_ys**2)'],
         outputs=['r_y_stress', 'r_y_strain'],
         example={'K': 50e6, 'sigma_ys': 500e6},
         keywords=['plastic zone', 'irwin', 'small scale yielding'],
         assumptions=['First-order Irwin radius: r_y = K^2/(2 pi sigma_ys^2) in plane stress and one third of that '
                      'in plane strain (from the von Mises constraint). The corrected (Irwin) plastic zone size ahead '
                      'of the tip is 2 r_y (and 2 r_y_strain in plane strain). Valid for small-scale yielding '
                      '(r_y << a and r_y << remaining ligament).'],
         refs=['Irwin plastic zone size']),

    Calc('paris_growth_rate', 'Paris-law crack growth rate (da/dN = C ΔK^m)',
         vars=[V('dadN', 'm', 'crack growth per cycle da/dN (m/cycle)'),
               V('DK', 'Pa*m^0.5', 'stress-intensity range ΔK', label='ΔK'),
               V('C', '', 'Paris constant (SI: m/cycle per (Pa·√m)^m; a typical steel C ≈ 1e-11 in MPa units is 1e-29 in SI for m = 3)'),
               V('m', '', 'Paris exponent'), V('dsigma', 'Pa', 'stress range Δσ', label='Δσ'),
               V('Y', '', _Y_DESC), V('a', 'm', 'crack length')],
         procedure=_paris_growth, outputs=['dadN', 'DK'],
         example={'C': 1e-29, 'm': 3.0, 'dsigma': 50e6, 'Y': 1.0, 'a': 0.005},
         keywords=['paris', 'fatigue crack growth', 'da/dn', 'delta k'],
         assumptions=['Region II (Paris) crack growth. C carries units that depend on m (kept as a bare number), so '
                      'keep dsigma in Pa and enter C in SI.'],
         refs=['Paris law, da/dN = C dK^m']),

    Calc('paris_cycles', 'Paris-law cycles to failure (constant Y)',
         vars=[V('N_f', '', 'number of cycles to grow the crack from a0 to ac'),
               V('C', '', 'Paris constant (SI: m/cycle per (Pa·√m)^m)'),
               V('m', '', 'Paris exponent'), V('dsigma', 'Pa', 'stress range Δσ', label='Δσ'),
               V('Y', '', _Y_DESC), V('a0', 'm', 'initial crack length'),
               V('ac', 'm', 'final (critical) crack length')],
         procedure=_paris_cycles, outputs=['N_f'],
         example={'C': 1e-29, 'm': 3.0, 'dsigma': 50e6, 'Y': 1.0, 'a0': 0.002, 'ac': 0.005},
         keywords=['paris', 'fatigue life', 'cycles to failure'],
         assumptions=['Constant Y over the growth interval (closed form). For m = 2 the closed form is logarithmic; '
                      'otherwise it is a power law. For a crack-length-dependent Y use paris_cycles_ya.'],
         refs=['Integrated Paris law (constant geometry factor)']),

    Calc('paris_cycles_ya', 'Paris-law cycles to failure (Y(a) table, numeric integration)',
         vars=[V('N_f', '', 'number of cycles to grow the crack from a0 to ac'),
               V('C', '', 'Paris constant (SI: m/cycle per (Pa·√m)^m)'),
               V('m', '', 'Paris exponent'), V('dsigma', 'Pa', 'stress range Δσ', label='Δσ'),
               V('Y_table', '', '[[a (m), Y], ...] geometry factor vs crack length, spanning [a0, ac]', kind='json'),
               V('a0', 'm', 'initial crack length'), V('ac', 'm', 'final (critical) crack length')],
         procedure=_paris_cycles_ya, outputs=['N_f'],
         example={'C': 1e-29, 'm': 3.0, 'dsigma': 50e6, 'Y_table': [[0.002, 1.0], [0.005, 1.0]],
                  'a0': 0.002, 'ac': 0.005},
         keywords=['paris', 'fatigue life', 'geometry factor', 'weight function'],
         assumptions=['Y(a) is linearly interpolated from the table and the Paris integral evaluated per table segment '
                      'with adaptive quadrature (scipy.integrate.quad). With a constant-Y table the result matches '
                      'paris_cycles.'],
         refs=['Integrated Paris law with variable geometry factor']),

    Calc('leak_before_break', 'Leak-before-break criterion',
         vars=[V('margin', '', 'LBB margin a_c/t; satisfied when > 1'),
               V('a_c', 'm', 'critical through-thickness crack half-length'),
               V('K_Ic', 'Pa*m^0.5', 'plane-strain fracture toughness'),
               V('Y', '', _Y_DESC), V('sigma', 'Pa', 'operating (hoop) stress', label='σ'),
               V('t', 'm', 'wall thickness')],
         procedure=_leak_before_break, outputs=['margin', 'a_c'],
         example={'K_Ic': 50e6, 'Y': 1.0, 'sigma': 200e6, 't': 0.005},
         keywords=['lbb', 'pressure vessel', 'through-wall crack', 'pipeline'],
         assumptions=['a_c is the critical half-length from K_Ic. A through-wall crack of total length ~2t '
                      '(half-length ~t) leaks detectably, so leak-before-break holds when the crack grows through the '
                      'wall (half-length t) before reaching a_c, i.e. margin = a_c/t > 1. Use the geometry factor Y for '
                      'a long through-thickness crack.'],
         refs=['Leak-before-break criterion (simplified)']),
]
