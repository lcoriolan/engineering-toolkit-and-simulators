# DESCRIPTION: Bearings section: rolling-element equivalent load, rating life (L10, reliability-adjusted,
#              Weibull), static load capacity, journal-bearing Sommerfeld number and Petroff friction,
#              minimum film thickness/eccentricity, oil flow, temperature rise, power loss and thrust
#              bearing load capacity.
"""Bearings calculators.

Conventions:
- Rolling-element catalogue values (X, Y, V, X0, Y0, C, C0, the reliability factor a1 and the Weibull
  parameters x0 / theta / b) are USER INPUTS naming the manufacturer catalogue or ISO 281 table; none are
  built in from memory. The reliability-adjusted life is L_na = a1 * L10 with a1 tabulated (ISO 281).
- The Weibull reliability relation R = exp(-((L - x0)/(theta - x0))^b) is the three-parameter form the
  bearing makers use; x0, theta and b are inputs. Commonly cited maker values are x0 = 0.02 L10,
  theta = 4.459 L10, b = 1.483 (recalled); those figures are inputs here, not defaults.
- Journal bearings: the Sommerfeld number and Petroff friction are closed forms. Every Raimondi–Boyd
  chart variable (dimensionless flow, side-flow ratio, temperature-rise variable, and the S–epsilon
  curve itself) is a USER INPUT; none are recalled. Lubricant viscosity is an input (tribology owns
  viscosity-at-temperature).
- Shaft/journal speed n is a CYCLIC frequency (rev/s = Hz): enter rpm as "1500 1/min" or rpm/60, not
  "1500 rpm" (the framework rejects rpm for a cyclic variable because pint reads rpm as an angular rate).

Relations are the standard ones in machine-design texts (e.g. Budynas & Nisbett, *Shigley's Mechanical
Engineering Design*; Khonsari & Booser, *Applied Tribology*), cited by relation name, recalled.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('bearings', 'Bearings', order=160,
                  summary='Rolling-element load and life, journal bearings (Sommerfeld, Petroff, film '
                          'thickness, oil flow, temperature), thrust bearing capacity.')

PI = math.pi


def _bearing_life(values):
    """Rating life L10 (millions of rev) and L10h (hours) from C, equivalent load P, exponent a, speed n.

    L10 = (C/P)^a with a = 3 (ball) or 10/3 (roller). L10h = L10 * 1e6 / (3600 * n) with n in rev/s (Hz).
    """
    C, P, a, n = values['C'], values['P'], values['a'], values['n']
    if P <= 0:
        raise MechError('P (equivalent dynamic load) must be positive')
    if n <= 0:
        raise MechError('n (speed) must be positive to convert life to hours')
    L10 = (C / P) ** a
    # 3600 is the s->h conversion so the output carries its declared unit 'h' (life in hours, the
    # engineering convention); n is rev/s (Hz), so L10*1e6/n is seconds.
    L10h = L10 * 1e6 / (3600.0 * n)
    return {'L10': L10, 'L10h': L10h}


def _weibull(values):
    """Reliability R from the three-parameter Weibull bearing-life distribution.

    R = exp(-((L - x0)/(theta - x0))^b). Requires theta > x0 (characteristic life above the guaranteed
    life) and L >= x0 (life of interest at least the guaranteed life), and b > 0; raises MechError
    otherwise, because the ratio (L - x0)/(theta - x0) must be a real fraction in [0, 1].
    """
    L, x0, theta, b = values['L'], values['x0'], values['theta'], values['b']
    if theta <= x0:
        raise MechError('theta (characteristic life) must exceed x0 (guaranteed life)')
    if L < x0:
        raise MechError('L (life of interest) must be at least x0 (guaranteed life)')
    if b <= 0:
        raise MechError('b (Weibull shape parameter) must be positive')
    return {'R': math.exp(-((L - x0) / (theta - x0)) ** b)}


CALCS = [
    # ---- rolling-element ----------------------------------------------------------------------
    Calc('rolling_load', 'Rolling-bearing equivalent dynamic load (P = X V Fr + Y Fa)',
         vars=[V('P', 'N', 'equivalent dynamic load'),
               V('X', '', 'radial load factor (catalogue)'),
               V('V', '', 'rotation factor: 1 rotating inner ring, 1.2 rotating outer ring'),
               V('Fr', 'N', 'radial load'),
               V('Y', '', 'axial load factor (catalogue)'),
               V('Fa', 'N', 'axial load')],
         equations=['P = X*V*Fr + Y*Fa'], outputs=['P'],
         example={'X': 0.56, 'V': 1.0, 'Fr': 5000.0, 'Y': 1.5, 'Fa': 2000.0},
         keywords=['SKF', 'ISO 281', 'catalogue', 'ball', 'roller'],
         assumptions=['X, Y and V are catalogue values from the manufacturer (ISO 281): they depend on '
                      'bearing type, the contact angle and the ratio Fa/Fr. V = 1 when the inner ring '
                      'rotates, 1.2 when the outer ring rotates (shaft stationary).'],
         refs=['Equivalent dynamic bearing load P = X V Fr + Y Fa (ISO 281; recalled)']),

    Calc('bearing_life', 'Basic dynamic load rating: L10 life (millions of rev and hours)',
         vars=[V('C', 'N', 'basic dynamic load rating (catalogue)'),
               V('P', 'N', 'equivalent dynamic load'),
               V('a', '', 'life exponent: 3 ball, 10/3 roller', lo=1.0, hi=5.0),
               V('n', 'Hz', 'shaft speed in rev/s (enter rpm/60 or "1500 1/min")'),
               V('L10', '', 'rating life in millions of revolutions'),
               V('L10h', 'h', 'rating life in hours')],
         procedure=_bearing_life, outputs=['L10', 'L10h'],
         example={'C': 50000.0, 'P': 5000.0, 'a': 3.0, 'n': 30.0},
         keywords=['rating life', 'C/P', 'ISO 281', 'L10'],
         assumptions=['L10 = (C/P)^a in millions of revolutions (a = 3 ball, 10/3 roller). L10h = '
                      'L10*1e6/(3600*n) with n in rev/s. Feed P from rolling_load; solve for C (bearing '
                      'selection) by giving a target L10h. C and P must be in the same units.'],
         refs=['Rating life L10 = (C/P)^a (ISO 281; recalled)']),

    Calc('bearing_reliability_life', 'Reliability-adjusted life (L_na = a1 * L10)',
         vars=[V('L_na', '', 'reliability-adjusted life (millions of rev)'),
               V('a1', '', 'life adjustment factor for reliability (ISO 281 / catalogue)'),
               V('L10', '', 'rating life (millions of rev)')],
         equations=['L_na = a1*L10'], outputs=['L_na'],
         example={'a1': 0.62, 'L10': 1000.0},
         keywords=['reliability', 'a1 factor', 'L5', 'ISO 281'],
         assumptions=['a1 is a tabulated input (ISO 281:2007, recalled): R = 90% -> 1, 95% -> 0.62, '
                      '96% -> 0.53, 97% -> 0.44, 98% -> 0.33, 99% -> 0.21. It is not built in.'],
         refs=['Reliability-adjusted life L_na = a1 L10 (ISO 281; recalled)']),

    Calc('weibull_reliability', 'Bearing reliability from three-parameter Weibull life distribution',
         vars=[V('R', '', 'reliability (survival probability) fraction', hi=1.0),
               V('L', '', 'life of interest (millions of rev)'),
               V('x0', '', 'guaranteed (minimum) life, x0 = f_x0 L10 (input)'),
               V('theta', '', 'characteristic life (Weibull scale parameter, input)'),
               V('b', '', 'Weibull shape (dispersion) parameter (input)')],
         procedure=_weibull, outputs=['R'],
         example={'L': 500.0, 'x0': 0.0, 'theta': 1000.0, 'b': 1.5},
         keywords=['weibull', 'survival', 'reliability', 'distribution'],
         assumptions=['Three-parameter Weibull R = exp(-((L-x0)/(theta-x0))^b) for L >= x0 and theta > x0 '
                      '(a MechError is raised otherwise, e.g. L = 1 with x0 = 2, theta = 1). x0, theta and b '
                      'are INPUTS; commonly cited maker values are x0 = 0.02 L10, theta = 4.459 L10, '
                      'b = 1.483 (recalled). At L = theta the reliability is e^-1 = 0.3679 for any b.'],
         refs=['Weibull reliability distribution for bearing life (recalled)']),

    Calc('static_load', 'Static load capacity and static safety factor (P0, s0 = C0/P0)',
         vars=[V('P0', 'N', 'equivalent static load'),
               V('s0', '', 'static safety factor C0/P0'),
               V('X0', '', 'static radial load factor (catalogue)'),
               V('Fr', 'N', 'radial load'),
               V('Y0', '', 'static axial load factor (catalogue)'),
               V('Fa', 'N', 'axial load'),
               V('C0', 'N', 'basic static load rating (catalogue)')],
         equations=['P0 = Max(X0*Fr + Y0*Fa, Fr)', 's0 = C0/P0'], outputs=['P0', 's0'],
         example={'X0': 0.6, 'Fr': 5000.0, 'Y0': 1.5, 'Fa': 2000.0, 'C0': 20000.0},
         keywords=['static rating', 'C0', 'ISO 76', 'brinell'],
         assumptions=['ISO 76 equivalent static load P0 = max(X0 Fr + Y0 Fa, Fr): the larger of the '
                      'combined term and the radial load alone, so P0 is never less than Fr (this is the '
                      'binding case when the axial load is small or zero). X0, Y0, C0 are catalogue values. '
                      's0 = C0/P0 > 1 (typically 2+) for smooth operation.'],
         refs=['Equivalent static bearing load P0 = max(X0 Fr + Y0 Fa, Fr) (ISO 76; recalled)']),

    # ---- journal bearings --------------------------------------------------------------------
    Calc('sommerfeld', 'Journal bearing Sommerfeld number (S = mu n (r/c)^2 / P)',
         vars=[V('S', '', 'Sommerfeld number'),
               V('mu', 'Pa*s', 'lubricant dynamic viscosity at operating temperature (input)'),
               V('n', 'Hz', 'journal speed in rev/s (enter rpm/60)'),
               V('r', 'm', 'journal radius'),
               V('c', 'm', 'radial clearance (bearing radius minus journal radius)'),
               V('L', 'm', 'bearing (axial) length'),
               V('W', 'N', 'journal load'),
               V('P', 'Pa', 'projected bearing pressure W/(2 r L)')],
         equations=['P = W/(2*r*L)', 'S = mu*n*(r/c)**2/P'], outputs=['P', 'S'],
         example={'mu': 0.03, 'n': 20.0, 'r': 0.025, 'c': 5e-5, 'L': 0.05, 'W': 5000.0},
         keywords=['hydrodynamic', 'journal', 'sommerfeld', 'bearing number'],
         assumptions=['S is dimensionless. The relationship between S and the eccentricity ratio epsilon '
                      'comes from the Raimondi–Boyd charts (NOT built in — see journal_min_film). Light '
                      'loads (large S) are valid for the Petroff friction model.'],
         refs=['Sommerfeld number S = (mu n/P)(r/c)^2 (recalled)']),

    Calc('petroff', 'Petroff friction coefficient and torque (light-load journal bearing)',
         vars=[V('P', 'Pa', 'projected bearing pressure W/(2 r L)'),
               V('f', '', 'coefficient of friction'),
               V('T_f', 'N*m', 'friction torque'),
               V('mu', 'Pa*s', 'lubricant dynamic viscosity (input)'),
               V('n', 'Hz', 'journal speed in rev/s (enter rpm/60)'),
               V('r', 'm', 'journal radius'),
               V('c', 'm', 'radial clearance'),
               V('L', 'm', 'bearing (axial) length'),
               V('W', 'N', 'journal load')],
         equations=['P = W/(2*r*L)', 'f = 2*pi**2*mu*n*(r/c)/P', 'T_f = f*W*r'],
         outputs=['P', 'f', 'T_f'],
         example={'mu': 0.03, 'n': 20.0, 'r': 0.025, 'c': 5e-5, 'L': 0.05, 'W': 5000.0},
         keywords=['petroff', 'friction torque', 'hydrodynamic', 'light load'],
         assumptions=['Petroff applies to lightly loaded (concentric, epsilon ~ 0) bearings: '
                      'f = 2 pi^2 (mu n / P)(r / c). Equivalent torque form T_f = 4 pi^2 mu n r^3 L / c. '
                      'For significant eccentricity use the Raimondi–Boyd friction variable (input).'],
         refs=['Petroff equation for lightly loaded journal bearing friction (recalled)']),

    Calc('journal_min_film', 'Minimum film thickness and eccentricity ratio (h0 = c(1 - epsilon))',
         vars=[V('eps', '', 'eccentricity ratio e/c', hi=1.0, label='ε'),
               V('e', 'm', 'eccentricity (journal centre offset)'),
               V('c', 'm', 'radial clearance'),
               V('h0', 'm', 'minimum film thickness')],
         equations=['eps = e/c', 'h0 = c - e'], outputs=['eps', 'h0'],
         example={'e': 3e-5, 'c': 5e-5},
         keywords=['film thickness', 'eccentricity', 'journal', 'hydrodynamic'],
         assumptions=['h0 = c - e = c(1 - epsilon) is the minimum film thickness at the point of closest '
                      'approach. epsilon vs the Sommerfeld number S comes from the Raimondi–Boyd charts '
                      '(input — not built in). Require h0 >= surface roughness to avoid metal contact.'],
         refs=['Minimum film thickness h0 = c(1 - epsilon) (recalled)']),

    Calc('journal_power_loss', 'Journal bearing friction power loss (H = f W 2 pi r n)',
         vars=[V('H', 'W', 'friction power loss'),
               V('f', '', 'coefficient of friction (from Petroff or Raimondi–Boyd input)'),
               V('W', 'N', 'journal load'),
               V('r', 'm', 'journal radius'),
               V('n', 'Hz', 'journal speed in rev/s (enter rpm/60)')],
         equations=['H = f*W*2*pi*r*n'], outputs=['H'],
         example={'f': 0.003, 'W': 5000.0, 'r': 0.025, 'n': 20.0},
         keywords=['friction power', 'heat generation', 'journal'],
         assumptions=['H = T_f omega with T_f = f W r and omega = 2 pi n. The coefficient f may come from '
                      'Petroff (light load) or the Raimondi–Boyd friction variable f (r/c) (input).'],
         refs=['Power loss H = f W (2 pi r n) (recalled)']),

    Calc('journal_temp_rise', 'Oil temperature rise across a bearing (dT = H / (rho cp Q))',
         vars=[V('dT', 'K', 'lubricant temperature rise (delta; enter in K or delta_degC)'),
               V('H', 'W', 'heat to be carried away (friction power loss)'),
               V('rho', 'kg/m^3', 'lubricant density', label='ρ'),
               V('cp', 'J/(kg*K)', 'lubricant specific heat'),
               V('Q', 'm^3/s', 'lubricant (oil) volumetric flow rate')],
         equations=['dT = H/(rho*cp*Q)'], outputs=['dT'],
         example={'H': 47.1, 'rho': 850.0, 'cp': 1900.0, 'Q': 5e-5},
         keywords=['temperature rise', 'oil flow', 'heat balance'],
         assumptions=['Steady energy balance over the bearing: the heat generated is carried off by the '
                      'oil flow, so dT = H/(rho cp Q). This form uses the total flow Q and neglects the '
                      'side (leakage) flow; the Raimondi–Boyd correction for side flow is dT = '
                      'H/(rho cp (Q - 0.5 Qs)) with Qs the side flow (input) — apply it when Qs/Q is '
                      'significant. Aim for dT ~ 10–20 K across a bearing.'],
         refs=['Energy balance H = rho cp Q dT (recalled)']),

    Calc('journal_oil_flow', 'Journal bearing oil flow (Q = q* r c n L)',
         vars=[V('Q', 'm^3/s', 'oil volumetric flow rate'),
               V('q_f', '', 'dimensionless flow variable Q/(r c n L) (Raimondi–Boyd chart input)'),
               V('r', 'm', 'journal radius'),
               V('c', 'm', 'radial clearance'),
               V('n', 'Hz', 'journal speed in rev/s (enter rpm/60)'),
               V('L', 'm', 'bearing (axial) length')],
         equations=['Q = q_f*r*c*n*L'], outputs=['Q'],
         example={'q_f': 4.3, 'r': 0.025, 'c': 5e-5, 'n': 20.0, 'L': 0.05},
         keywords=['oil flow', 'flow variable', 'raimondi boyd'],
         assumptions=['q_f = Q/(r c n L) is the dimensionless flow variable read from the Raimondi–Boyd '
                      'charts at the operating S and L/D — an INPUT, not built in. The total flow feeds '
                      'journal_temp_rise.'],
         refs=['Dimensionless flow Q/(r c n L) (Raimondi–Boyd; recalled)']),

    # ---- thrust ------------------------------------------------------------------------------
    Calc('thrust_bearing_load', 'Thrust (collar/pivot) bearing load capacity (W = p_allow A)',
         vars=[V('A', 'm^2', 'projected (bearing) area'),
               V('W', 'N', 'load capacity'),
               V('p_allow', 'Pa', 'allowable bearing pressure (material/application input)'),
               V('d_o', 'm', 'outer diameter of the thrust surface'),
               V('d_i', 'm', 'inner diameter (0 for a solid/pivot surface)')],
         equations=['A = pi*(d_o**2 - d_i**2)/4', 'W = p_allow*A'], outputs=['A', 'W'],
         example={'p_allow': 2e6, 'd_o': 0.06, 'd_i': 0.02},
         keywords=['thrust collar', 'pivot bearing', 'michael', 'step bearing'],
         assumptions=['Uniform pressure over the annular thrust face; p_allow is a table value for the '
                      'material pair and duty (input). For rolling thrust bearings use bearing_life with '
                      'the axial dynamic rating C_a and a = 10/3 (roller) or 3 (ball).'],
         refs=['Thrust bearing capacity W = p_allow A, A = pi/4 (d_o^2 - d_i^2) (recalled)']),
]
