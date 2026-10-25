# DESCRIPTION: Tribology section: Coulomb friction and friction power, Archard adhesive wear, lubricant
#              viscosity at temperature (Walther ASTM D341, viscosity index ASTM D2270), EHL film
#              thickness (Hamrock–Dowson), specific film thickness (lambda ratio) and Archard flash
#              temperature.
"""Tribology calculators.

Conventions: sliding friction follows Coulomb's law (F = μN); wear follows the Archard adhesive-wear
model V = K·W·s/H with dimensionless wear coefficient K and material hardness H. Kinematic viscosity is
declared in m²/s (1 mm²/s = 1 cSt); the Walther (ASTM D341) relation uses the Wright 0.7-cSt offset.
Flash temperature is Archard's mean circular-contact result.

Relations are standard tribology results (e.g. Hamrock, Schmid & Jacobson, *Fundamentals of Fluid Film
Lubrication*; Stachowiak & Batchelor, *Engineering Tribology*). Cited by relation name, recalled rather
than re-read, so no page or equation numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('tribology', 'Tribology', order=220,
                  summary='Friction and friction power, Archard wear, lubricant viscosity (Walther, '
                          'viscosity index), EHL film thickness (Hamrock–Dowson) and flash temperature.')

PI = math.pi


# ---- procedures -----------------------------------------------------------------------------------
def _walther(values):
    """Walther (ASTM D341) viscosity–temperature relation fitted from two calibration points.

    log10(log10(nu_cSt + 0.7)) = A − B·log10(T_K) is fitted through (T1, nu1) and (T2, nu2), then
    evaluated at T_query. The +0.7 offset has units of cSt, so the declared m²/s values are converted to
    cSt (× 1e6) inside the fit and converted back for the result.
    """
    T1, T2, Tq = values['T1'], values['T2'], values['T_query']
    nu1_cst = values['nu1'] * 1e6
    nu2_cst = values['nu2'] * 1e6
    if T1 <= 0 or T2 <= 0 or Tq <= 0:
        raise MechError('temperatures must be positive (enter K)')
    if nu1_cst + 0.7 <= 1.0 or nu2_cst + 0.7 <= 1.0:
        raise MechError('Walther form needs kinematic viscosity above ~0.3 cSt (3e-7 m²/s)')
    if abs(T1 - T2) < 1e-12:
        raise MechError('T1 and T2 must be distinct calibration temperatures')
    w1 = math.log10(math.log10(nu1_cst + 0.7))
    w2 = math.log10(math.log10(nu2_cst + 0.7))
    y1, y2 = math.log10(T1), math.log10(T2)
    B = (w1 - w2) / (y2 - y1)          # slope of W = A − B·log10(T)
    A = w1 + B * y1
    wq = A - B * math.log10(Tq)
    nu_q_cst = 10 ** (10 ** wq) - 0.7
    return {'nu_query': nu_q_cst / 1e6, 'A': A, 'B': B}


def _ehl_film(values):
    """Hamrock–Dowson (1977) fully-flooded elliptical-contact EHL film thickness.

    Reduced modulus E' = 2 / ((1−ν1²)/E1 + (1−ν2²)/E2); dimensionless speed U = η0·u/(E'·R_x),
    materials G = α·E', load W = F/(E'·R_x²) and ellipticity k ≈ 1.03·(R_y/R_x)^0.64. Minimum and
    central film thickness follow the canonical power-law correlations (constants and exponents
    recalled; not re-read).
    """
    E1, nu1, E2, nu2 = values['E1'], values['nu1'], values['E2'], values['nu2']
    eta0, alpha, u, F = values['eta0'], values['alpha'], values['u'], values['F']
    Rx, Ry = values['R_x'], values['R_y']
    if Rx <= 0 or Ry <= 0:
        raise MechError('R_x and R_y must be positive')
    if E1 <= 0 or E2 <= 0:
        raise MechError('elastic moduli must be positive')
    E_red = 2.0 / ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)
    U = eta0 * u / (E_red * Rx)
    G = alpha * E_red
    W = F / (E_red * Rx ** 2)
    k = 1.03 * (Ry / Rx) ** 0.64
    H_min = 3.63 * U ** 0.68 * G ** 0.49 * W ** -0.073 * (1 - math.exp(-0.68 * k))
    H_c = 2.69 * U ** 0.67 * G ** 0.53 * W ** -0.067 * (1 - 0.61 * math.exp(-0.73 * k))
    return {'h_min': H_min * Rx, 'H_min': H_min, 'h_c': H_c * Rx, 'H_c': H_c,
            'U': U, 'G': G, 'W': W, 'k': k, 'E_red': E_red}


CALCS = [
    Calc('friction_power', 'Friction force and friction power loss (F = μN, P = F·v)',
         vars=[V('F', 'N', 'friction force (sliding)'), V('P', 'W', 'friction power loss'),
               V('mu', '', 'friction coefficient (kinetic when sliding)', label='μ'),
               V('N', 'N', 'normal force'), V('v', 'm/s', 'sliding speed')],
         equations=['F = mu*N', 'P = F*v'], outputs=['F', 'P'],
         example={'mu': 0.3, 'N': 1000.0, 'v': 2.0},
         keywords=['coulomb', 'sliding', 'heat generation'],
         assumptions=['Coulomb friction at the sliding interface; μ is the kinetic coefficient once '
                      'sliding, so P is the steady heat generated at the contact.'],
         refs=['Coulomb friction law; power P = F·v']),

    Calc('archard_wear', 'Archard adhesive wear volume and wear rate (V = K·W·s/H)',
         vars=[V('V', 'm^3', 'wear volume'), V('w', 'm^2', 'wear rate V/s (wear volume per unit sliding distance)'),
               V('K', '', 'dimensionless wear coefficient (material/pair dependent)'),
               V('W', 'N', 'normal load'), V('s', 'm', 'sliding distance'), V('H', 'Pa', 'hardness of the softer surface')],
         equations=['V = K*W*s/H', 'w = V/s'], outputs=['V', 'w'],
         example={'K': 2e-4, 'W': 100.0, 's': 500.0, 'H': 1e9},
         keywords=['adhesive wear', 'archard', 'wear volume', 'specific wear'],
         assumptions=['Adhesive-wear model with dimensionless K (typically 1e-7…1e-4); H is the indentation '
                      'hardness of the softer body. The specific wear rate k = K/H (m²/N) is the alternative form.'],
         refs=['Archard wear equation V = K·W·s/H (recalled form)']),

    Calc('walther_viscosity', 'Lubricant viscosity vs temperature (Walther, ASTM D341 two-point fit)',
         vars=[V('T1', 'K', 'first calibration temperature'), V('nu1', 'm^2/s', 'kinematic viscosity at T1 (1 mm²/s = 1 cSt)'),
               V('T2', 'K', 'second calibration temperature'), V('nu2', 'm^2/s', 'kinematic viscosity at T2'),
               V('T_query', 'K', 'temperature at which viscosity is wanted'),
               V('nu_query', 'm^2/s', 'kinematic viscosity at T_query'),
               V('A', '', 'Walther constant A'), V('B', '', 'Walther constant B')],
         procedure=_walther, outputs=['nu_query', 'A', 'B'],
         example={'T1': 313.15, 'nu1': 100e-6, 'T2': 373.15, 'nu2': 14e-6, 'T_query': 333.15},
         keywords=['viscosity-temperature', 'astm d341', 'walther', 'oil'],
         assumptions=['log10(log10(ν_cSt + 0.7)) = A − B·log10(T_K), the Wright form of ASTM D341, fitted from '
                      'two user-supplied viscosity points and extrapolated to T_query.',
                      'Valid for ν ≳ 2 cSt (the plain ASTM D341 Z = ν + 0.7 form; below ~2 cSt a correction is '
                      'needed and is not implemented here). Inputs below ~0.3 cSt are rejected so the double '
                      'log stays real. Extrapolating far outside the two calibration points reduces accuracy. '
                      'T in kelvin; ν in mm²/s (= cSt) inside the relation.'],
         refs=['Walther equation / ASTM D341 viscosity–temperature (recalled)']),

    Calc('viscosity_index', 'Viscosity index from ASTM D2270 L and H values (VI = 100(L−U)/(L−H))',
         vars=[V('VI', '', 'viscosity index', signed=True, hi=100.0),
               V('U', 'm^2/s', 'kinematic viscosity at 40 °C of the oil (1 mm²/s = 1 cSt)'),
               V('L', 'm^2/s', 'kinematic viscosity at 40 °C of a 0-VI oil with the same 100 °C viscosity (table)'),
               V('H', 'm^2/s', 'kinematic viscosity at 40 °C of a 100-VI oil with the same 100 °C viscosity (table)')],
         equations=['VI = 100*(L - U)/(L - H)'], outputs=['VI'],
         example={'U': 100e-6, 'L': 200e-6, 'H': 60e-6},
         keywords=['astm d2270', 'oil', 'vi'],
         assumptions=['L and H are tabulated functions of the 100 °C viscosity (ASTM D2270 tables) — user inputs, '
                      'not built in. This linear form is valid for VI ≤ 100 (VI is negative for oils thicker at '
                      '40 °C than the 0-VI reference, i.e. U > L); for VI > 100 use the extended D2270 form, '
                      'which is not implemented here.'],
         refs=['ASTM D2270 viscosity index, VI = 100(L−U)/(L−H) (recalled)']),

    Calc('ehl_film_thickness', 'EHL film thickness (Hamrock–Dowson elliptical contact)',
         vars=[V('E1', 'Pa', 'elastic modulus of body 1'), V('nu1', '', 'Poisson ratio of body 1', hi=0.5, label='ν1'),
               V('E2', 'Pa', 'elastic modulus of body 2'), V('nu2', '', 'Poisson ratio of body 2', hi=0.5, label='ν2'),
               V('eta0', 'Pa*s', 'inlet dynamic viscosity of the lubricant', label='η0'),
               V('alpha', '1/Pa', 'pressure–viscosity coefficient', label='α'),
               V('u', 'm/s', 'mean entrainment (rolling) speed'),
               V('F', 'N', 'normal load'), V('R_x', 'm', 'effective radius in the rolling direction'),
               V('R_y', 'm', 'effective radius transverse to rolling'),
               V('h_min', 'm', 'minimum film thickness'), V('H_min', '', 'dimensionless minimum film thickness'),
               V('h_c', 'm', 'central film thickness'), V('H_c', '', 'dimensionless central film thickness'),
               V('U', '', 'dimensionless speed parameter η0·u/(E′·R_x)'),
               V('G', '', 'dimensionless materials parameter α·E′'),
               V('W', '', 'dimensionless load parameter F/(E′·R_x²)'),
               V('k', '', 'ellipticity parameter ≈ 1.03·(R_y/R_x)^0.64'),
               V('E_red', 'Pa', 'reduced elastic modulus E′')],
         procedure=_ehl_film, outputs=['h_min', 'H_min', 'h_c', 'H_c', 'U', 'G', 'W', 'k', 'E_red'],
         example={'E1': 207e9, 'nu1': 0.3, 'E2': 207e9, 'nu2': 0.3, 'eta0': 0.05, 'alpha': 2.2e-8,
                  'u': 1.0, 'F': 1000.0, 'R_x': 0.02, 'R_y': 0.06},
         keywords=['elastohydrodynamic', 'hamrock-dowson', 'lubricant film', 'bearing'],
         assumptions=['Fully flooded, isothermal, smooth surfaces, Newtonian lubricant; elliptical contact. '
                      'Valid for R_y ≥ R_x (ellipticity k ≳ 1) with 1 ≤ k ≲ 8. The coefficients 3.63/2.69 and '
                      'exponents are the Hamrock–Dowson (1977) fits for typical EHL ranges (U ~ 1e-11, G ~ 5000, '
                      'W ~ 1e-5); k ≈ 1.03·(R_y/R_x)^0.64 is an approximation.'],
         refs=['Hamrock–Dowson minimum/central film thickness (recalled)']),

    Calc('lambda_ratio', 'Specific film thickness (lambda ratio) Λ = h_min/√(σ₁² + σ₂²)',
         vars=[V('Lambda', '', 'lambda ratio (specific film thickness)', label='Λ'),
               V('h_min', 'm', 'minimum film thickness'),
               V('sigma1', 'm', 'RMS surface roughness of surface 1', label='σ1'),
               V('sigma2', 'm', 'RMS surface roughness of surface 2', label='σ2')],
         equations=['Lambda = h_min/sqrt(sigma1**2 + sigma2**2)'], outputs=['Lambda'],
         example={'h_min': 0.3e-6, 'sigma1': 0.05e-6, 'sigma2': 0.05e-6},
         keywords=['lambda ratio', 'specific film thickness', 'regime', 'surface roughness', 'ehl'],
         assumptions=['Λ = h_min/√(σ1² + σ2²) with RMS roughnesses; √(σ1² + σ2²) is the standard composite '
                      'roughness estimate. Λ < 1 ≈ boundary lubrication, 1 < Λ < 3 ≈ mixed, Λ > 3 ≈ full-film '
                      'EHL. h_min may come from the Hamrock–Dowson calculator.'],
         refs=['Specific film thickness (lambda ratio), Λ = h_min/√(σ1² + σ2²) (recalled)']),

    Calc('flash_temperature', 'Flash temperature rise (Archard circular contact)',
         vars=[V('dT_flash', 'K', 'mean flash temperature rise above the bulk', label='ΔTf'),
               V('mu', '', 'friction coefficient', label='μ'), V('F', 'N', 'normal load'),
               V('v_s', 'm/s', 'sliding speed'), V('a', 'm', 'contact radius (circular contact)'),
               V('k1', 'W/(m*K)', 'thermal conductivity of body 1'),
               V('k2', 'W/(m*K)', 'thermal conductivity of body 2')],
         equations=['dT_flash = mu*F*v_s/(4*a*(k1 + k2))'], outputs=['dT_flash'],
         example={'mu': 0.1, 'F': 100.0, 'v_s': 2.0, 'a': 5e-4, 'k1': 50.0, 'k2': 50.0},
         keywords=['blok', 'archard', 'hot spot', 'scuffing'],
         assumptions=['Archard mean flash temperature for a single circular contact with the heat partitioned '
                      'between the two semi-infinite bodies (equal interface temperature). Valid at low sliding '
                      'speed (Péclet number Pe = v_s·a/(2κ) ≲ 0.1, heat conduction dominates); at higher speeds '
                      'the maximum (moving-source) temperature exceeds this mean. The Blok line-contact form '
                      'differs.'],
         refs=['Archard flash temperature, ΔT = μFv/(4a(k1+k2)) (recalled)']),
]
