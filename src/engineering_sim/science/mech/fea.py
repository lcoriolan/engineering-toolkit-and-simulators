# DESCRIPTION: Finite-element / numerical-analysis section: element stiffness matrices (spring/bar/beam/
#              truss), direct-stiffness assembly (1-D springs and bars, 2-D truss, Euler–Bernoulli beam),
#              shape functions and strain-displacement, element stress and strain energy, Richardson/GCI
#              mesh convergence, and 1-D finite-difference nodes for heat conduction and advection-diffusion.
"""Finite element and numerical analysis calculators.

Conventions: axial (spring/bar) member force is positive in tension; a 2-D truss member from node i to
node j carries an axial force N = (EA/L)·((u_j − u_i)·ê), positive in tension. Beam DOFs per node are
transverse displacement v (up) then rotation θ (CCW). Reactions are R = K·u − F, i.e. the force/moment the
support applies to the structure. The direct-stiffness solvers return the global stiffness matrix K,
nodal displacements, support reactions and element forces/stresses; oracles cross-check these against
closed-form deflections and method-of-joints results, never against re-typed matrices.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('fea', 'Finite element and numerical analysis', order=460,
                  summary='Element stiffness matrices, direct-stiffness assembly (springs, bars, truss, beam), '
                          'shape functions, element stress and strain energy, mesh convergence and finite-difference nodes.')


# ---- shared direct-stiffness core -----------------------------------------------------------------
def _axial_solve(elements, fixed, loads, n_nodes):
    """Assemble and solve a 1-D (spring/bar) system. elements = [(i, j, k_axial)]; fixed = [[node, u]];
    loads = [[node, F]]. Returns (K, u, R) with R = K·u − F (support reactions)."""
    K = np.zeros((n_nodes, n_nodes))
    for i, j, k in elements:
        K[i, i] += k
        K[j, j] += k
        K[i, j] -= k
        K[j, i] -= k
    fixed_map = {int(node): float(val) for node, val in (fixed or [])}
    F = np.zeros(n_nodes)
    for node, f in (loads or []):
        F[int(node)] += float(f)
    u = np.zeros(n_nodes)
    for n, val in fixed_map.items():
        u[n] = val
    free = [n for n in range(n_nodes) if n not in fixed_map]
    if free:
        Kff = K[np.ix_(free, free)]
        if np.linalg.matrix_rank(Kff) < len(free):
            raise MechError('system is a mechanism (singular stiffness matrix)')
        if fixed_map:
            fn = sorted(fixed_map)
            Ff = F[free] - K[np.ix_(free, fn)] @ np.array([fixed_map[n] for n in fn])
        else:
            Ff = F[free]
        u[free] = np.linalg.solve(Kff, Ff)
    R = K @ u - F
    return K, u, R


def _max_node(elements, *node_lists):
    """Number of nodes (max index + 1), validating every index.

    ``elements`` are the [i, j, ...] rows (two node indices per row); ``node_lists`` are the
    [node, ...] rows (fixed/loads/supports, one index per row). A negative index would silently wrap in
    numpy (K[-1, -1] indexes the last row/col), and a non-integer index would silently truncate with
    int(), so every index is validated here and at each parse site.
    """
    n = 0
    for row in elements:
        n = max(n, _node_idx(row[0], 'element') + 1, _node_idx(row[1], 'element') + 1)
    for rows in node_lists:
        for row in rows:
            n = max(n, _node_idx(row[0], 'node index') + 1)
    return n


def _node_idx(val, what, n=None):
    """Validate one node index: an integer in 0..n-1 (or >= 0 when n is None)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise MechError(f'{what}: node index must be an integer, got {val!r}')
    if float(val) != int(val):
        raise MechError(f'{what}: node index {val!r} is not an integer')
    node = int(val)
    if node < 0 or (n is not None and node >= n):
        hi = f'0..{n - 1}' if n is not None else 'non-negative'
        raise MechError(f'{what}: node index {node} out of range ({hi})')
    return node


# ---- procedures ----------------------------------------------------------------------------------
def _spring_system(values):
    """1-D spring direct stiffness: springs [[i, j, k (N/m)], ...], fixed [[node, u (m)], ...],
    loads [[node, F (N)], ...]."""
    springs = values['springs']
    fixed = values.get('fixed') or []
    loads = values.get('loads') or []
    if not isinstance(springs, list) or not springs:
        raise MechError('springs: give a non-empty list of [i, j, k (N/m)] rows')
    elems = []
    for row in springs:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise MechError('springs: each row must be [i, j, k (N/m)]')
        i = _node_idx(row[0], 'springs')
        j = _node_idx(row[1], 'springs')
        k = float(row[2])
        if k <= 0:
            raise MechError('spring stiffness must be positive')
        elems.append((i, j, k))
    n = _max_node(springs, fixed, loads)
    K, u, R = _axial_solve(elems, fixed, loads, n)
    return {'K': K.tolist(), 'displacements': u.tolist(),
            'reactions': [[int(node), float(R[int(node)])] for node, _ in fixed]}


def _bar_system(values):
    """1-D bar (rod) direct stiffness: bars [[i, j, E (Pa), A (m^2), L (m)], ...]."""
    bars = values['bars']
    fixed = values.get('fixed') or []
    loads = values.get('loads') or []
    if not isinstance(bars, list) or not bars:
        raise MechError('bars: give a non-empty list of [i, j, E, A, L] rows')
    elems, meta = [], []
    for row in bars:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise MechError('bars: each row must be [i, j, E, A, L]')
        i = _node_idx(row[0], 'bars')
        j = _node_idx(row[1], 'bars')
        E, A, L = float(row[2]), float(row[3]), float(row[4])
        if L <= 0 or A <= 0:
            raise MechError('bar length and area must be positive')
        k = E * A / L
        elems.append((i, j, k))
        meta.append((i, j, E, A))
    n = _max_node(bars, fixed, loads)
    K, u, R = _axial_solve(elems, fixed, loads, n)
    forces = [k * (u[j] - u[i]) for (i, j, k), (i0, j0, E, A) in zip(elems, meta)]
    stresses = [f / A for f, (i0, j0, E, A) in zip(forces, meta)]
    return {'K': K.tolist(), 'displacements': u.tolist(),
            'reactions': [[int(node), float(R[int(node)])] for node, _ in fixed],
            'element_forces': [float(x) for x in forces], 'element_stresses': [float(x) for x in stresses]}


def _truss_fea(values):
    """2-D plane truss by direct stiffness: nodes [[x, y] (m)], elements [[i, j, E (Pa), A (m^2)]],
    supports [[node, 'x'|'y']] (zero displacement), loads [[node, Fx, Fy] (N)]."""
    nodes = np.asarray(values['nodes'], dtype=float)
    elements = values['elements']
    supports = values.get('supports') or []
    loads = values.get('loads') or []
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise MechError('nodes: give [[x, y], ...] in m')
    if not isinstance(elements, list) or not elements:
        raise MechError('elements: give a non-empty list of [i, j, E, A] rows')
    n = len(nodes)
    K = np.zeros((2 * n, 2 * n))
    member = []
    for row in elements:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise MechError('elements: each member must be [i, j, E, A]')
        i = _node_idx(row[0], 'elements', n)
        j = _node_idx(row[1], 'elements', n)
        E, A = float(row[2]), float(row[3])
        d = nodes[j] - nodes[i]
        L = float(np.hypot(*d))
        if L <= 0:
            raise MechError(f'member {i}-{j} has zero length')
        c, s = d / L
        k = E * A / L
        ke = k * np.array([[c * c, c * s, -c * c, -c * s],
                           [c * s, s * s, -c * s, -s * s],
                           [-c * c, -c * s, c * c, c * s],
                           [-c * s, -s * s, c * s, s * s]])
        dof = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(dof, dof)] += ke
        member.append((i, j, E, A, c, s))
    F = np.zeros(2 * n)
    for row in loads:
        node = _node_idx(row[0], 'loads', n)
        F[2 * node] += float(row[1])
        F[2 * node + 1] += float(row[2])
    constrained = []
    for row in supports:
        node = _node_idx(row[0], 'supports', n)
        comp = str(row[1])
        if comp not in ('x', 'y'):
            raise MechError("supports: component must be 'x' or 'y'")
        constrained.append(2 * node + (0 if comp == 'x' else 1))
    constrained = sorted(set(constrained))
    free = [d for d in range(2 * n) if d not in constrained]
    u = np.zeros(2 * n)
    if free:
        Kff = K[np.ix_(free, free)]
        if np.linalg.matrix_rank(Kff) < len(free):
            raise MechError('truss is a mechanism (singular stiffness matrix)')
        u[free] = np.linalg.solve(Kff, F[free])
    R = K @ u - F
    forces, stresses = [], []
    for (i, j, E, A, c, s) in member:
        N = (E * A / math.hypot(nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1])) * \
            ((u[2 * j] - u[2 * i]) * c + (u[2 * j + 1] - u[2 * i + 1]) * s)
        forces.append(float(N))
        stresses.append(float(N / A))
    reactions = [[int(node), comp, float(R[2 * int(node) + (0 if comp == 'x' else 1)])]
                 for node, comp in supports]
    return {'K': K.tolist(),
            'displacements': [[float(u[2 * i]), float(u[2 * i + 1])] for i in range(n)],
            'reactions': reactions, 'element_forces': forces, 'element_stresses': stresses}


def _beam_fea(values):
    """Euler–Bernoulli beam assembly: elements [[i, j, E (Pa), I (m^4), L (m)]], supports
    [[node, 'v'|'theta']] (zero), loads [[node, P (N), M (N·m)]]. 2 DOF per node (v, θ)."""
    elements = values['elements']
    supports = values.get('supports') or []
    loads = values.get('loads') or []
    if not isinstance(elements, list) or not elements:
        raise MechError('elements: give a non-empty list of [i, j, E, I, L] rows')
    n_nodes = _max_node(elements, supports, loads)
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)
    elem = []
    for row in elements:
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise MechError('elements: each beam must be [i, j, E, I, L]')
        i = _node_idx(row[0], 'elements')
        j = _node_idx(row[1], 'elements')
        E, I, L = float(row[2]), float(row[3]), float(row[4])
        if L <= 0 or I <= 0:
            raise MechError('beam length and second moment must be positive')
        k = E * I / L ** 3
        ke = k * np.array([[12.0, 6 * L, -12.0, 6 * L],
                           [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                           [-12.0, -6 * L, 12.0, -6 * L],
                           [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
        dof = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        K[np.ix_(dof, dof)] += ke
        elem.append((i, j, k, L))
    for row in loads:
        node = _node_idx(row[0], 'loads')
        F[2 * node] += float(row[1])
        F[2 * node + 1] += float(row[2])
    constrained = []
    for row in supports:
        node = _node_idx(row[0], 'supports')
        comp = str(row[1])
        if comp not in ('v', 'theta'):
            raise MechError("supports: component must be 'v' or 'theta'")
        constrained.append(2 * node + (0 if comp == 'v' else 1))
    constrained = sorted(set(constrained))
    free = [d for d in range(n_dof) if d not in constrained]
    u = np.zeros(n_dof)
    if free:
        Kff = K[np.ix_(free, free)]
        if np.linalg.matrix_rank(Kff) < len(free):
            raise MechError('beam is a mechanism (singular stiffness matrix)')
        u[free] = np.linalg.solve(Kff, F[free])
    R = K @ u - F
    end_forces = []
    for (i, j, k, L) in elem:
        ke = k * np.array([[12.0, 6 * L, -12.0, 6 * L],
                           [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                           [-12.0, -6 * L, 12.0, -6 * L],
                           [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
        ue = u[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1]]
        end_forces.append([float(x) for x in (ke @ ue)])
    reactions = [[int(node), comp, float(R[2 * int(node) + (0 if comp == 'v' else 1)])]
                 for node, comp in supports]
    return {'K': K.tolist(),
            'displacements': [[float(u[2 * i]), float(u[2 * i + 1])] for i in range(n_nodes)],
            'reactions': reactions, 'end_forces': end_forces}


def _bar_element(values):
    """1-D bar element stiffness matrix (2×2) and axial stiffness EA/L."""
    E, A, L = values['E'], values['A'], values['L']
    if L <= 0 or A <= 0:
        raise MechError('length and area must be positive')
    k = E * A / L
    return {'k_axial': k, 'k_matrix': [[k, -k], [-k, k]]}


def _beam_element(values):
    """Euler–Bernoulli beam element stiffness matrix (4×4, DOF [v_i, θ_i, v_j, θ_j]) and 12EI/L³."""
    E, I, L = values['E'], values['I'], values['L']
    if L <= 0 or I <= 0:
        raise MechError('length and second moment must be positive')
    k = E * I / L ** 3
    m = [[12.0 * k, 6.0 * k * L, -12.0 * k, 6.0 * k * L],
         [6.0 * k * L, 4.0 * k * L ** 2, -6.0 * k * L, 2.0 * k * L ** 2],
         [-12.0 * k, -6.0 * k * L, 12.0 * k, -6.0 * k * L],
         [6.0 * k * L, 2.0 * k * L ** 2, -6.0 * k * L, 4.0 * k * L ** 2]]
    return {'k_flex': 12.0 * k, 'k_matrix': m}


def _truss_element_2d(values):
    """2-D truss element stiffness matrix in global coordinates (4×4) and axial stiffness EA/L."""
    E, A, L, theta = values['E'], values['A'], values['L'], values['theta']
    if L <= 0 or A <= 0:
        raise MechError('length and area must be positive')
    c, s = math.cos(theta), math.sin(theta)
    k = E * A / L
    kxx, kc, kyy = k * c * c, k * c * s, k * s * s
    m = [[kxx, kc, -kxx, -kc],
         [kc, kyy, -kc, -kyy],
         [-kxx, -kc, kxx, kc],
         [-kc, -kyy, kc, kyy]]
    return {'kxx': kxx, 'k_axial': k, 'k_matrix': m}


def _fd_heat_1d(values):
    """Steady 1-D heat conduction with uniform generation by central finite differences.

    k d²T/dx² + q = 0 on [0, L], T(0) = T_left, T(L) = T_right; n interior nodes. Central differences are
    exact for this constant-source quadratic, so the nodal values equal the closed form
    T(x) = T_left + (T_right − T_left)x/L + q·x(L − x)/(2k).
    """
    L, k, TL, TR, q = values['L'], values['k'], values['T_left'], values['T_right'], values['q_gen']
    n = int(values['n'])
    if L <= 0 or k <= 0 or n < 1 or n > 1000:
        raise MechError('L and k must be positive and 1 <= n <= 1000')
    dx = L / (n + 1)
    a = k / dx ** 2
    A = np.zeros((n, n))
    b = np.full(n, -q)
    for i in range(n):
        A[i, i] = -2 * a
        if i > 0:
            A[i, i - 1] = a
        if i < n - 1:
            A[i, i + 1] = a
    b[0] -= a * TL
    b[-1] -= a * TR
    T = np.linalg.solve(A, b)
    nodes = [[0.0, float(TL)]] + [[(i + 1) * dx, float(T[i])] for i in range(n)] + [[L, float(TR)]]
    return {'nodes': nodes, 'T_max': max([TL, TR] + [float(t) for t in T])}


def _fd_advection_diffusion(values):
    """Steady 1-D advection-diffusion u dc/dx = D d²c/dx², c(0)=c0, c(L)=cL, by central finite differences.

    n interior nodes; central scheme needs a cell Peclet number |u Δx/D| < 2. The analytic solution is
    c(x) = c0 + (cL − c0)(e^(Pe x/L) − 1)/(e^Pe − 1), Pe = uL/D (a convergence landmark, not a re-typed matrix).
    """
    u, D, L = values['u'], values['D'], values['L']
    c0, cL = values['c0'], values['cL']
    n = int(values['n'])
    if L <= 0 or D <= 0 or n < 1 or n > 1000:
        raise MechError('L and D must be positive and 1 <= n <= 1000')
    dx = L / (n + 1)
    pe_cell = u * dx / D
    if abs(pe_cell) >= 2.0:
        raise MechError(f'cell Peclet number {pe_cell:.3f} >= 2: central differences oscillate; use more nodes')
    lo = D / dx ** 2 + u / (2 * dx)   # coefficient of c_{i-1}
    diag = -2 * D / dx ** 2            # coefficient of c_i
    hi = D / dx ** 2 - u / (2 * dx)    # coefficient of c_{i+1}
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = lo
        if i < n - 1:
            A[i, i + 1] = hi
    b[0] = -lo * c0
    b[-1] = -hi * cL
    c = np.linalg.solve(A, b)
    xs = [(i + 1) * dx for i in range(n)]
    nodes = [[0.0, float(c0)]] + [[xs[i], float(c[i])] for i in range(n)] + [[L, float(cL)]]
    allx = [0.0] + xs + [L]
    allc = [float(c0)] + [float(x) for x in c] + [float(cL)]
    c_mid = float(np.interp(L / 2, allx, allc))
    return {'nodes': nodes, 'c_mid': c_mid}


CALCS = [
    Calc('spring_element', 'Spring element: force and elongation (δ = u₂ − u₁)',
         vars=[V('k', 'N/m', 'spring stiffness'), V('u1', 'm', 'nodal displacement 1', signed=True),
               V('u2', 'm', 'nodal displacement 2', signed=True), V('delta', 'm', 'elongation u2 − u1', signed=True, label='δ'),
               V('F1', 'N', 'element force at node 1 (−kδ)', signed=True),
               V('F2', 'N', 'element force at node 2 (+kδ)', signed=True)],
         equations=['delta = u2 - u1', 'F2 = k*delta', 'F1 = -k*delta'], outputs=['delta', 'F1', 'F2'],
         example={'k': 1000.0, 'u1': 0.0, 'u2': 0.05},
         keywords=['stiffness matrix', 'spring element'],
         assumptions=['The 2×2 element stiffness matrix is [[k, −k], [−k, k]] acting on [u1, u2].'],
         refs=['Linear spring element: f = k·δ']),

    Calc('bar_element', 'Bar (truss) element stiffness matrix (2×2)',
         vars=[V('E', 'Pa', 'elastic modulus'), V('A', 'm^2', 'cross-sectional area'), V('L', 'm', 'element length'),
               V('k_axial', 'N/m', 'axial stiffness EA/L'),
               V('k_matrix', '', 'element stiffness matrix [[k, −k], [−k, k]] in N/m', kind='json')],
         procedure=_bar_element, outputs=['k_axial', 'k_matrix'],
         example={'E': 200e9, 'A': 1e-4, 'L': 2.0},
         keywords=['rod', 'axial', 'stiffness matrix', 'bar element'],
         refs=['Bar element stiffness matrix k = EA/L [[1, −1], [−1, 1]]']),

    Calc('beam_element', 'Euler–Bernoulli beam element stiffness matrix (4×4)',
         vars=[V('E', 'Pa', 'elastic modulus'), V('I', 'm^4', 'second moment of area'), V('L', 'm', 'element length'),
               V('k_flex', 'N/m', 'transverse stiffness coefficient 12EI/L³'),
               V('k_matrix', '', '4×4 element stiffness matrix [v_i, θ_i, v_j, θ_j] (rows: N/m and N)', kind='json')],
         procedure=_beam_element, outputs=['k_flex', 'k_matrix'],
         example={'E': 200e9, 'I': 8e-6, 'L': 2.0},
         keywords=['beam element', 'stiffness matrix', 'euler bernoulli'],
         assumptions=['Flexural DOF only (no axial): [v_i, θ_i, v_j, θ_j]. Entries are 12, 6L, 4L², 2L² times EI/L³.'],
         refs=['Euler–Bernoulli beam element stiffness matrix']),

    Calc('truss_element_2d', '2-D truss element stiffness matrix (global coordinates)',
         vars=[V('E', 'Pa', 'elastic modulus'), V('A', 'm^2', 'cross-sectional area'), V('L', 'm', 'element length'),
               V('theta', 'rad', 'orientation angle from +x', lo=0.0, hi=2 * math.pi, label='θ'),
               V('kxx', 'N/m', 'stiffness entry (0,0) = (EA/L)cos²θ'),
               V('k_axial', 'N/m', 'axial stiffness EA/L'),
               V('k_matrix', '', '4×4 global stiffness matrix in N/m', kind='json')],
         procedure=_truss_element_2d, outputs=['kxx', 'k_axial', 'k_matrix'],
         example={'E': 200e9, 'A': 1e-4, 'L': 2.0, 'theta': math.pi / 4},
         keywords=['truss element', 'stiffness matrix', 'transformation'],
         assumptions=['Matrix = (EA/L)·[c², cs, −c², −cs; cs, s², −cs, −s²; −c², −cs, c², cs; −cs, −s², cs, s²], c = cos θ, s = sin θ.'],
         refs=['2-D truss element stiffness in global coordinates']),

    Calc('spring_system', '1-D spring system by direct stiffness (K, displacements, reactions)',
         vars=[V('springs', '', 'springs [[i, j, k (N/m)], ...] (0-based node indices)', kind='json'),
               V('fixed', '', 'prescribed displacements [[node, u (m)], ...]', kind='json'),
               V('loads', '', 'nodal forces [[node, F (N)], ...]', kind='json'),
               V('K', '', 'global stiffness matrix (N/m)', kind='json'),
               V('displacements', '', 'nodal displacements [u0, u1, ...] in m', kind='json'),
               V('reactions', '', '[[node, reaction force (N)], ...] at the fixed nodes', kind='json')],
         procedure=_spring_system, outputs=['K', 'displacements', 'reactions'],
         example={'springs': [[0, 1, 1000.0], [1, 2, 2000.0]], 'fixed': [[0, 0.0]], 'loads': [[2, 500.0]]},
         keywords=['direct stiffness', 'assembly', 'spring network'],
         assumptions=['Reactions are R = K·u − F (force the support applies to the structure). At least one '
                      'fixed node is required or the system is a mechanism. The stored strain energy of the '
                      'assembled system is ½uᵀKu (= ½Fᵀu).'],
         refs=['Direct stiffness assembly of spring elements']),

    Calc('bar_system', '1-D bar (rod) system by direct stiffness',
         vars=[V('bars', '', 'bars [[i, j, E (Pa), A (m^2), L (m)], ...]', kind='json'),
               V('fixed', '', 'prescribed displacements [[node, u (m)], ...]', kind='json'),
               V('loads', '', 'nodal forces [[node, F (N)], ...]', kind='json'),
               V('K', '', 'global stiffness matrix (N/m)', kind='json'),
               V('displacements', '', 'nodal displacements [u0, u1, ...] in m', kind='json'),
               V('reactions', '', '[[node, reaction force (N)], ...]', kind='json'),
               V('element_forces', '', 'axial forces in N (tension positive), one per bar', kind='json'),
               V('element_stresses', '', 'axial stresses in Pa (tension positive), one per bar', kind='json')],
         procedure=_bar_system, outputs=['K', 'displacements', 'reactions', 'element_forces', 'element_stresses'],
         example={'bars': [[0, 1, 200e9, 1e-4, 2.0]], 'fixed': [[0, 0.0]], 'loads': [[1, 50e3]]},
         keywords=['rod', 'axial', 'direct stiffness', 'bar'],
         assumptions=['Prismatic bars, axial deformation only. Cross-check: a single fixed-free bar recovers δ = PL/EA.'],
         refs=['Direct stiffness assembly of bar elements']),

    Calc('truss_fea', '2-D plane truss by direct stiffness (K, displacements, reactions, member forces)',
         vars=[V('nodes', '', 'joint coordinates [[x (m), y (m)], ...]', kind='json'),
               V('elements', '', 'members [[i, j, E (Pa), A (m^2)], ...] (0-based)', kind='json'),
               V('supports', '', "reaction components [[joint, 'x' or 'y'], ...] (zero displacement)", kind='json'),
               V('loads', '', 'joint loads [[joint, Fx (N), Fy (N)], ...]', kind='json'),
               V('K', '', 'global stiffness matrix (N/m)', kind='json'),
               V('displacements', '', '[[ux (m), uy (m)], ...] per joint', kind='json'),
               V('reactions', '', "[[joint, component, force (N)], ...]", kind='json'),
               V('element_forces', '', 'member axial forces in N, tension positive', kind='json'),
               V('element_stresses', '', 'member axial stresses in Pa, tension positive', kind='json')],
         procedure=_truss_fea, outputs=['K', 'displacements', 'reactions', 'element_forces', 'element_stresses'],
         example={'nodes': [[0, 0], [4, 0], [2, 3]], 'elements': [[0, 1, 200e9, 0.01], [1, 2, 200e9, 0.01], [0, 2, 200e9, 0.01]],
                  'supports': [[0, 'x'], [0, 'y'], [1, 'y']], 'loads': [[2, 0, -12000]]},
         keywords=['fea', 'stiffness matrix', 'truss', 'pin jointed'],
         assumptions=['Pin-jointed truss, loads at joints. Member forces match the method of joints for a '
                      'statically determinate truss. Reactions are the support forces applied to the structure.'],
         refs=['Direct stiffness method for plane trusses']),

    Calc('beam_fea', 'Euler–Bernoulli beam by direct stiffness (K, displacements, reactions, end forces)',
         vars=[V('elements', '', 'beam elements [[i, j, E (Pa), I (m^4), L (m)], ...]', kind='json'),
               V('supports', '', "restraints [[node, 'v' or 'theta'], ...] (zero displacement/rotation)", kind='json'),
               V('loads', '', 'nodal loads [[node, P (N), M (N·m)], ...]', kind='json'),
               V('K', '', 'global stiffness matrix', kind='json'),
               V('displacements', '', '[[v (m), theta (rad)], ...] per node', kind='json'),
               V('reactions', '', "[[node, component, reaction], ...] (N for 'v', N·m for 'theta')", kind='json'),
               V('end_forces', '', 'per-element [V_i, M_i, V_j, M_j] in local coordinates', kind='json')],
         procedure=_beam_fea, outputs=['K', 'displacements', 'reactions', 'end_forces'],
         example={'elements': [[0, 1, 200e9, 8e-6, 2.0]], 'supports': [[0, 'v'], [0, 'theta']], 'loads': [[1, 1000.0, 0.0]]},
         keywords=['beam', 'bending', 'fea', 'direct stiffness'],
         assumptions=['Euler–Bernoulli beam (flexure only, no shear deformation). A single-element cantilever '
                      'recovers tip deflection PL³/3EI and rotation PL²/2EI. end_forces are the per-element nodal '
                      'force vectors k_e·u_e = [V_i, M_i, V_j, M_j] in local coordinates (V = shear, M = bending '
                      'moment at each end), positive in the element stiffness-matrix sense.'],
         refs=['Direct stiffness method for Euler–Bernoulli beam elements']),

    Calc('shape_linear', 'Linear bar element shape functions (N₁, N₂ at natural coordinate ξ)',
         vars=[V('xi', '', 'natural coordinate (−1 ≤ ξ ≤ 1)', lo=-1.0, hi=1.0, signed=True, label='ξ'),
               V('N1', '', 'shape function at node 1', signed=True), V('N2', '', 'shape function at node 2', signed=True)],
         equations=['N1 = (1 - xi)/2', 'N2 = (1 + xi)/2'], outputs=['N1', 'N2'],
         example={'xi': 0.5},
         keywords=['interpolation', 'isoparametric', 'shape function'],
         assumptions=['N1 + N2 = 1 and N_i = 1 at node i, 0 at the other node.'],
         refs=['Linear 1-D shape functions N1 = (1−ξ)/2, N2 = (1+ξ)/2']),

    Calc('shape_quadratic', 'Quadratic bar element shape functions (N₁, N₂, N₃)',
         vars=[V('xi', '', 'natural coordinate (−1 ≤ ξ ≤ 1)', lo=-1.0, hi=1.0, signed=True, label='ξ'),
               V('N1', '', 'shape function at node 1', signed=True), V('N2', '', 'shape function at mid node', signed=True),
               V('N3', '', 'shape function at node 3', signed=True)],
         equations=['N1 = xi*(xi - 1)/2', 'N2 = 1 - xi**2', 'N3 = xi*(xi + 1)/2'], outputs=['N1', 'N2', 'N3'],
         example={'xi': 0.5},
         keywords=['quadratic', 'interpolation', 'shape function'],
         assumptions=['Lagrangian quadratic element (nodes at ξ = −1, 0, +1).'],
         refs=['Quadratic 1-D shape functions']),

    Calc('bar_strain_stress', 'Bar element strain and stress from nodal displacements (B matrix)',
         vars=[V('eps', '', 'axial strain (u2 − u1)/L', signed=True, label='ε'),
               V('sigma', 'Pa', 'axial stress E·ε', signed=True, label='σ'),
               V('u1', 'm', 'nodal displacement 1', signed=True), V('u2', 'm', 'nodal displacement 2', signed=True),
               V('L', 'm', 'element length'), V('E', 'Pa', 'elastic modulus')],
         equations=['eps = (u2 - u1)/L', 'sigma = E*(u2 - u1)/L'], outputs=['eps', 'sigma'],
         example={'u1': 0.0, 'u2': 0.001, 'L': 2.0, 'E': 200e9},
         keywords=['strain displacement', 'b matrix', 'element stress'],
         assumptions=['Constant-strain bar element: the strain-displacement matrix is B = [−1/L, 1/L].'],
         refs=['ε = B·u with B = [−1/L, 1/L]; σ = E·ε']),

    Calc('element_strain_energy', 'Elastic strain energy of a spring/bar element (U = ½ k δ²)',
         vars=[V('U', 'J', 'strain energy'), V('k', 'N/m', 'element stiffness (EA/L for a bar)'),
               V('u1', 'm', 'nodal displacement 1', signed=True), V('u2', 'm', 'nodal displacement 2', signed=True)],
         equations=['U = k*(u2 - u1)**2/2'], outputs=['U'],
         example={'k': 1e7, 'u1': 0.0, 'u2': 0.001},
         keywords=['strain energy', 'element energy', 'castigliano'],
         assumptions=['Linear elastic element; U = ½ uᵀ k u for a bar element.'],
         refs=['U = ½ k δ²']),

    Calc('mesh_convergence', 'Richardson extrapolation and grid-convergence index (GCI)',
         vars=[V('f1', '', 'solution on the fine grid', signed=True),
               V('f2', '', 'solution on the coarse grid', signed=True),
               V('h1', 'm', 'fine-grid spacing'), V('h2', 'm', 'coarse-grid spacing'),
               V('p', '', 'observed order of accuracy', lo=0.5, hi=8.0),
               V('Fs', '', 'GCI safety factor (1.25 for 3+ grids with observed p, 3.0 for 2 grids)', lo=1.0),
               V('r', '', 'refinement ratio h2/h1'),
               V('f_exact', '', 'Richardson-extrapolated value', signed=True),
               V('eps', '', 'estimated relative error on the fine grid'), V('GCI', '', 'grid-convergence index')],
         equations=['r = h2/h1', 'f_exact = f1 + (f1 - f2)/(r**p - 1)', 'eps = Abs(f1 - f2)/(Abs(f1)*(r**p - 1))', 'GCI = Fs*eps'],
         outputs=['r', 'f_exact', 'eps', 'GCI'],
         example={'f1': 0.5, 'f2': 0.52, 'h1': 0.5, 'h2': 1.0, 'p': 2.0, 'Fs': 1.25},
         keywords=['gci', 'richardson', 'grid refinement', 'convergence', 'verification'],
         assumptions=['f1/f2 are the same field on two grids with h2/h1 > 1; p is the observed order (2 for a '
                      'second-order scheme). GCI = Fs·|f1−f2|/(|f1|(r^p − 1)) bounds the fine-grid error. '
                      'Fs is 1.25 when the observed order p is estimated from 3+ grids, 3.0 when only two grids '
                      'are available (the safety factor depends on how p was obtained, not on r). A zero fine-grid '
                      'solution (f1 = 0) leaves the relative error undefined.'],
         refs=['Richardson extrapolation; Roache grid-convergence index']),

    Calc('fd_heat_1d', 'Steady 1-D heat conduction nodes (finite differences)',
         vars=[V('L', 'm', 'rod length'), V('k', 'W/(m*K)', 'thermal conductivity'),
               V('T_left', 'K', 'temperature at x = 0'), V('T_right', 'K', 'temperature at x = L'),
               V('q_gen', 'W/m^3', 'uniform internal heat generation'), V('n', '', 'number of interior nodes', integer=True),
               V('nodes', '', '[[x (m), T (K)], ...] including both ends', kind='json'),
               V('T_max', 'K', 'maximum nodal temperature')],
         procedure=_fd_heat_1d, outputs=['nodes', 'T_max'],
         example={'L': 1.0, 'k': 50.0, 'T_left': 300.0, 'T_right': 300.0, 'q_gen': 1e4, 'n': 5},
         no_invert={'n': 'integer node count: not a continuous function of any output, and scanning it would '
                         'build an n×n system for arbitrarily large n'},
         keywords=['finite difference', 'conduction', 'laplace', 'poisson'],
         assumptions=['Uniform k and q, steady state. Central differences are exact for the constant-source '
                      'quadratic, so the nodes match T = T_left + (T_right − T_left)x/L + q·x(L − x)/(2k).'],
         refs=['Central finite difference of k d²T/dx² + q = 0']),

    Calc('fd_advection_diffusion', 'Steady 1-D advection-diffusion nodes (finite differences)',
         vars=[V('u', 'm/s', 'flow velocity', signed=True), V('D', 'm^2/s', 'diffusivity'),
               V('L', 'm', 'length'), V('c0', 'kg/m^3', 'concentration at x = 0'), V('cL', 'kg/m^3', 'concentration at x = L'),
               V('n', '', 'number of interior nodes', integer=True),
               V('nodes', '', '[[x (m), c (kg/m^3)], ...] including both ends', kind='json'),
               V('c_mid', 'kg/m^3', 'concentration at mid-length (interpolated)')],
         procedure=_fd_advection_diffusion, outputs=['nodes', 'c_mid'],
         example={'u': 0.01, 'D': 0.01, 'L': 1.0, 'c0': 0.0, 'cL': 1.0, 'n': 20},
         no_invert={'n': 'integer node count: not a continuous function of any output, and scanning it would '
                         'build an n×n system for arbitrarily large n'},
         keywords=['finite difference', 'advection', 'diffusion', 'peclet', 'concentration'],
         assumptions=['Central differences; the cell Peclet number |u Δx/D| must be < 2 or an error is raised. '
                      'Converges to c = c0 + (cL − c0)(e^(Pe x/L) − 1)/(e^Pe − 1), Pe = uL/D.'],
         refs=['Central finite difference of u dc/dx = D d²c/dx²']),
]
