import numpy as np
import scipy.special
from numpy.polynomial.legendre import leggauss

# ---------- Lebesgue L2 quadrature on [a,b] ----------
def get_legendre_quad_on_interval(N, a, b):
    t, w = leggauss(N)  # nodes/weights for ∫_{-1}^1 f(t) dt
    x = 0.5 * (t + 1.0) * (b - a) + a
    wx = 0.5 * (b - a) * w
    return t, w, x, wx

def get_cosine_quad_on_interval(N, a, b):
    """DCT-II nodes: z_k = (2k+1)/(2N), shifted to [a,b]."""
    k = np.arange(N)
    z = (2*k + 1) / (2*N)          # nodes in [0,1]
    x = a + z * (b - a)
    w = np.full(N, (b - a) / N)
    return None, None, x, w

def get_sine_quad_on_interval(N, a, b):
    """Uniform quadrature — correct for sine/Fourier bases."""
    x = np.linspace(a, b, N, endpoint=False)
    w = np.full(N, (b - a) / N)
    return None, None, x, w   # no standard nodes needed

def get_basis_interpolation(domain, N_quad, basis_type="legendre"):
    a, b = domain[0]
    c, d = domain[1]

    if basis_type == "cosine":
        _, _, x_quad, wx = get_cosine_quad_on_interval(N_quad, a, b)
        _, _, y_quad, wy = get_cosine_quad_on_interval(N_quad, c, d)
    elif basis_type in ("sine", "fourier"):
        _, _, x_quad, wx = get_sine_quad_on_interval(N_quad, a, b)
        _, _, y_quad, wy = get_sine_quad_on_interval(N_quad, c, d)
    else:
        _, _, x_quad, wx = get_legendre_quad_on_interval(N_quad, a, b)
        _, _, y_quad, wy = get_legendre_quad_on_interval(N_quad, c, d)

    Xq, Yq = np.meshgrid(x_quad, y_quad, indexing="ij")
    return x_quad, y_quad, wx, wy, Xq, Yq

# def get_basis_interpolation(N_quad, domain, basis_type="legendre"):
#     a, b = domain[0]
#     c, d = domain[1]

#     tx, wx_std, x_quad, w_x = get_legendre_quad_on_interval(N_quad, a, b)
#     ty, wy_std, y_quad, w_y = get_legendre_quad_on_interval(N_quad, c, d)

#     # We return both standard nodes (tx,ty) and physical nodes (x_quad,y_quad),
#     # but the *evaluation* for physical L2 should be on x_quad,y_quad.
#     return tx, wx_std, ty, wy_std, x_quad, y_quad, w_x, w_y

# ---------- Physical-domain orthonormal Legendre ----------
def orthonormal_legendre_physical(n, a, b):
    # φ_n(x) = sqrt(2/(b-a)) * sqrt((2n+1)/2) * P_n(t(x))
    #       = sqrt((2n+1)/(b-a)) * P_n(t(x))
    norm = np.sqrt((2*n + 1) / (b - a))
    Pn = scipy.special.legendre(n)
    def phi(x):
        t = (2.0*(x - a)/(b - a)) - 1.0
        return norm * Pn(t)
    return phi

# ---------- Chebyshev-derived orthonormal basis under Lebesgue ----------
# We build ψ_k(t) = sum_{j<=k} C[j,k] T_j(t) such that ∫_{-1}^1 ψ_k ψ_l dt = δ_kl (approximately),
# then lift to physical domain with sqrt(2/(b-a)) and t(x) mapping.

def _chebyshev_T(j, t):
    # scipy.special.chebyt(j) returns a polynomial object; evaluating it on arrays is fine.
    return scipy.special.chebyt(j)(t)

def _build_cheb_lebesgue_orthonorm_coeffs(N_basis, N_quad):
    # Use Gauss–Legendre nodes to approximate Lebesgue inner products on [-1,1]
    t, w = leggauss(N_quad)
    A = np.stack([_chebyshev_T(j, t) for j in range(N_basis)], axis=1)  # (N_quad, N_basis)
    # Gram matrix under weighted discrete inner product <u,v> ≈ u^T diag(w) v
    G = A.T @ (w[:, None] * A)  # (N_basis, N_basis), SPD in practice if N_quad >= N_basis
    L = np.linalg.cholesky(G)   # G = L L^T
    C = np.linalg.solve(L.T, np.eye(N_basis))  # C = L^{-T}; columns are coefficients in T_j basis
    # Then B = A @ C has B^T diag(w) B = I
    return C  # shape (N_basis, N_basis), lower-ish dense

def orthonormal_chebyshev_lebesgue_physical(n, N_basis, a, b, N_quad_for_orth=200):
    # Precompute coefficients once per (N_basis, N_quad_for_orth).
    # If you care, wrap this in an LRU cache keyed by (N_basis, N_quad_for_orth).
    C = _build_cheb_lebesgue_orthonorm_coeffs(N_basis, N_quad_for_orth)
    cn = C[:, n]  # coefficients for ψ_n in terms of T_0..T_{N_basis-1}

    lift = np.sqrt(2.0 / (b - a))  # standard -> physical normalization
    def phi(x):
        t = (2.0*(x - a)/(b - a)) - 1.0
        # evaluate sum_j cn[j] T_j(t)
        Tvals = np.stack([_chebyshev_T(j, t) for j in range(N_basis)], axis=0)  # (N_basis, ...)
        return lift * np.tensordot(cn, Tvals, axes=(0, 0))
    return phi

def orthonormal_cosine_physical(n, a, b):
    """
    Cosine basis (Neumann-ish):
      n=0:  1/sqrt(L)
      n>=1: sqrt(2/L) * cos(n*pi*(x-a)/L)
    """
    L = (b - a)
    if n == 0:
        def phi(x):
            x = np.asarray(x)
            return np.ones_like(x, dtype=float) / np.sqrt(L)
        return phi
    k = n
    scale = np.sqrt(2.0 / L)
    def phi(x):
        x = np.asarray(x)
        z = (x - a) / L
        return scale * np.cos(k * np.pi * z)
    return phi

def orthonormal_sine_physical(n, a, b):
    """
    Sine basis (Dirichlet-ish):
      index n=0.. corresponds to mode k=n+1:
      phi_n(x) = sqrt(2/L) * sin((n+1)*pi*(x-a)/L)
    """
    L = (b - a)
    k = n + 1
    scale = np.sqrt(2.0 / L)
    def phi(x):
        x = np.asarray(x)
        z = (x - a) / L
        return scale * np.sin(k * np.pi * z)
    return phi

def orthonormal_fourier_physical(n, a, b):
    """
    Periodic Fourier basis on [a,b] (continuous orthonormality in Lebesgue L2):
      n=0: 1/sqrt(L)
      n=1: sqrt(2/L)*cos(2π*1*z)
      n=2: sqrt(2/L)*sin(2π*1*z)
      n=3: sqrt(2/L)*cos(2π*2*z)
      n=4: sqrt(2/L)*sin(2π*2*z)
      ...
      z=(x-a)/L
    """
    L = (b - a)
    if n == 0:
        def phi(x):
            x = np.asarray(x)
            return np.ones_like(x, dtype=float) / np.sqrt(L)
        return phi
    k = (n + 1) // 2  # 1,1,2,2,3,3,...
    scale = np.sqrt(2.0 / L)
    def phi(x):
        x = np.asarray(x)
        z = (x - a) / L
        if n % 2 == 1:
            return scale * np.cos(2.0 * np.pi * k * z)
        else:
            return scale * np.sin(2.0 * np.pi * k * z)
    return phi


# ---------- Basis dispatcher ----------
def get_basis_function(n, N_basis, basis_type, domain, N_quad_for_orth=200):
    """
    domain is expected as [[a,b],[c,d]] in your 2D pipeline; we only use the first interval here.
    """
    a, b = domain[0]
    if basis_type == "legendre":
        return orthonormal_legendre_physical(n, a, b)
    if basis_type == "chebyshev":
        return orthonormal_chebyshev_lebesgue_physical(n, N_basis=N_basis, a=a, b=b, N_quad_for_orth=N_quad_for_orth)
    if basis_type == "cosine":
        return orthonormal_cosine_physical(n, a, b)
    if basis_type == "sine":
        return orthonormal_sine_physical(n, a, b)
    if basis_type == "fourier":
        return orthonormal_fourier_physical(n, a, b)
    raise NotImplementedError(f"Unknown basis_type: {basis_type}")


def get_basis_function_1d(n, N_basis, basis_type, interval, N_quad_for_orth=200):
    """
    domain is expected as [[a,b],[c,d]] in your 2D pipeline; we only use the first interval here.
    """
    a, b = interval
    if basis_type == "legendre":
        return orthonormal_legendre_physical(n, a, b)
    if basis_type == "chebyshev":
        return orthonormal_chebyshev_lebesgue_physical(n, N_basis=N_basis, a=a, b=b, N_quad_for_orth=N_quad_for_orth)
    if basis_type == "cosine":
        return orthonormal_cosine_physical(n, a, b)
    if basis_type == "sine":
        return orthonormal_sine_physical(n, a, b)
    if basis_type == "fourier":
        return orthonormal_fourier_physical(n, a, b)
    raise NotImplementedError(f"Unknown basis_type: {basis_type}")



# ---------- Sampling: thread N_quad_for_orth so sampling/projection can use the same constructed basis ----------
def decay_calculator(decay_type, decay_rate, n, m):
    if decay_type == "none":
        return 1.0
    if decay_type == "exponential":
        return np.exp(-decay_rate * ((n + m)**2))
    if decay_type == "polynomial":
        return 1.0 / ((1 + n + m) ** decay_rate)
    raise ValueError(f"Unknown decay_type: {decay_type}")

''' not used. '''
def sample_random_function_from_basis_2d(
    N_basis=3,
    N_quad=50,
    coeff_sampler=None,
    basis_type="legendre",
    batch_size=1,
    domain=[[0, 1], [0, 1]],
    decay_type="exponential",
    decay_rate=0.1,
    random_seed=None,
    N_quad_for_orth=200,   # critical for chebyshev consistency
):
    if random_seed is not None:
        np.random.seed(random_seed)

    tx, wx_std, ty, wy_std, x_quad, y_quad, w_x, w_y = get_basis_interpolation(
        N_quad, domain=domain, basis_type=basis_type
    )
    X, Y = np.meshgrid(x_quad, y_quad, indexing="ij")

    if coeff_sampler is None:
        def coeff_sampler(n, m):
            return np.random.uniform(-1.0, 1.0) * decay_calculator(decay_type, decay_rate, n, m)

    dom2d = [list(domain[0]), list(domain[1])]

    Lx = np.stack(
        [get_basis_function_1d(n, N_basis, basis_type, dom2d[0], N_quad_for_orth)(x_quad) for n in range(N_basis)],
        axis=0
    )
    Ly = np.stack(
        [get_basis_function_1d(m, N_basis, basis_type, dom2d[1], N_quad_for_orth)(y_quad) for m in range(N_basis)],
        axis=0
    )

    coeffs = np.zeros((batch_size, N_basis, N_basis))
    f_grids = np.zeros((batch_size, N_quad, N_quad))
    for i in range(batch_size):
        C = np.array([[coeff_sampler(n, m) for m in range(N_basis)] for n in range(N_basis)])
        f = np.zeros((N_quad, N_quad))
        for n in range(N_basis):
            for m in range(N_basis):
                f += C[n, m] * np.outer(Lx[n], Ly[m])
        coeffs[i] = C
        f_grids[i] = f

    return coeffs, f_grids, X, Y, (w_x, w_y)


# ---------- Sampling: evaluate on physical nodes x_quad,y_quad ----------
def decay_calculator(decay_type, decay_rate, n, m):
    if decay_type == "none":
        return 1.0
    elif decay_type == "exponential":
        return np.exp(-decay_rate * ((n + m)**2))
    elif decay_type == "polynomial":
        return 1.0 / ((1 + n + m) ** decay_rate)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

# def sample_random_function_from_basis_2d(
#     N_basis=3,
#     N_quad=50,
#     coeff_sampler=None,
#     basis_type="legendre",
#     batch_size=1,
#     domain=[[0, 1], [0, 1]],
#     decay_type="exponential",
#     decay_rate=0.1,
#     random_seed=None
# ):
#     if random_seed is not None:
#         np.random.seed(random_seed)

#     tx, wx_std, ty, wy_std, x_quad, y_quad, w_x, w_y = get_basis_interpolation(
#         N_quad, domain=domain, basis_type=basis_type
#     )
#     X, Y = np.meshgrid(x_quad, y_quad, indexing="ij")

#     if coeff_sampler is None:
#         def coeff_sampler(n, m):
#             return np.random.uniform(-1.0, 1.0) * decay_calculator(decay_type, decay_rate, n, m)

#     # Evaluate basis on physical quadrature nodes
#     Lx = np.stack([get_basis_function(n, N_basis, basis_type, domain)(x_quad) for n in range(N_basis)], axis=0)
#     Ly = np.stack([get_basis_function(m, N_basis, basis_type, domain)(y_quad) for m in range(N_basis)], axis=0)

#     coeffs = np.zeros((batch_size, N_basis, N_basis))
#     f_grids = np.zeros((batch_size, N_quad, N_quad))
#     for i in range(batch_size):
#         coeff = np.array([[coeff_sampler(n, m) for m in range(N_basis)] for n in range(N_basis)])
#         f_grid = np.zeros((N_quad, N_quad))
#         for n in range(N_basis):
#             for m in range(N_basis):
#                 f_grid += coeff[n, m] * np.outer(Lx[n], Ly[m])
#         coeffs[i] = coeff
#         f_grids[i] = f_grid

#     return coeffs, f_grids, X, Y, (w_x, w_y)
