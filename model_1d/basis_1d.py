import numpy as np

# -----------------------------
# Periodic Fourier basis on [0,1)
# Orthonormal in L2([0,1]) w.r.t. Lebesgue measure.
# phi_0(x) = 1
# phi_{2k-1}(x) = sqrt(2) * cos(2π k x)
# phi_{2k}(x)   = sqrt(2) * sin(2π k x)
# -----------------------------

# tested. passed. 


def fourier_periodic_phi(j: int):
    if j == 0:
        return lambda x: np.ones_like(x)
    if j % 2 == 1:  # cos
        k = (j + 1) // 2
        return lambda x: np.sqrt(2.0) * np.cos(2.0 * np.pi * k * x)
    else:  # sin
        k = j // 2
        return lambda x: np.sqrt(2.0) * np.sin(2.0 * np.pi * k * x)

def eval_basis_matrix(basis: str, N_basis: int, x: np.ndarray) -> np.ndarray:
    """
    Returns Phi of shape (N_basis, len(x)) with Phi[j,i] = phi_j(x_i).
    """
    if basis == "fourier_periodic":
        Phi = np.vstack([fourier_periodic_phi(j)(x) for j in range(N_basis)])
        return Phi.astype(float)
    raise ValueError(f"Unknown basis={basis}")

def uniform_quad_on_01(N: int):
    """
    Quadrature nodes and weights for L2([0,1]) with periodic/trapezoidal rule.
    For smooth periodic functions, this is very accurate; for Fourier modes below Nyquist,
    it is effectively exact.
    """
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    w = np.full(N, 1.0 / N)
    return x, w

def gram_matrix(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Compute discrete Gram matrix G_{ij} = sum_k w_k Phi[i,k] Phi[j,k].
    """
    # Phi: (N_basis, N_pts), w: (N_pts,)
    return (Phi * w) @ Phi.T

def project_L2_from_samples(Phi: np.ndarray, w: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Given f(x_k) on the same x_k as Phi, compute coefficients via discrete L2 projection:
    c = G^{-1} b, where b_i = <f, phi_i>, G_ij = <phi_i, phi_j>.
    """
    G = gram_matrix(Phi, w)
    b = (Phi * w) @ f
    # Solve; if Phi is truly orthonormal under this quadrature, G ~ I
    return np.linalg.solve(G, b)

def reconstruct_from_coeffs(Phi: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    f_hat(x_k) = sum_j c_j phi_j(x_k).
    """
    return c @ Phi
