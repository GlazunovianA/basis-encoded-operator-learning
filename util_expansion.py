import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss
import scipy.special
# === General Basis Functions ===
def orthonormal_fourier(n):
    if n == 0:
        return lambda x: np.ones_like(x)
    elif n % 2 == 1:  # sine
        k = (n // 2) + 1
        return lambda x: np.sqrt(2) * np.sin(k * np.pi * x)
    else:  # cosine
        k = n // 2
        return lambda x: np.sqrt(2) * np.cos(k * np.pi * x)

def orthonormal_chebyshev(n):
    # not orthonormal 
    factor = np.sqrt(1/np.pi) if n == 0 else np.sqrt(2/np.pi)
    return lambda x: factor * scipy.special.chebyt(n)(x)

def orthonormal_legendre(n):
    return lambda x: np.sqrt((2*n + 1)/2) * scipy.special.legendre(n)(x)

def hat_basis(n, N):
    """Piecewise linear hat function centered at node n/N"""
    def phi(x):
        if n == 0:
            return np.sqrt(2*N)*np.where((x >= 0) & (x <= 1/N), 1 - N*x, 0)
        elif n == N:
            return np.sqrt(2*N)*np.where((x >= 1 - 1/N) & (x <= 1), N*x - (N-1), 0)
        else:
            return np.sqrt(N)*np.where((x >= (n-1)/N) & (x <= n/N), N*x - (n-1), 
                            np.where((x > n/N) & (x <= (n+1)/N), (n+1) - N*x, 0))
    return phi

# === Basis Functions for H01 ===
def h01_sine_basis(n):
    return lambda x: np.sqrt(2) * np.sin((n + 1) * np.pi * x)

def h01_sine_basis_derivative(n):
    return lambda x: np.sqrt(2) * (n + 1) * np.pi * np.cos((n + 1) * np.pi * x)


def get_legendre_quad(N):
    x, w = leggauss(N)
    return x, w

def get_chebyshev_quad(N):
    x, w = chebgauss(N)  # weights include the Chebyshev weight
    return x, w

def get_uniform_quad(N):
    x = np.linspace(0, 1, N, endpoint=False)
    w = np.full(N, 1.0 / N)
    return x, w




def get_basis_interpolation(N_quad, domain=None, basis_type="legendre"):
    
    basis_quad_registry = dict({
    "legendre": get_legendre_quad,
    "chebyshev": get_chebyshev_quad,
    "fourier": get_uniform_quad,
    "sine": get_uniform_quad,
    "hat": get_uniform_quad,
    "H0_legendre": get_legendre_quad,
})
    if domain:
        a, b, c, d = domain[0][0], domain[0][1], domain[1][0], domain[1][1]

    if basis_type not in basis_quad_registry.keys():
        raise NotImplementedError(f"Quadrature for basis '{basis_type}' not registered.")

    get_quad = basis_quad_registry[basis_type]

    # 1D quadrature
    x_std, w_x = get_quad(N_quad)
    y_std, w_y = get_quad(N_quad)
    # print(w_x, w_y)
    if domain:  
        # Map from standard interval to [a,b] and [c,d]
        x_quad = 0.5 * (x_std + 1) * (b - a) + a if basis_type not in ["fourier", "sine", "hat"] else (b - a) * x_std + a
        y_quad = 0.5 * (y_std + 1) * (d - c) + c if basis_type not in ["fourier", "sine", "hat"] else (d - c) * y_std + c

        # Rescale weights (Jacobian scaling)
        w_x_quad = 0.5 * (b - a) * w_x if basis_type not in ["fourier", "sine", "hat"] else (b - a) * w_x
        w_y_quad = 0.5 * (d - c) * w_y if basis_type not in ["fourier", "sine", "hat"] else (d - c) * w_y

        return x_std, w_x, y_std, w_y, x_quad, y_quad, w_x_quad, w_y_quad

    return x_std, w_x, y_std, w_y




# darcy flow: a: cosine/KL/Fourier

def get_basis_function(n, N_basis, basis_type="legendre"):
    if basis_type == "legendre":
        return orthonormal_legendre(n)
    elif basis_type == "chebyshev":
        return orthonormal_chebyshev(n)
    elif basis_type == "fourier":
        return orthonormal_fourier(n)
    elif basis_type == "sine":
        return h01_sine_basis(n)
    elif basis_type == "hat":
        return hat_basis(n, N_basis)    
    else:
        raise NotImplementedError(f"Unknown basis: {basis_type}")

def get_basis_derivative(n, basis_type="sine"):
    if basis_type == "sine":
        return h01_sine_basis_derivative(n)  
    else:
        raise NotImplementedError(f"Unknown basis: {basis_type}")


def decay_calculator(decay_type, decay_rate, n, m):
    if decay_type == "none":
        return 1.0
    elif decay_type == "exponential":
        return np.exp(-decay_rate * ((n + m)**2))
    elif decay_type == "polynomial": # decay as 1/(n+m)^decay_rate, n, m are x, y indices
        return 1.0 / ((1 + n + m) ** decay_rate)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    

    