import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, griddata
import scipy.special
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
# TODO: projection for 1d data 
# simple test data for main model precision
from util_expansion import get_basis_interpolation,\
                                get_basis_function,\
                                decay_calculator
from L2_series_expansion_2d import project_on_L2_basis_2d

def sample_random_function_from_basis_2d(
    N_basis=3,
    N_quad=50,
    coeff_sampler=None,
    basis_type="legendre",
    batch_size=1,             # number of samples to generate
    domain=[[0, 1], [0, 1]],
    decay_type="exponential",  # "none", "exponential", "polynomial"
    decay_rate=0.1,            # controls how fast high-frequency terms decay
    random_seed=None
):
    """
    Sample a random 2D function as a linear combination of basis functions, with optional coefficient decay.

    Parameters:
        N_basis : int – number of basis functions in each direction
        N_quad : int – number of quadrature points per dimension
        coeff_sampler : callable or None – function (n, m) -> c_{n,m};
            if None, a decayed standard normal is used
        basis_type : str – type of orthonormal basis (e.g. "legendre", "chebyshev")
        domain : list – 2D domain [[a, b], [c, d]] for evaluation grid
        decay_type : str – one of "none", "exponential", "polynomial"
        decay_rate : float – controls decay speed
        random_seed : int or None – for reproducibility

    Returns:
        coeffs : ndarray (N_basis, N_basis)
        f_grid : ndarray (N_quad, N_quad)
        X, Y : ndarray (N_quad, N_quad)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # [a, b], [c, d] = domain

    # Quadrature grid
    x_std, _, y_std, _, x_quad, y_quad, _, _ = get_basis_interpolation(
        N_quad, domain=domain, basis_type=basis_type
    )
    X, Y = np.meshgrid(x_quad, y_quad, indexing="ij")

    # Define coefficient decay if no sampler is provided

    if coeff_sampler is None:
        def coeff_sampler(n, m):
            return np.random.uniform(-1.0, 1.0) * decay_calculator(decay_type, decay_rate, n, m)
    # NOTE not efficient
    # NOTE we either use decay calculator here or use weight decay in feature sampling. 
    # Sample coefficients
    coeffs = np.zeros((batch_size, N_basis, N_basis))
    f_grids = np.zeros((batch_size, N_quad, N_quad))
    for i in range(batch_size):
        coeff = np.array([[coeff_sampler(n, m) for m in range(N_basis)] for n in range(N_basis)])

        # Evaluate basis
        Lx = np.array([[get_basis_function(n, N_basis, basis_type)(x) for x in x_std] for n in range(N_basis)])
        Ly = np.array([[get_basis_function(m, N_basis, basis_type)(y) for y in y_std] for m in range(N_basis)])

        # Combine basis and coefficients
        f_grid = np.zeros((N_quad, N_quad))
        for n in range(N_basis):
            for m in range(N_basis):
                f_grid += coeff[n, m] * np.outer(Lx[n], Ly[m])
        coeffs[i] = coeff
        f_grids[i] = f_grid

    return coeffs, f_grids, X, Y


coeffs, f_grid, X, Y = sample_random_function_from_basis_2d(
    N_basis=3,
    N_quad=64,
    decay_type="exponential",
    decay_rate=0.1,
    basis_type="legendre",
    batch_size=10, 
    random_seed=42
)


def test_basis_projection_recovery(
    N_basis=5,
    N_quad=64,
    basis_type="legendre",
    domain=[[0, 1], [0, 1]],
    decay_type="exponential",
    decay_rate=0.1,
    tol=1e-6,
    verbose=True
):
    """
    Test that coefficients used to create a function via basis expansion can be recovered
    by projecting the function back onto the same basis.

    Returns:
        passed : bool – whether the test passed
        rel_error : float – relative error in coefficient recovery
    """

    # Step 1: Sample known function with controlled coefficients

    def coeff_sampler(n, m):
        return np.random.randn() * decay_calculator(decay_type, decay_rate, n, m)

    coeffs_true, f_grid, X, Y = sample_random_function_from_basis_2d(
        N_basis=N_basis,
        N_quad=N_quad,
        coeff_sampler=coeff_sampler,
        basis_type=basis_type,
        batch_size=10,  # Only need one sample for testing
        domain=domain
    )
    coeffs_recovered = []
    for i in range(coeffs_true.shape[0]):
        # Step 2: Flatten data for projection function
        x_data = X.flatten()
        y_data = Y.flatten()
        f_data = f_grid[i].flatten()

        # Step 3: Project the function back to the basis
        coeff_recovered, _, _, _, _ = project_on_L2_basis_2d(
            x_data, y_data, f_data,
            N_basis=N_basis,
            N_quad=N_quad,
            basis_type=basis_type,
            space='L2'
        )
        coeffs_recovered.append(coeff_recovered)
    # Step 4: Compare original and recovered coefficients
    rel_error = np.linalg.norm(coeffs_true - coeffs_recovered) / (np.linalg.norm(coeffs_true) + 1e-12)
    passed = rel_error < tol

    if verbose:
        print("Coefficient recovery test:")
        print(f"  Relative error = {rel_error:.2e}")
        print(f"  Test passed: {passed}")
    
    plt.imshow(np.average(coeffs_true - coeffs_recovered, axis=0), cmap='bwr')
    plt.colorbar()
    plt.title("Coefficient Error Matrix")

    return passed, rel_error


import numpy as np
from util_expansion import get_basis_interpolation, get_basis_function, decay_calculator

def sample_random_function_from_basis_1d(
    N_basis=8,
    N_quad=256,
    basis_type="legendre",
    domain=[0, 1],
    decay_type="polynomial",
    decay_rate=0.5,
    batch_size=1,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Reuse your interpolation util; if it's strictly 2D, replace with linspace.
    x_std, _, _, _, x_quad, _, _, _ = get_basis_interpolation(
        N_quad, domain=[domain, domain], basis_type=basis_type
    )
    # If get_basis_interpolation returns 2D-specific things, the robust fallback is:
    # x_std = np.linspace(-1, 1, N_quad) or mapped domain version
    # x_quad = np.linspace(domain[0], domain[1], N_quad)

    # Basis values: (N_basis, N_quad)
    Phi = np.array([[get_basis_function(n, N_basis, basis_type)(x) for x in x_std]
                    for n in range(N_basis)], dtype=float)

    # Decay weights: (N_basis,)
    w = np.array([abs(decay_calculator(decay_type, decay_rate, n, 0)) for n in range(N_basis)], dtype=float)

    coeffs = np.zeros((batch_size, N_basis), dtype=float)
    f_grids = np.zeros((batch_size, N_quad), dtype=float)

    for i in range(batch_size):
        c = rng.uniform(-1.0, 1.0, size=(N_basis,)) * w
        coeffs[i] = c
        f_grids[i] = c @ Phi  # (N_basis,) @ (N_basis, N_quad) -> (N_quad,)

    return coeffs, f_grids, x_quad


# test_basis_projection_recovery(N_basis=8, N_quad=64, decay_rate=0.2)



# plt.contourf(X, Y, f_grid[0], levels=30, cmap="viridis")
# plt.colorbar()
# plt.title("Random Function with Exponential Coefficient Decay")
# plt.show()
