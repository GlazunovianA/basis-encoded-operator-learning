import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, griddata
import scipy.special
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
# TODO: projection for 1d data 
# simple test data for main model precision
from util_expansion import get_basis_interpolation,\
                                get_basis_function





# projection on 2d normalized legendre basis 
def project_on_L2_basis_2d(x_data, y_data, f_data, N_basis = 5, N_quad=50, basis_type="legendre", space = 'L2'):
    """
    Project real data sampled on a square domain [a,b] x [c,d] onto Legendre basis functions.

    Parameters:
        f_data : ndarray (Nx*Ny,) – sampled data
        x_data : ndarray (Nx*Ny,) – grid points in x
        y_data : ndarray (Nx*Ny,) – grid points in y
        N_basis_x : int – number of Legendre basis functions in x
        N_basis_y : int – number of Legendre basis functions in y
        N_quad : int – number of quadrature points in each dimension

    Returns:
        coeffs : ndarray (N_basis_x, N_basis_y)
        f_reconstructed : ndarray (N_quad, N_quad)
        rel_error : ndarray (N_quad, N_quad)
        X_quad, Y_quad : quadrature grid points (for plotting)
    """

    # Get domain bounds
    a, b = np.min(x_data), np.max(x_data)
    c, d = np.min(y_data), np.max(y_data)

    # Interpolator from real data
    x_std, w_x, y_std, w_y, x_quad, y_quad, w_x_quad, w_y_quad = get_basis_interpolation(N_quad, domain = list([[a, b], [c, d]]), basis_type=basis_type)
    # Map to [a, b] and [c, d]
  
    scaling = np.outer(w_x, w_y)
    # print(scaling)
    X_quad, Y_quad = np.meshgrid(x_quad, y_quad, indexing='ij')
    x_query = X_quad.flatten()
    y_query = Y_quad.flatten()
    query = np.column_stack([x_query, y_query])

    f_quad = griddata(np.column_stack((x_data, y_data)), f_data, query, method= 'linear')
    f_quad = f_quad.reshape(X_quad.shape)
    # Compute basis at standard points
    Lx = np.array([[get_basis_function(n, N_basis, basis_type=basis_type)(x) for x in x_std] for n in range(N_basis)])
    Ly = np.array([[get_basis_function(m, N_basis, basis_type=basis_type)(y) for y in y_std] for m in range(N_basis)])

    # Project using inner product
    coeffs = np.zeros((N_basis, N_basis))
    for n in range(N_basis):
        for m in range(N_basis):
            basis_product = np.outer(Lx[n], Ly[m])
            if space == 'L2':
                integral = np.sum(f_quad * basis_product * scaling)
            # if space == 'H1' or space == 'H_0_1':
            #     df_dx = np.gradient(f_quad, axis=0) / dx
            #     df_dy = np.gradient(f_quad, axis=1) / dy

            #     # Basis function derivatives
            #     dphi_dx = np.gradient(phi_n(x), dx)
            #     dpsi_dy = np.gradient(psi_m(y), dy)

            #     # Full projection:
            #     c_nm = np.sum(df_dx * dphi_dx[:, None] * psi_m[None, :] * weights)
            #         + np.sum(df_dy * phi_n[:, None] * dpsi_dy[None, :] * weights)
            else:
                raise NotImplementedError(f"Projection space '{space}' not implemented.")
            coeffs[n, m] = integral # * (b - a)/2 * (d - c)/2  # rescaled
    # print(coeffs.shape)

    # Reconstruct function on mapped grid
    f_reconstructed = np.zeros_like(f_quad)
    for n in range(N_basis):
        Pn_x = get_basis_function(n, N_basis, basis_type=basis_type)(x_std)
        for m in range(N_basis):
            Pm_y = get_basis_function(m, N_basis, basis_type=basis_type)(y_std)
            f_reconstructed += coeffs[n, m] * np.outer(Pn_x, Pm_y)
    f_compare = evaluate_series_L2(coeffs, N_quad, basis_type=basis_type)
    # Compute error
    # rel_error = np.abs((f_quad - f_compare) / np.maximum(np.abs(f_quad), 1e-12))

    return coeffs, f_compare, X_quad, Y_quad, f_quad

def evaluate_series_L2(coeffs, N_quad=50, basis_type="legendre"):
    ''' 
    Evaluate a 2D Legendre series given the coefficients and the number of quadrature points. 
    
    Parameters:     
        coeffs : ndarray (N_basis_x, N_basis_y) – coefficients of the Legendre series
        N_quad : int – number of quadrature points in each dimension
    ''' 
    
    N_basis_x, N_basis_y = coeffs.shape
    x_std, w_x, y_std, w_y = get_basis_interpolation(N_quad, domain = None, basis_type=basis_type)
    f_reconstructed = np.zeros((len(x_std), len(y_std)))
    
    for n in range(N_basis_x):
        Pn_x = get_basis_function(n, N_basis_x, basis_type=basis_type)(x_std)
        for m in range(N_basis_y):
            Pm_y = get_basis_function(m, N_basis_y, basis_type=basis_type)(y_std)
            f_reconstructed += coeffs[n, m] * np.outer(Pn_x, Pm_y)

    return f_reconstructed


# equation = 'sine_gordon'
# err = []
# for basis in ['legendre', 'chebyshev', 'sine']: # these work
#     for i in range(100):
#         # # Try it on a non-[-1,1] domain
#         # x_data = np.linspace(-1, 1, 100)
#         # y_data = np.linspace(-1, 1, 100)
#         # X_data, Y_data = np.meshgrid(x_data, y_data, indexing='ij')
#         # # f_data = np.exp(-((X_data - 2)**2 + Y_data**2))
#         # # f_data = np.sin(np.pi * X_data) * np.cos(np.pi * Y_data)
#         # f_data = np.fmax(np.cos((X_data**2 + Y_data**2)*np.pi/2), np.zeros_like(X_data))
#         array_sol = np.load(f'data/{equation}/fem_solution_{i}.npy')
#         X_data, Y_data, f_data = array_sol[:, 0], array_sol[:, 1], array_sol[:, 2]
#         # print(f_data.mean())
#         # Run the projection
#         coeffs, f_rec, Xq, Yq, fq = project_on_L2_basis_2d(X_data.flatten(), 
#                                                                         Y_data.flatten(), 
#                                                                         f_data.flatten(), 
#                                                                         N_basis=10, 
#                                                                     basis_type=basis,)
#         err.append(np.linalg.norm(fq - f_rec))
#     print('MSE', np.array(err).mean())
#     # Plot
#     fig = plt.figure(figsize=(18, 5))
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.plot_surface(Xq, Yq, fq, cmap='viridis')
#     ax1.set_title('Original Function')

#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.plot_surface(Xq, Yq, f_rec, cmap='viridis')
#     ax2.set_title('Reconstructed')

#     ax3 = fig.add_subplot(133, projection='3d')
#     ax3.plot_surface(Xq, Yq, np.abs(fq - f_rec), cmap='inferno')
#     ax3.set_title('Relative Error')

#     plt.tight_layout()
#     plt.show()


