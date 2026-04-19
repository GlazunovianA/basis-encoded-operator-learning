import numpy as np
from scipy.interpolate import griddata

from test_expansion_refactored.basis_function import (
    get_legendre_quad_on_interval,
    get_basis_function_1d, get_basis_interpolation
)


def _eval_basis_1d(
    N_basis: int,
    nodes: np.ndarray,
    basis_type: str,
    interval: list[float] | tuple[float, float],
    N_quad_for_orth: int = 200,
) -> np.ndarray:
    """
    Evaluate a 1D orthonormal basis on physical nodes.

    Returns
    -------
    B : ndarray, shape (N_basis, N_nodes)
        Row n contains phi_n(nodes).
    """
    return np.stack(
        [
            get_basis_function_1d(
                n,
                N_basis,
                basis_type,
                interval,
                N_quad_for_orth=N_quad_for_orth,
            )(nodes)
            for n in range(N_basis)
        ],
        axis=0,
    )


def _project_coeffs_2d(
    f_grid: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> np.ndarray:
    """
    Orthogonal projection in tensor-product Lebesgue L2 quadrature.

    Parameters
    ----------
    f_grid : ndarray, shape (Nx, Ny)
        Function values on the tensor-product physical quadrature grid.
    Bx : ndarray, shape (N_basis_x, Nx)
    By : ndarray, shape (N_basis_y, Ny)
    wx : ndarray, shape (Nx,)
    wy : ndarray, shape (Ny,)

    Returns
    -------
    C : ndarray, shape (N_basis_x, N_basis_y)
        Expansion coefficients.
    """
    weighted_f = (wx[:, None] * f_grid) * wy[None, :]
    return Bx @ weighted_f @ By.T


def _reconstruct_from_coeffs_2d(
    coeffs: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a tensor-product field from coefficients and basis matrices.

    Parameters
    ----------
    coeffs : ndarray, shape (N_basis_x, N_basis_y)
    Bx : ndarray, shape (N_basis_x, Nx)
    By : ndarray, shape (N_basis_y, Ny)

    Returns
    -------
    f_grid : ndarray, shape (Nx, Ny)
    """
    return (Bx.T @ coeffs) @ By


# def _prepare_physical_grid(
#     domain: list[list[float]] | tuple[tuple[float, float], tuple[float, float]],
#     N_quad: int,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Build tensor-product Gauss-Legendre quadrature on the physical rectangle.
#     """
#     (ax, bx), (ay, by) = domain

#     _, _, x_quad, wx = get_legendre_quad_on_interval(N_quad, ax, bx)
#     _, _, y_quad, wy = get_legendre_quad_on_interval(N_quad, ay, by)

#     Xq, Yq = np.meshgrid(x_quad, y_quad, indexing="ij")
#     return x_quad, y_quad, wx, wy, Xq, Yq


def encode_field_l2_2d(
    x_data: np.ndarray,
    y_data: np.ndarray,
    f_data: np.ndarray,
    N_basis: int,
    basis_type: str = "legendre",
    domain: list[list[float]] | None = None,
    N_quad: int = 80,
    N_quad_for_orth: int = 200,
    interp_method: str = "linear",
):
    """
    Encode scattered/raw 2D field data into tensor-product L2 coefficients.

    This is the replacement for the old projection function.

    Parameters
    ----------
    x_data, y_data, f_data : 1D arrays of equal length
        Scattered or flattened field samples on a rectangle.
    N_basis : int
        Number of modes in each direction.
    basis_type : str
        One of the supported L2-orthonormal basis families in basis_function.py.
    domain : [[ax,bx],[ay,by]] or None
        If None, inferred from min/max of input coordinates.
    N_quad : int
        Quadrature/interpolation grid size per dimension.
    N_quad_for_orth : int
        Auxiliary size for basis orthonormalization when needed
        (notably the numerically orthonormalized Chebyshev branch).
    interp_method : str
        Method passed to scipy.interpolate.griddata.

    Returns
    -------
    coeffs : ndarray, shape (N_basis, N_basis)
    f_rec : ndarray, shape (N_quad, N_quad)
        Reconstruction on the physical quadrature grid.
    Xq, Yq : ndarray, shape (N_quad, N_quad)
        Physical quadrature grid.
    f_quad : ndarray, shape (N_quad, N_quad)
        Input field interpolated onto the same quadrature grid.
    meta : dict
        Metadata required for stable downstream decoding.
    """
    x_data = np.asarray(x_data, dtype=float).ravel()
    y_data = np.asarray(y_data, dtype=float).ravel()
    f_data = np.asarray(f_data, dtype=float).ravel()

    if not (len(x_data) == len(y_data) == len(f_data)):
        raise ValueError("x_data, y_data, f_data must have the same length")

    if domain is None:
        domain = [
            [float(np.min(x_data)), float(np.max(x_data))],
            [float(np.min(y_data)), float(np.max(y_data))],
        ]

    x_quad, y_quad, wx, wy, Xq, Yq = get_basis_interpolation(
       domain, N_quad, basis_type=basis_type
    )
    # _prepare_physical_grid(domain, N_quad)

    query = np.column_stack([Xq.ravel(), Yq.ravel()])
    points = np.column_stack([x_data, y_data])

    # # Skip griddata if input points already match the quadrature grid exactly
    # if points.shape == query.shape and np.allclose(points, query, atol=1e-12):
    #     f_quad = f_data.reshape(N_quad, N_quad)
    # else:
    #     f_quad = griddata(points, f_data, query, method=interp_method)
    #     if np.any(np.isnan(f_quad)):
    #         nan_mask = np.isnan(f_quad)
    #         f_quad_nearest = griddata(points, f_data, query, method="nearest")
    #         f_quad[nan_mask] = f_quad_nearest[np.isnan(f_quad)]
    #     f_quad = f_quad.reshape(N_quad, N_quad)
    
    # Sort both by (x,y) before comparing to handle ravel-order mismatches
    def _sort_points(p): return p[np.lexsort((p[:,1], p[:,0]))]
    _q = _sort_points(query)
    _p = _sort_points(points)
    if _p.shape == _q.shape and np.allclose(_p, _q, atol=1e-12):
        # input is already on the quadrature grid; reshape directly
        # need to sort f_data to match Xq.ravel() order too
        sort_idx = np.lexsort((y_data, x_data))
        # does this make sense? TODO
        f_quad = f_data[sort_idx].reshape(N_quad, N_quad)
    else:
        f_quad = griddata(points, f_data, query, method=interp_method)
        if np.any(np.isnan(f_quad)):
            nan_mask = np.isnan(f_quad)
            f_quad_nearest = griddata(points, f_data, query, method="nearest")
            f_quad[nan_mask] = f_quad_nearest[nan_mask]
        f_quad = f_quad.reshape(N_quad, N_quad)

    if np.any(np.isnan(f_quad)):
        # linear interpolation may leave NaNs near the convex-hull boundary;
        # fill them robustly using nearest-neighbor interpolation
        f_quad_nearest = griddata(points, f_data, query, method="nearest")
        nan_mask = np.isnan(f_quad)
        f_quad[nan_mask] = f_quad_nearest[nan_mask]

    f_quad = f_quad.reshape(N_quad, N_quad)

    Bx = _eval_basis_1d(
        N_basis=N_basis,
        nodes=x_quad,
        basis_type=basis_type,
        interval=domain[0],
        N_quad_for_orth=N_quad_for_orth,
    )
    By = _eval_basis_1d(
        N_basis=N_basis,
        nodes=y_quad,
        basis_type=basis_type,
        interval=domain[1],
        N_quad_for_orth=N_quad_for_orth,
    )

    coeffs = _project_coeffs_2d(f_quad, Bx, By, wx, wy)
    f_rec = _reconstruct_from_coeffs_2d(coeffs, Bx, By)

    meta = {
        "domain": [list(domain[0]), list(domain[1])],
        "basis_type": basis_type,
        "N_basis": int(N_basis),
        "N_quad": int(N_quad),
        "N_quad_for_orth": int(N_quad_for_orth),
        "x_quad": x_quad,
        "y_quad": y_quad,
        "wx": wx,
        "wy": wy,
    }

    return coeffs, f_rec, Xq, Yq, f_quad, meta


def decode_field_l2_2d(
    coeffs: np.ndarray,
    basis_type: str = "legendre",
    domain: list[list[float]] | None = None,
    N_quad: int = 80,
    N_quad_for_orth: int = 200,
):
    """
    Decode/reconstruct a coefficient tensor into a field on a physical quadrature grid.

    This is the replacement for the old evaluate_series_L2 function.

    Parameters
    ----------
    coeffs : ndarray, shape (N_basis_x, N_basis_y)
    basis_type : str
    domain : [[ax,bx],[ay,by]] or None
        Defaults to [[0,1],[0,1]] if not given.
    N_quad : int
        Number of physical quadrature nodes per dimension for evaluation.
    N_quad_for_orth : int
        Auxiliary size for basis orthonormalization when needed.

    Returns
    -------
    f_rec : ndarray, shape (N_quad, N_quad)
    Xq, Yq : ndarray, shape (N_quad, N_quad)
    meta : dict
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.ndim != 2:
        raise ValueError("coeffs must be a 2D array")

    if domain is None:
        domain = [[0.0, 1.0], [0.0, 1.0]]

    N_basis_x, N_basis_y = coeffs.shape
    x_quad, y_quad, wx, wy, Xq, Yq = get_basis_interpolation(domain, N_quad, basis_type=basis_type)
    # _prepare_physical_grid(domain, N_quad)

    Bx = _eval_basis_1d(
        N_basis=N_basis_x,
        nodes=x_quad,
        basis_type=basis_type,
        interval=domain[0],
        N_quad_for_orth=N_quad_for_orth,
    )
    By = _eval_basis_1d(
        N_basis=N_basis_y,
        nodes=y_quad,
        basis_type=basis_type,
        interval=domain[1],
        N_quad_for_orth=N_quad_for_orth,
    )

    f_rec = _reconstruct_from_coeffs_2d(coeffs, Bx, By)

    meta = {
        "domain": [list(domain[0]), list(domain[1])],
        "basis_type": basis_type,
        "N_basis_x": int(N_basis_x),
        "N_basis_y": int(N_basis_y),
        "N_quad": int(N_quad),
        "N_quad_for_orth": int(N_quad_for_orth),
        "x_quad": x_quad,
        "y_quad": y_quad,
        "wx": wx,
        "wy": wy,
    }

    return f_rec, Xq, Yq, meta


# -------------------------------------------------------------------
# Compatibility wrappers with the old external format
# -------------------------------------------------------------------

def project_on_L2_basis_2d(
    x_data,
    y_data,
    f_data,
    N_basis=5,
    N_quad=80,
    basis_type="legendre",
    space="L2",
    domain=None,
    N_quad_for_orth=200,
    interp_method="linear",
):
    """
    Backward-compatible replacement for the old function.

    Same leading return structure:
        coeffs, f_rec, Xq, Yq, f_quad

    Optional metadata is intentionally omitted here to preserve the old
    unpacking style in legacy code. Use encode_field_l2_2d directly if
    you also want metadata.
    """
    if space != "L2":
        raise NotImplementedError("Only Lebesgue L2 is supported in this replacement.")

    coeffs, f_rec, Xq, Yq, f_quad, _ = encode_field_l2_2d(
        x_data=x_data,
        y_data=y_data,
        f_data=f_data,
        N_basis=N_basis,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
        interp_method=interp_method,
    )
    return coeffs, f_rec, Xq, Yq, f_quad

# TODO this defaults to domain [0, 1]^2. should be adjusted
# let encoding return domain size -- feed into D. 
def evaluate_series_L2(
    coeffs,
    N_quad=80,
    basis_type="legendre",
    domain=None,
    N_quad_for_orth=200,
):
    """
    Backward-compatible replacement for the old decoder.

    Returns only the reconstructed field, like the historical function.
    """
    f_rec, _, _, _ = decode_field_l2_2d(
        coeffs=coeffs,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )
    return f_rec