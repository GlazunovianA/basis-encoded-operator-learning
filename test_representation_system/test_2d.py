import numpy as np
import pytest

from test_expansion_refactored.basis_function import get_basis_function_1d, get_legendre_quad_on_interval
from ed_2d import (
    encode_field_l2_2d,
    decode_field_l2_2d,
    project_on_L2_basis_2d,
    evaluate_series_L2,
)


def eval_basis_1d(N_basis, nodes, basis_type, interval, N_quad_for_orth=200):
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


def gram_matrix(B, w):
    return (B * w[None, :]) @ B.T


def project_coeffs_2d(f_grid, Bx, By, wx, wy):
    weighted_f = (wx[:, None] * f_grid) * wy[None, :]
    return Bx @ weighted_f @ By.T


def reconstruct_2d(C, Bx, By):
    return (Bx.T @ C) @ By


@pytest.mark.parametrize(
    "basis_type,N_basis,N_quad,N_quad_for_orth,tol",
    [
        ("legendre", 6, 96, 200, 1e-12),
        ("cosine", 8, 256, 200, 1e-11),
        ("sine", 8, 256, 200, 1e-11),
        ("fourier", 8, 512, 200, 1e-11),
        ("chebyshev", 6, 256, 256, 1e-9),
    ],
)
def test_1d_basis_is_orthonormal_on_physical_interval(
    basis_type, N_basis, N_quad, N_quad_for_orth, tol
):
    a, b = -1.3, 2.4
    _, _, x, wx = get_legendre_quad_on_interval(N_quad, a, b)

    B = eval_basis_1d(
        N_basis=N_basis,
        nodes=x,
        basis_type=basis_type,
        interval=[a, b],
        N_quad_for_orth=N_quad_for_orth,
    )
    G = gram_matrix(B, wx)
    I = np.eye(N_basis)

    rel_err = np.linalg.norm(G - I, ord="fro") / np.linalg.norm(I, ord="fro")
    assert rel_err < tol


@pytest.mark.parametrize(
    "basis_type,N_basis,N_quad,N_quad_for_orth,tol",
    [
        ("legendre", 5, 128, 200, 1e-11),
        ("cosine", 6, 256, 200, 1e-10),
        ("sine", 6, 256, 200, 1e-10),
        ("fourier", 6, 512, 200, 1e-10),
        ("chebyshev", 5, 256, 256, 1e-8),
    ],
)
def test_tensor_projection_reconstruction_identity(
    basis_type, N_basis, N_quad, N_quad_for_orth, tol
):
    domain = [[-0.8, 1.7], [0.3, 2.1]]

    _, _, x, wx = get_legendre_quad_on_interval(N_quad, *domain[0])
    _, _, y, wy = get_legendre_quad_on_interval(N_quad, *domain[1])

    Bx = eval_basis_1d(N_basis, x, basis_type, domain[0], N_quad_for_orth)
    By = eval_basis_1d(N_basis, y, basis_type, domain[1], N_quad_for_orth)

    rng = np.random.default_rng(123)
    C_true = rng.standard_normal((N_basis, N_basis))

    f = reconstruct_2d(C_true, Bx, By)
    C_hat = project_coeffs_2d(f, Bx, By, wx, wy)

    rel_err = np.linalg.norm(C_hat - C_true) / np.linalg.norm(C_true)
    assert rel_err < tol


@pytest.mark.parametrize(
    "basis_type,N_basis,N_quad,N_quad_for_orth,tol",
    [
        ("legendre", 5, 128, 200, 1e-11),
        ("cosine", 6, 256, 200, 1e-10),
        ("sine", 6, 256, 200, 1e-10),
        ("fourier", 6, 512, 200, 1e-10),
        ("chebyshev", 5, 256, 256, 1e-8),
    ],
)
def test_parseval_identity_for_tensor_product_basis(
    basis_type, N_basis, N_quad, N_quad_for_orth, tol
):
    domain = [[0.0, 1.0], [0.0, 1.0]]

    _, _, x, wx = get_legendre_quad_on_interval(N_quad, *domain[0])
    _, _, y, wy = get_legendre_quad_on_interval(N_quad, *domain[1])

    Bx = eval_basis_1d(N_basis, x, basis_type, domain[0], N_quad_for_orth)
    By = eval_basis_1d(N_basis, y, basis_type, domain[1], N_quad_for_orth)

    rng = np.random.default_rng(7)
    C = rng.standard_normal((N_basis, N_basis))
    f = reconstruct_2d(C, Bx, By)

    f_norm_sq = np.sum((wx[:, None] * (f ** 2)) * wy[None, :])
    c_norm_sq = np.sum(C ** 2)

    rel_err = abs(f_norm_sq - c_norm_sq) / abs(f_norm_sq)
    assert rel_err < tol


@pytest.mark.parametrize(
    "basis_type,N_basis,N_quad,N_quad_for_orth,tol",
    [
        ("legendre", 5, 128, 200, 1e-11),
        ("cosine", 6, 256, 200, 1e-10),
        ("sine", 6, 256, 200, 1e-10),
        ("fourier", 6, 512, 200, 1e-10),
        ("chebyshev", 5, 256, 256, 1e-8),
    ],
)
def test_decode_matches_direct_tensor_reconstruction(
    basis_type, N_basis, N_quad, N_quad_for_orth, tol
):
    domain = [[-1.0, 0.5], [0.2, 1.9]]

    _, _, x, _ = get_legendre_quad_on_interval(N_quad, *domain[0])
    _, _, y, _ = get_legendre_quad_on_interval(N_quad, *domain[1])

    Bx = eval_basis_1d(N_basis, x, basis_type, domain[0], N_quad_for_orth)
    By = eval_basis_1d(N_basis, y, basis_type, domain[1], N_quad_for_orth)

    rng = np.random.default_rng(77)
    C = rng.standard_normal((N_basis, N_basis))

    f_direct = reconstruct_2d(C, Bx, By)
    f_decoded, Xq, Yq, meta = decode_field_l2_2d(
        coeffs=C,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )

    rel_err = np.linalg.norm(f_decoded - f_direct) / np.linalg.norm(f_direct)
    assert rel_err < tol
    assert Xq.shape == (N_quad, N_quad)
    assert Yq.shape == (N_quad, N_quad)
    assert meta["basis_type"] == basis_type


@pytest.mark.parametrize(
    "basis_type,N_basis,N_quad,N_quad_for_orth,tol",
    [
        ("legendre", 5, 128, 200, 1e-11),
        ("cosine", 6, 256, 200, 1e-10),
        ("sine", 6, 256, 200, 1e-10),
        ("fourier", 6, 512, 200, 1e-10),
        ("chebyshev", 5, 256, 256, 1e-8),
    ],
)
def test_encode_recovers_coefficients_from_exact_tensor_field(
    basis_type, N_basis, N_quad, N_quad_for_orth, tol
):
    domain = [[0.0, 1.0], [0.0, 1.0]]

    _, _, x, _ = get_legendre_quad_on_interval(N_quad, *domain[0])
    _, _, y, _ = get_legendre_quad_on_interval(N_quad, *domain[1])

    Bx = eval_basis_1d(N_basis, x, basis_type, domain[0], N_quad_for_orth)
    By = eval_basis_1d(N_basis, y, basis_type, domain[1], N_quad_for_orth)

    rng = np.random.default_rng(999)
    C_true = rng.standard_normal((N_basis, N_basis))
    f_grid = reconstruct_2d(C_true, Bx, By)

    Xq, Yq = np.meshgrid(x, y, indexing="ij")

    C_hat, f_rec, _, _, f_quad, meta = encode_field_l2_2d(
        x_data=Xq.ravel(),
        y_data=Yq.ravel(),
        f_data=f_grid.ravel(),
        N_basis=N_basis,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
        interp_method="linear",
    )

    coeff_rel_err = np.linalg.norm(C_hat - C_true) / np.linalg.norm(C_true)
    recon_rel_err = np.linalg.norm(f_rec - f_grid) / np.linalg.norm(f_grid)

    assert coeff_rel_err < tol
    assert recon_rel_err < tol
    assert np.allclose(f_quad, f_grid, atol=1e-12, rtol=1e-12)
    assert meta["basis_type"] == basis_type


@pytest.mark.parametrize("basis_type", ["legendre", "chebyshev", "cosine", "sine", "fourier"])
def test_backward_compatible_wrappers_match_core_functions(basis_type):
    domain = [[0.0, 1.0], [0.0, 1.0]]
    N_basis = 4
    N_quad = 40
    N_quad_for_orth = 128 if basis_type == "chebyshev" else 200

    rng = np.random.default_rng(1234)
    x = rng.uniform(domain[0][0], domain[0][1], size=500)
    y = rng.uniform(domain[1][0], domain[1][1], size=500)
    f = np.sin(np.pi * x) * np.cos(2.0 * np.pi * y)

    coeffs_core, f_rec_core, Xq_core, Yq_core, f_quad_core, _ = encode_field_l2_2d(
        x_data=x,
        y_data=y,
        f_data=f,
        N_basis=N_basis,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )

    coeffs_wrap, f_rec_wrap, Xq_wrap, Yq_wrap, f_quad_wrap = project_on_L2_basis_2d(
        x_data=x,
        y_data=y,
        f_data=f,
        N_basis=N_basis,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )

    f_dec_wrap = evaluate_series_L2(
        coeffs_wrap,
        N_quad=N_quad,
        basis_type=basis_type,
        domain=domain,
        N_quad_for_orth=N_quad_for_orth,
    )

    assert np.allclose(coeffs_wrap, coeffs_core)
    assert np.allclose(f_rec_wrap, f_rec_core)
    assert np.allclose(Xq_wrap, Xq_core)
    assert np.allclose(Yq_wrap, Yq_core)
    assert np.allclose(f_quad_wrap, f_quad_core)
    assert np.allclose(f_dec_wrap, f_rec_core)