import numpy as np
from basis_1d import uniform_quad_on_01, eval_basis_matrix, gram_matrix, project_L2_from_samples, reconstruct_from_coeffs

def test_fourier_periodic_orthonormality():
    N_basis = 17
    N_x = 4096  # large so trapezoidal is very accurate for these modes
    x, w = uniform_quad_on_01(N_x)
    Phi = eval_basis_matrix("fourier_periodic", N_basis, x)
    G = gram_matrix(Phi, w)

    I = np.eye(N_basis)
    err = np.linalg.norm(G - I, ord=np.inf)
    # this should be tiny for Fourier periodic with uniform quad if N_x is big enough
    assert err < 1e-3, f"Gram matrix too far from I, inf-norm err={err}"

def test_reconstruction_projection_consistency():
    rng = np.random.default_rng(0)
    N_basis = 17
    N_x = 4096
    x, w = uniform_quad_on_01(N_x)
    Phi = eval_basis_matrix("fourier_periodic", N_basis, x)

    c_true = rng.uniform(-1, 1, size=(N_basis,))
    f = reconstruct_from_coeffs(Phi, c_true)
    c_hat = project_L2_from_samples(Phi, w, f)

    rel = np.linalg.norm(c_hat - c_true) / (np.linalg.norm(c_true) + 1e-12)
    assert rel < 1e-3, f"Projection did not recover coeffs, rel_err={rel}"

test_fourier_periodic_orthonormality()
test_reconstruction_projection_consistency()