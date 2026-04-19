import numpy as np
from basis_1d import eval_basis_matrix, reconstruct_from_coeffs

def decay_weights(N_basis: int, decay_type: str, decay_rate: float) -> np.ndarray:
    if decay_type == "none":
        return np.ones(N_basis, dtype=float)
    if decay_type == "polynomial":
        n = np.arange(N_basis, dtype=float)
        return (1.0 + n) ** (-decay_rate)
    if decay_type == "exponential":
        n = np.arange(N_basis, dtype=float)
        return np.exp(-decay_rate * n)
    raise ValueError(f"Unknown decay_type={decay_type}")

def sample_u0_from_uniform_params(
    rng: np.random.Generator,
    basis: str,
    N_basis: int,
    x: np.ndarray,
    decay_type: str = "polynomial",
    decay_rate: float = 0.5,
    mean_zero: bool = True,
):
    """
    theta ~ Unif(-1,1)^N_basis # uniform sampling.
    coeffs = theta * w # actual coeffs. 
    u0(x) = sum coeffs_j phi_j(x)
    """
    theta = rng.uniform(-1.0, 1.0, size=(N_basis,))
    w = decay_weights(N_basis, decay_type, decay_rate)
    coeffs = theta * w

    Phi = eval_basis_matrix(basis, N_basis, x)
    u0 = reconstruct_from_coeffs(Phi, coeffs)

    if mean_zero:
        u0 = u0 - float(np.mean(u0))

    return theta, coeffs, u0
