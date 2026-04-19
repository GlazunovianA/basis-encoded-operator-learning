import numpy as np

try:
    from scipy.linalg import qr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from encoding_decoding import trapezoid_weights


def _build_time_candidate_matrix_cosine(t_grid: np.ndarray, Nt: int) -> np.ndarray:
    """
    Build a cosine candidate matrix A \in R^{T x Nt} on the sampled time grid.

    Column 0 is the constant mode.
    Columns j>=1 are cosine modes on the affine-rescaled interval [0,1].
    """
    t = np.asarray(t_grid, dtype=float)
    T = len(t)
    if Nt > T:
        raise ValueError(f"Nt={Nt} cannot exceed number of time samples T={T}")

    t0 = float(t[0])
    t1 = float(t[-1])
    if not (t1 > t0):
        raise ValueError("Need t1 > t0")

    tau = (t - t0) / (t1 - t0)  # map to [0,1]
    A = np.empty((T, Nt), dtype=float)
    A[:, 0] = 1.0
    for k in range(1, Nt):
        A[:, k] = np.cos(np.pi * k * tau)
    return A


def _weighted_qr_orthonormalize(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Given candidate matrix A \in R^{T x Nt} and positive weights w \in R^T,
    return Q \in R^{T x Nt} with Q^T diag(w) Q = I,
    using QR on W^{1/2} A.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)

    if A.shape[0] != len(w):
        raise ValueError("A and w have incompatible shapes")
    if np.any(w <= 0):
        raise ValueError("Weights must be strictly positive")

    sqrt_w = np.sqrt(w)
    WA = sqrt_w[:, None] * A  # W^{1/2} A

    if _HAS_SCIPY:
        Qe, Re = qr(WA, mode="economic")
    else:
        Qe, Re = np.linalg.qr(WA, mode="reduced")

    # Undo the W^{1/2} scaling:
    Q = Qe / sqrt_w[:, None]

    return Q


class Scheme1CacheDiscreteTimeONB:
    """
    Continuous-output encoding/decoding with:
      - time basis = weighted discrete orthonormal basis built on the sampled time grid
      - space basis = rFFT modes, truncated to Kx

    The time basis is constructed from cosine candidates and then orthonormalized
    in the discrete weighted inner product induced by trapezoid weights.

    Hence, by construction:
        Phi_t^T W Phi_t \approx I
    up to floating-point error.
    """

    def __init__(self, t_grid: np.ndarray, x_grid: np.ndarray, Nt: int | None = None, Kx: int | None = None):
        self.t = np.asarray(t_grid, dtype=float)
        self.x = np.asarray(x_grid, dtype=float)

        self.T = len(self.t)
        self.N_x = len(self.x)

        self.K_full = self.N_x // 2 + 1  # rfft length
        self.Nt = self.T if Nt is None else int(Nt)
        self.Kx = self.K_full if Kx is None else int(Kx)

        if not (1 <= self.Nt <= self.T):
            raise ValueError(f"Nt must satisfy 1 <= Nt <= T={self.T}, got Nt={self.Nt}")
        if not (1 <= self.Kx <= self.K_full):
            raise ValueError(f"Kx must satisfy 1 <= Kx <= K_full={self.K_full}, got Kx={self.Kx}")

        self.wt = trapezoid_weights(self.t)

        A = _build_time_candidate_matrix_cosine(self.t, self.Nt)
        self.Phi_t = _weighted_qr_orthonormalize(A, self.wt)  # (T, Nt)

        self.Gt = self.Phi_t.T @ (self.wt[:, None] * self.Phi_t)
        self.Gt_minus_I_norm = float(np.linalg.norm(self.Gt - np.eye(self.Nt)))
        self.Gt_cond = float(np.linalg.cond(self.Gt))

    def encode(self, U: np.ndarray) -> np.ndarray:
        """
        Encode U of shape (T, N_x) to a real vector of length 2*Nt*Kx.
        """
        U = np.asarray(U)
        if U.shape != (self.T, self.N_x):
            raise ValueError(f"U shape {U.shape} does not match expected {(self.T, self.N_x)}")

        Uhat = np.fft.rfft(U, axis=1, norm="ortho")[:, :self.Kx]  # (T, Kx), complex

        # Since Phi_t^T W Phi_t = I, coefficients are just weighted projections
        C = self.Phi_t.T @ (self.wt[:, None] * Uhat)  # (Nt, Kx), complex

        y_vec = np.concatenate([C.real.ravel(), C.imag.ravel()], axis=0)
        return y_vec.astype(np.float64)

    def decode(self, y_vec: np.ndarray) -> np.ndarray:
        """
        Decode y_vec of length 2*Nt*Kx back to U_rec of shape (T, N_x).
        """
        y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        expected_len = 2 * self.Nt * self.Kx
        if y_vec.shape[0] != expected_len:
            raise ValueError(f"y_vec length {y_vec.shape[0]} != expected {expected_len}")

        half = y_vec.shape[0] // 2
        Cre = y_vec[:half].reshape(self.Nt, self.Kx)
        Cim = y_vec[half:].reshape(self.Nt, self.Kx)
        C = Cre + 1j * Cim

        Uhat_trunc = self.Phi_t @ C  # (T, Kx), complex

        Uhat_full = np.zeros((self.T, self.K_full), dtype=np.complex128)
        Uhat_full[:, :self.Kx] = Uhat_trunc

        U_rec = np.fft.irfft(Uhat_full, n=self.N_x, axis=1, norm="ortho")
        return U_rec