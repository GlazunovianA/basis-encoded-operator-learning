import numpy as np
import scipy.special as sp
from basis_1d import eval_basis_matrix, project_L2_from_samples, reconstruct_from_coeffs
# -------------------------
# 1) Utilities: periodic x grid and FFT packing
# -------------------------

def rfft_truncate(u_x: np.ndarray, K: int) -> np.ndarray:
    """
    u_x: (N_x,) real values on uniform periodic grid x in [0,1).
    Returns uhat: (K,) complex rfft coefficients using orthonormal FFT.
    """
    Uhat_full = np.fft.rfft(u_x, norm="ortho")  # length N_x//2+1 complex
    if K > Uhat_full.shape[0]:
        raise ValueError(f"K={K} exceeds available rfft modes {Uhat_full.shape[0]}")
    return Uhat_full[:K].copy()

def irfft_from_trunc(uhat: np.ndarray, N_x: int) -> np.ndarray:
    """
    uhat: (K,) complex truncated rfft coeffs (orthonormal).
    Pads zeros to full rfft length and returns u_x: (N_x,) real.
    """
    K = uhat.shape[0]
    K_full = N_x // 2 + 1
    Uhat_full = np.zeros(K_full, dtype=np.complex128)
    Uhat_full[:K] = uhat
    u_x = np.fft.irfft(Uhat_full, n=N_x, norm="ortho")
    return u_x

def pack_rfft_coeffs(uhat: np.ndarray, N_x: int) -> np.ndarray:
    """
    Pack truncated rfft coeffs into a real vector.
    Convention: store Re and Im for all modes, including DC and Nyquist (if present),
    because that is simplest and fully invertible (Im parts will be ~0 for DC/Nyquist).
    Output shape: (2*K,)
    """
    return np.concatenate([uhat.real, uhat.imag], axis=0)

def unpack_rfft_coeffs(z: np.ndarray) -> np.ndarray:
    """
    Inverse of pack_rfft_coeffs. z shape (2*K,) -> uhat shape (K,) complex.
    """
    K2 = z.shape[0]
    if K2 % 2 != 0:
        raise ValueError("packed rfft vector must have even length")
    K = K2 // 2
    return z[:K] + 1j * z[K:]


# -------------------------
# 2) Time basis: Legendre orthonormal on [0, t_final]
#    with discrete L2 projection via Gram correction
# -------------------------

def legendre_orthonormal_matrix(points: np.ndarray, N: int, a: float, b: float) -> np.ndarray:
    """
    Phi[i,n] = phi_n(points[i]) where phi_n is orthonormal in Lebesgue L2([a,b]).
    discrete formation: SPD close to identity
    """
    points = np.asarray(points, dtype=float)
    if b <= a:
        raise ValueError("need b > a")
    tstd = (2.0 * (points - a) / (b - a)) - 1.0  # map to [-1,1]
    Phi = np.empty((len(points), N), dtype=float)
    for n in range(N):
        Pn = sp.legendre(n)(tstd)
        Phi[:, n] = np.sqrt((2*n + 1) / (b - a)) * Pn
    return Phi

def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """
    Trapezoidal weights for Lebesgue integral on a 1D grid (supports nonuniform).
    For uniform grid on [0,T], this is standard trapezoid.
    """
    g = np.asarray(grid, dtype=float)
    w = np.zeros_like(g)
    if len(g) == 1:
        w[0] = 1.0
        return w
    dg = np.diff(g)
    w[1:-1] = 0.5 * (dg[:-1] + dg[1:])
    w[0] = 0.5 * dg[0]
    w[-1] = 0.5 * dg[-1]
    return w

def project_time_series_to_legendre(t: np.ndarray, y: np.ndarray, N_t_ex: int, t0: float, t1: float) -> tuple[np.ndarray, dict]:
    """
    y: (T,) real or complex values sampled at times t.
    Returns c: (N_t_ex,) coefficients minimizing discrete weighted L2 residual,
    i.e. using Gram correction: c = G^{-1} Phi^T W y.
    """
    Phi = legendre_orthonormal_matrix(t, N_t_ex, t0, t1)     # (T, N_t_ex)
    wt = trapezoid_weights(t)                                 # (T,)
    G = Phi.T @ (wt[:, None] * Phi)                           # (N_t_ex, N_t_ex)
    b = Phi.T @ (wt * y)                                      # (N_t_ex,) (works for complex too)
    c = np.linalg.solve(G, b)
    meta = {"Phi_t": Phi, "w_t": wt, "G_t": G, "t0": t0, "t1": t1}
    return c, meta

def eval_legendre_series(Phi_t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Given Phi_t (T, N_t_ex) and coeffs c (N_t_ex,), return y_hat (T,).
    Works with real/complex c.
    """
    return Phi_t @ c


# -------------------------
# 3) Output encoders/decoders
# -------------------------

def encode_output_discrete_time_fourier(t: np.ndarray, x: np.ndarray, U: np.ndarray, Kx: int) -> tuple[np.ndarray, dict]:
    """
    Discrete stepping output: NO time expansion.
    Encode full U(t_k, x) for all k by truncated rFFT in x at each time step.
    Output vector shape: (T * 2*Kx,).
    """
    T, N_x = U.shape
    if len(t) != T or len(x) != N_x:
        raise ValueError("shape mismatch between (t,x) and U")

    Uhat = np.fft.rfft(U, axis=1, norm="ortho")[:, :Kx]  # (T, Kx) complex
    Y = np.concatenate([Uhat.real, Uhat.imag], axis=1)    # (T, 2*Kx)
    y_vec = Y.reshape(-1)                                 # (T*2*Kx,)

    meta = {
        "scheme": "discrete_time_fourier",
        "T": int(T),
        "N_x": int(N_x),
        "Kx": int(Kx),
        "fft_norm": "ortho",
        "t0": float(t[0]),
        "t1": float(t[-1]),
    }
    return y_vec, meta

def decode_output_discrete_time_fourier(y_vec: np.ndarray, meta: dict) -> np.ndarray:
    """
    Decode discrete stepping output vector back to U(t_k, x) on the original grid length N_x.
    Returns U_rec shape (T, N_x).
    """
    T = meta["T"]
    N_x = meta["N_x"]
    Kx = meta["Kx"]

    Y = y_vec.reshape(T, 2*Kx)
    Uhat_trunc = Y[:, :Kx] + 1j * Y[:, Kx:]               # (T, Kx)

    K_full = N_x // 2 + 1
    Uhat_full = np.zeros((T, K_full), dtype=np.complex128)
    Uhat_full[:, :Kx] = Uhat_trunc

    U_rec = np.fft.irfft(Uhat_full, n=N_x, axis=1, norm="ortho")
    return U_rec

# --------------- consistent with input basis projection --------------

def encode_output_slice_real_fourier_coeffs(
    t: np.ndarray,
    x: np.ndarray,
    U: np.ndarray,
    t_index: int,
    basis: str,
    N_basis: int,
) -> tuple[np.ndarray, dict]:
    """
    Encode u(t_index, x) by projecting onto the SAME real Fourier basis used for u0.
    Output y_vec is real of shape (N_basis,).
    """
    if t_index < 0 or t_index >= U.shape[0]:
        raise ValueError("t_index out of range")

    # quadrature consistent with your data generation (uniform on [0,1), periodic)
    # if x came from uniform_quad_on_01(N_x), then w = 1/N_x is correct
    N_x = len(x)
    w = np.full(N_x, 1.0 / N_x)

    Phi = eval_basis_matrix(basis=basis, N_basis=N_basis, x=x)  # (N_basis, N_x)
    u_slice = U[t_index, :]                                      # (N_x,)

    c = project_L2_from_samples(Phi, w, u_slice)                 # (N_basis,)
    y_vec = np.asarray(c, dtype=np.float64)

    meta = {
        "scheme": "slice_real_fourier_coeffs",
        "basis": basis,
        "N_basis": int(N_basis),
        "N_x": int(N_x),
        "x_grid": np.asarray(x),
        "w_x": w,                      # optional but makes inner product explicit
        "t_index": int(t_index),
        "t_value": float(t[t_index]),
    }
    return y_vec, meta


def decode_output_slice_real_fourier_coeffs(y_vec: np.ndarray, meta: dict) -> np.ndarray:
    """
    Decode coefficients back to u(x) on the stored x_grid.
    """
    basis = meta["basis"]
    N_basis = int(meta["N_basis"])
    x = np.asarray(meta["x_grid"])
    Phi = eval_basis_matrix(basis=basis, N_basis=N_basis, x=x)  # (N_basis, N_x)
    c = np.asarray(y_vec).reshape(N_basis)
    u_rec = reconstruct_from_coeffs(Phi, c)                     # (N_x,)
    return u_rec

# --------------- consistent with input basis projection --------------

def encode_output_scheme2_slice_fourier(
    t: np.ndarray,
    x: np.ndarray,
    U: np.ndarray,
    t_index: int,
    Kx: int,  # reinterpret: Kx == N_basis (for compatibility with caller)
) -> tuple[np.ndarray, dict]:
    """
    Replacement Scheme (2): encode u(t_index, x) by projection onto real Fourier basis.
    Returns y_vec real of shape (Kx,) where Kx is now N_basis.
    """
    if t_index < 0 or t_index >= U.shape[0]:
        raise ValueError("t_index out of range")

    N_x = len(x)
    w = np.full(N_x, 1.0 / N_x)

    basis = "fourier_periodic"
    N_basis = int(Kx)  # treat Kx as N_basis to keep call sites unchanged

    Phi = eval_basis_matrix(basis=basis, N_basis=N_basis, x=x)  # (N_basis, N_x)
    u_slice = U[t_index, :]                                      # (N_x,)
    c = project_L2_from_samples(Phi, w, u_slice)                 # (N_basis,)

    meta = {
        "scheme": "slice_real_fourier_coeffs",
        "basis": basis,
        "N_basis": N_basis,
        "N_x": N_x,
        "x_grid": np.asarray(x),
        "t_index": int(t_index),
        "t_value": float(t[t_index]),
        "quad": "uniform_on_01",
        'u_slice': u_slice,  # for debugging/inspection
        "w_x": w,  # optional but explicit
    }
    return np.asarray(c, dtype=np.float64), meta


def decode_output_scheme2_slice_fourier(y_vec: np.ndarray, meta: dict) -> np.ndarray:
    """
    Decode back to u(x) on meta['x_grid'] using the same real Fourier basis.
    """
    basis = meta["basis"]
    N_basis = int(meta["N_basis"])
    x = np.asarray(meta["x_grid"])

    Phi = eval_basis_matrix(basis=basis, N_basis=N_basis, x=x)  # (N_basis, N_x)
    c = np.asarray(y_vec, dtype=np.float64).reshape(N_basis)
    u_x = reconstruct_from_coeffs(Phi, c)                       # (N_x,)
    return u_x



def encode_output_scheme1_legendre_time_fourier_space(t: np.ndarray, x: np.ndarray, U: np.ndarray, Nt: int, Kx: int) -> tuple[np.ndarray, dict]:
    """
    Scheme (1): represent U(t,x) by:
      - truncate rFFT in x to Kx modes at each time -> Uhat(t, k)
      - for each spatial mode k, project its time series onto Nt Legendre basis -> C_time[n,k]
    Output vector packs Re/Im of C_time: shape (2*Nt*Kx,).

    This encodes a separable approximation:
        U(t,x) ≈ sum_{k< Kx} ( sum_{n<Nt} C_{n,k} phi_n(t) ) * e^{2π i k x}   (in rFFT convention)
    """
    T, N_x = U.shape
    if len(t) != T or len(x) != N_x:
        raise ValueError("shape mismatch between (t,x) and U")
    t0, t1 = float(t[0]), float(t[-1])

    # Precompute time basis once
    Phi_t = legendre_orthonormal_matrix(t, Nt, t0, t1)
    wt = trapezoid_weights(t)
    Gt = Phi_t.T @ (wt[:, None] * Phi_t)   # (Nt,Nt)
    Gt_inv = np.linalg.inv(Gt)             # Nt moderate; for stability you can use solve per mode

    # Step 1: spatial rFFT at each time
    Uhat = np.fft.rfft(U, axis=1, norm="ortho")  # (T, N_x//2+1)
    Uhat = Uhat[:, :Kx]                           # (T, Kx) complex

    # Step 2: project each spatial mode time series onto Legendre with Gram correction
    C = np.zeros((Nt, Kx), dtype=np.complex128)
    # b_k = Phi^T W y_k
    B = Phi_t.T @ (wt[:, None] * Uhat)           # (Nt, Kx) complex
    C = Gt_inv @ B                                # (Nt, Kx) complex

    # pack to real vector
    y_vec = np.concatenate([C.real.ravel(), C.imag.ravel()], axis=0)  # (2*Nt*Kx,)
    meta = {
        "scheme": "legendre_time_fourier_space",
        "Nt": int(Nt),
        "Kx": int(Kx),
        "N_x": int(N_x),
        "t0": t0,
        "t1": t1,
        "fft_norm": "ortho",
        # store what decoder needs
        "Phi_t": Phi_t,
        "wt": wt,
        "Gt": Gt,
    }
    return y_vec, meta

def decode_output_scheme1_legendre_time_fourier_space(y_vec: np.ndarray, meta: dict, t: np.ndarray) -> np.ndarray:
    """
    Decode scheme (1) vector back to U(t,x) on the given t-grid and original x grid length N_x.
    Uses the stored time basis Phi_t if t matches; otherwise recomputes Phi_t for the provided t.
    """
    Nt = meta["Nt"]
    Kx = meta["Kx"]
    N_x = meta["N_x"]
    t0 = meta["t0"]
    t1 = meta["t1"]

    # unpack C (Nt,Kx) complex
    half = y_vec.shape[0] // 2
    Cre = y_vec[:half].reshape(Nt, Kx)
    Cim = y_vec[half:].reshape(Nt, Kx)
    C = Cre + 1j * Cim

    # Build Phi_t for the provided t-grid (safer than assuming same object identity)
    Phi_t = legendre_orthonormal_matrix(t, Nt, t0, t1)  # (T,Nt)

    # Reconstruct truncated Fourier coeffs over time: Uhat(t,k) = Phi_t @ C[:,k]
    Uhat_trunc = Phi_t @ C                                # (T, Kx) complex

    # Pad to full rFFT length and inverse transform
    K_full = N_x // 2 + 1
    Uhat_full = np.zeros((len(t), K_full), dtype=np.complex128)
    Uhat_full[:, :Kx] = Uhat_trunc
    U_rec = np.fft.irfft(Uhat_full, n=N_x, axis=1, norm="ortho")  # (T,N_x)
    return U_rec


# -------------------------
# 4) A reconstruction-possibility test (encoder/decoder consistency)
# -------------------------

def check_reconstruction_scheme1(t, x, U, Nt, Kx):
    y, meta = encode_output_scheme1_legendre_time_fourier_space(t, x, U, Nt=Nt, Kx=Kx)
    Urec = decode_output_scheme1_legendre_time_fourier_space(y, meta, t)

    # discrete L2 on tensor grid: (1/T measure approx) * (1/N_x) * sum
    wt = trapezoid_weights(t)
    wx = np.full(len(x), 1.0/len(x))  # periodic trapezoid weights on uniform x
    err2 = np.sum((wt[:,None] * (U - Urec)**2) * wx[None,:])
    ref2 = np.sum((wt[:,None] * (U)**2) * wx[None,:]) + 1e-12
    return np.sqrt(err2 / ref2)

def check_reconstruction_scheme2(t, x, U, t_index, Kx):
    y, meta = encode_output_scheme2_slice_fourier(t, x, U, t_index=t_index, Kx=Kx)
    urec = decode_output_scheme2_slice_fourier(y, meta)
    u = U[t_index, :]
    err = np.linalg.norm(u - urec) / (np.linalg.norm(u) + 1e-12)
    return err

def check_reconstruction_scheme2_in_span(Kx: int = 10, Nx: int = 256):
    x = np.linspace(0.0, 1.0, Nx, endpoint=False)
    t = np.linspace(0.0, 1.0, 5)

    basis = "fourier_periodic"
    Phi = eval_basis_matrix(basis=basis, N_basis=Kx, x=x)  # (Kx, Nx)

    rng = np.random.default_rng(42)
    c = rng.normal(size=Kx)
    u = reconstruct_from_coeffs(Phi, c)  # exactly in span

    U = np.tile(u[None, :], (len(t), 1))
    err = check_reconstruction_scheme2(t=t, x=x, U=U, t_index=0, Kx=Kx)
    return err

print("Scheme (2) in-span reconstruction error:", check_reconstruction_scheme2_in_span(Kx=20))

T, Nx = 100, 256
t = np.linspace(0.0, 1.0, T)                 # (T,)
x = np.linspace(0.0, 1.0, Nx, endpoint=False) # periodic grid (Nx,)

t_std = 2.0*t - 1.0                           # map [0,1] -> [-1,1]
P2 = sp.eval_legendre(2, t_std)               # (T,)
bx = np.cos(2*np.pi*3*x)                      # (Nx,)
U1 = P2[:, None] * bx[None, :]                # (T, Nx)

err_1 = check_reconstruction_scheme1(t=t, x=x, U=U1, Nt=10, Kx=20)
err_2 = check_reconstruction_scheme2_in_span(Kx=20, Nx=Nx)

print(f"Scheme (1) reconstruction relative L2 error: {err_1:.2e}"   )
print(f"Scheme (2) reconstruction relative L2 error: {err_2:.2e}"   )    # this is huge. 