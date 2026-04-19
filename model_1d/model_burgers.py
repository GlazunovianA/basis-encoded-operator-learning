import os
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from encoding_decoding import (
    encode_output_scheme2_slice_fourier,
    decode_output_scheme2_slice_fourier,
    encode_output_scheme1_legendre_time_fourier_space,
    decode_output_scheme1_legendre_time_fourier_space,
    trapezoid_weights,  legendre_orthonormal_matrix
)
import sys
import pandas as pd
from pathlib import Path

from ED_error_test import (plot_expansion_error_curve,expansion_error_curve_for_one_snapshot)

ROOT = Path(__file__).resolve().parents[1]  # .../neuralsurrogate_reproduce
sys.path.insert(0, str(ROOT))

from G_N import NN_cff_vec, RFF

from io_pipeline import load_solution, load_parameter  #
# ============================================================
# 0) Small helpers: caching, time->index selection, vector errors
# ============================================================

def nearest_time_index(t: np.ndarray, t1: float) -> int:
    return int(np.argmin(np.abs(t - float(t1))))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

def save_npz(path: str, **kwargs):
    ensure_dir(os.path.dirname(path))
    np.savez(path, **kwargs)

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}
import numpy as np

try:
    from scipy.linalg import cho_factor, cho_solve
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def weighted_global_rel_l2(U_pred: np.ndarray, U_true: np.ndarray, t: np.ndarray, x: np.ndarray) -> float:
    wt = trapezoid_weights(t)  # (T,)
    wx = np.full(len(x), 1.0 / len(x))
    err2 = np.sum((wt[:, None] * (U_pred - U_true) ** 2) * wx[None, :])
    ref2 = np.sum((wt[:, None] * (U_true) ** 2) * wx[None, :]) + 1e-12
    return float(np.sqrt(err2 / ref2))

def pca_compression_study_on_Y(
    outB: dict,
    data_dir: str,
    test_indices: np.ndarray,
    ranks=(16, 32, 64, 128, 256, 512),
    n_show=1,
):
    """
    outB: run_training_pipeline output for task="continuous"
          must contain Ytrue, Yhat, meta
    We fit PCA on Ytrue (train-like) is better, but minimally we can fit on outB["Ytrue"] of test
    if you also kept Ytr, prefer fitting on training Y. Here we fit on Ytrue from test for convenience.

    Computes:
      - explained variance curve
      - for each rank r: additional error introduced by PCA compression on (a) true coeffs, (b) predicted coeffs
      - compares to repr bound and pred error.
    """
    meta = outB["meta"]
    Nt = int(meta["Nt"])
    Kx = int(meta["Kx"])
    t_grid = np.asarray(meta["t_grid"], dtype=float)
    x_grid = np.asarray(meta["x_grid"], dtype=float)

    # Build decode cache (fast)
    cache = Scheme1Cache(t_grid=t_grid, x_grid=x_grid, Nt=Nt, Kx=Kx)
    print("||Gt-I||/||I|| (rel Fro):", cache.Gt_minus_I_norm / (np.sqrt(Nt) + 1e-12))

    Ytrue = np.asarray(outB["Ytrue"], dtype=float)  # (n_test, D)
    Yhat  = np.asarray(outB["Yhat"], dtype=float)   # (n_test, D)
    n_test, D = Ytrue.shape
    assert D == 2 * Nt * Kx

    # Fit PCA on Ytrue distribution (ideally use training Y instead; see note below)
    pca = PCA(n_components=min(D, max(ranks)))
    pca.fit(Ytrue)

    # Explained variance plot
    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    plt.figure()
    plt.semilogy(np.arange(1, len(evr) + 1), 1.0 - cumevr, marker="o")
    plt.xlabel("PCA rank r")
    plt.ylabel("1 - cumulative explained variance")
    plt.title("Y-space PCA residual energy")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

    # Evaluate on a few examples in function space
    # We'll use the first n_show test trajectories (decode full U for those)
    show_ids = list(range(min(n_show, len(test_indices))))

    results = []
    for i in show_ids:
        idx0 = int(test_indices[i])
        t, x, U_true = load_solution(data_dir, idx0)
        # sanity: assume grids match meta (you said uniform/shared)
        # If not, you'd need a cache for each t,x.

        # baseline: repr bound from true coeffs as stored (already encoded)
        U_repr = cache.decode(Ytrue[i], t_grid=t)  # (T,Nx)
        base_repr = weighted_global_rel_l2(U_repr, U_true, t, x)

        # baseline: pred decoded
        U_pred = cache.decode(Yhat[i], t_grid=t)
        base_pred = weighted_global_rel_l2(U_pred, U_true, t, x)

        print(f"[ex {i}] baseline: pred={base_pred:.6f}, repr_bound={base_repr:.6f}, gap={base_pred-base_repr:.6f}")

        # PCA ranks
        for r in ranks:
            r = int(r)
            # compress+reconstruct Ytrue and Yhat
            Zt = pca.transform(Ytrue[i:i+1])[:, :r]
            Zp = pca.transform(Yhat[i:i+1])[:, :r]

            # To invert with truncated components: use components_[:r]
            Ytrue_pca = pca.mean_ + Zt @ pca.components_[:r, :]
            Yhat_pca  = pca.mean_ + Zp @ pca.components_[:r, :]

            U_repr_pca = cache.decode(Ytrue_pca[0], t_grid=t)
            U_pred_pca = cache.decode(Yhat_pca[0],  t_grid=t)

            err_repr_pca = weighted_global_rel_l2(U_repr_pca, U_true, t, x)
            err_pred_pca = weighted_global_rel_l2(U_pred_pca, U_true, t, x)

            results.append({
                "example": i,
                "rank": r,
                "pred_err": err_pred_pca,
                "repr_err": err_repr_pca,
                "pred_minus_repr": err_pred_pca - err_repr_pca,
                "repr_minus_base_repr": err_repr_pca - base_repr,
                "pred_minus_base_pred": err_pred_pca - base_pred,
            })

    # Aggregate over shown examples
    # (if you want a reliable aggregate, set n_show=50 or 100; but it will take longer)
    if len(results) > 0:
        
        df = pd.DataFrame(results)
        grp = df.groupby("rank").mean(numeric_only=True).reset_index()

        plt.figure()
        plt.plot(grp["rank"], grp["repr_minus_base_repr"], marker="o", label="PCA adds to repr-bound")
        plt.plot(grp["rank"], grp["pred_minus_base_pred"], marker="o", label="PCA adds to pred")
        plt.xlabel("PCA rank r")
        plt.ylabel("Δ global rel L2 error")
        plt.title("Extra function-space error introduced by PCA")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(grp["rank"], grp["pred_err"], marker="o", label="pred (after PCA)")
        plt.plot(grp["rank"], grp["repr_err"], marker="o", label="repr bound (after PCA)")
        plt.xlabel("PCA rank r")
        plt.ylabel("global rel L2 error")
        plt.title("Function-space errors vs PCA rank")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        plt.show()

        print(grp)


def print_rank_curve_table(results: list[dict], title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for row in results:
        print(row)


def choose_smallest_rank_under_tolerance(
    results: list[dict],
    rank_key: str,
    metric_key: str,
    tolerance: float,
):
    if len(results) == 0:
        raise ValueError("results must not be empty")

    rows_sorted = sorted(results, key=lambda row: int(row[rank_key]))
    for row in rows_sorted:
        metric_value = float(row[metric_key])
        if metric_value <= float(tolerance):
            return int(row[rank_key]), row, "threshold_met"

    return int(rows_sorted[-1][rank_key]), rows_sorted[-1], "threshold_not_met"


def choose_smallest_pair_under_tolerance(
    results: list[dict],
    nt_key: str,
    kx_key: str,
    metric_key: str,
    tolerance: float,
):
    """
    Choose the smallest (Nt, Kx) pair meeting tolerance.

    Ordering:
        1) smallest encoded dimension 2*Nt*Kx
        2) smaller Nt
        3) smaller Kx
    """
    if len(results) == 0:
        raise ValueError("results must not be empty")

    rows_sorted = sorted(
        results,
        key=lambda row: (
            2 * int(row[nt_key]) * int(row[kx_key]),
            int(row[nt_key]),
            int(row[kx_key]),
        ),
    )

    for row in rows_sorted:
        metric_value = float(row[metric_key])
        if metric_value <= float(tolerance):
            return (int(row[nt_key]), int(row[kx_key])), row, "threshold_met"

    last = rows_sorted[-1]
    return (int(last[nt_key]), int(last[kx_key])), last, "threshold_not_met"

class Scheme1Cache:
    """
    Cache for Scheme (1): Legendre(time) x rFFT(space) encoding/decoding.

    Assumes:
      - t_grid and x_grid are fixed across dataset (you said they are).
      - encode uses trapezoid_weights(t_grid) and legendre_orthonormal_matrix(t_grid, Nt, t0, t1).
      - FFT norm="ortho".

    Stores:
      - Phi_t: (T, Nt)
      - wt: (T,)
      - solver for Gt: Phi_t.T @ (wt * Phi_t)   (Nt,Nt)
      - shapes for rFFT padding
    """
    def __init__(self, t_grid: np.ndarray, x_grid: np.ndarray, Nt: int, Kx: int):
        self.t = np.asarray(t_grid, dtype=float)
        self.x = np.asarray(x_grid, dtype=float)
        self.T = len(self.t)
        self.N_x = len(self.x)
        self.T = len(self.t)
        # rFFT sizing
        self.K_full = self.N_x // 2 + 1  # rfft length
        self.Nt = self.T if Nt is None else int(Nt)
        self.Kx = self.K_full if Kx is None else int(Kx)

        self.t0 = float(self.t[0])
        self.t1 = float(self.t[-1])

        # --- time basis + weights ---
        self.Phi_t = legendre_orthonormal_matrix(self.t, self.Nt, self.t0, self.t1)  # (T, Nt)
        self.wt = trapezoid_weights(self.t)  # (T,)

        # Gram matrix under discrete inner product
        self.Gt = self.Phi_t.T @ (self.wt[:, None] * self.Phi_t)  # (Nt, Nt)

        # Precompute solver; prefer Cholesky (Gt should be SPD if basis is independent on grid)
        self._use_cholesky = False
        self._cho = None

        if _HAS_SCIPY:
            try:
                self._cho = cho_factor(self.Gt, lower=True, check_finite=False)
                self._use_cholesky = True
            except Exception:
                self._use_cholesky = False

        

        # Optional: store diagnostics
        self.Gt_minus_I_norm = float(np.linalg.norm(self.Gt - np.eye(self.Nt)))
        # condition number can be expensive; enable if you want
        self.Gt_cond = float(np.linalg.cond(self.Gt))

    def solve_Gt(self, B: np.ndarray) -> np.ndarray:
        """
        Solve Gt * C = B for C, where B shape is (Nt, Kx), complex allowed.
        """
        if self._use_cholesky:
            return cho_solve(self._cho, B, check_finite=False)
        # fallback: numpy solve (still stable enough for moderate Nt)
        return np.linalg.solve(self.Gt, B)

    def encode(self, U: np.ndarray) -> np.ndarray:
        """
        Encode a single trajectory U with shape (T, N_x) into y_vec shape (2*Nt*Kx,).
        """
        U = np.asarray(U)
        if U.shape != (self.T, self.N_x):
            raise ValueError(f"U shape {U.shape} does not match expected {(self.T, self.N_x)}")

        Uhat = np.fft.rfft(U, axis=1, norm="ortho")[:, : self.Kx]  # (T, Kx) complex

        # B = Phi_t^T (wt * Uhat)
        B = self.Phi_t.T @ (self.wt[:, None] * Uhat)  # (Nt, Kx) complex
        C = self.solve_Gt(B)  # (Nt, Kx) complex

        y_vec = np.concatenate([C.real.ravel(), C.imag.ravel()], axis=0)  # (2*Nt*Kx,)
        return y_vec.astype(np.float64)

    def decode(self, y_vec: np.ndarray, t_grid: np.ndarray | None = None) -> np.ndarray:
        """
        Decode y_vec back to U(t,x). If t_grid is None or equals cached t, use cached Phi_t.
        Otherwise recompute Phi_t on provided t_grid using same [t0,t1] and Nt.
        """
        y_vec = np.asarray(y_vec, dtype=float).reshape(-1)
        half = y_vec.shape[0] // 2
        if y_vec.shape[0] != 2 * self.Nt * self.Kx:
            raise ValueError(f"y_vec length {y_vec.shape[0]} != 2*Nt*Kx = {2*self.Nt*self.Kx}")

        Cre = y_vec[:half].reshape(self.Nt, self.Kx)
        Cim = y_vec[half:].reshape(self.Nt, self.Kx)
        C = Cre + 1j * Cim  # (Nt, Kx)

        if t_grid is None:
            Phi_t = self.Phi_t
            T = self.T
        else:
            t_grid = np.asarray(t_grid, dtype=float)
            Phi_t = legendre_orthonormal_matrix(t_grid, self.Nt, self.t0, self.t1)
            T = len(t_grid)

        Uhat_trunc = Phi_t @ C  # (T, Kx) complex

        Uhat_full = np.zeros((T, self.K_full), dtype=np.complex128)
        Uhat_full[:, : self.Kx] = Uhat_trunc
        U_rec = np.fft.irfft(Uhat_full, n=self.N_x, axis=1, norm="ortho")  # (T, N_x)
        return U_rec

# =================== sanity check ==========================

def analyze_input_truncation_curve_1d(
    data_dir: str,
    indices: np.ndarray,
    input_ranks: Sequence[int],
):
    """
    Since the parameter vector is already the intended canonical input,
    input-side dimension choice is a direct truncation problem, not PCA.

    We compare truncated parameter vectors to the full parameter vector.
    """
    input_ranks = sorted(set(int(r) for r in input_ranks))
    if len(input_ranks) == 0:
        raise ValueError("input_ranks must not be empty")

    full_params = []
    for idx in indices:
        idx = int(idx)
        _, _, _, coeffs = load_parameter(data_dir, idx)
        full_params.append(np.asarray(coeffs, dtype=np.float64).reshape(-1))

    d_full = full_params[0].shape[0]

    results = []
    for r in input_ranks:
        errs = []
        for c in full_params:
            r_eff = min(int(r), d_full)
            c_trunc = np.zeros_like(c)
            c_trunc[:r_eff] = c[:r_eff]
            errs.append(rel_l2(c_trunc, c))

        results.append({
            "input_rank": int(r),
            "parameter_rel_l2_mean_vs_full": float(np.mean(errs)),
            "parameter_rel_l2_median_vs_full": float(np.median(errs)),
        })

    return results


def plot_input_truncation_curve_1d(
    results: list[dict],
    metric_keys: Sequence[str] = ("parameter_rel_l2_mean_vs_full",),
    title: str = "1D input truncation analysis",
):
    ranks = [int(row["input_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    plt.xlabel("input truncation rank")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# 1) Dataset builders for the TWO tasks (uses your encoders)
# ============================================================

def build_dataset_task_step(
    data_dir: str,
    indices: np.ndarray,
    Kx: int,
    t1: float,
    Kin: int | None = None,   # NEW: optional input truncation dim
):
    """
    Task A: learn operator u0 -> u(t1,·) where output encoded by slice_fourier (2*Kx dims).
    Input is coeffs loaded from fem_parameter_coeffs_{idx}.npy via your load_parameter().
    """
    X_list, Y_list = [], []
    meta_ref = None

    for idx in indices:
        idx = int(idx)
        # input coefficients
        _, _, _, coeffs = load_parameter(data_dir, idx)  # coeffs: (d_in,)
        coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if Kin is None:
            x_vec = coeffs
        else:
            Kin_eff = int(min(Kin, coeffs.shape[0]))
            x_vec = coeffs[:Kin_eff]

        # output solution
        t, x, U = load_solution(data_dir, idx)  # t:(T,), x:(Nx,), U:(T,Nx)
        ti = nearest_time_index(t, t1)

        y_vec, y_meta = encode_output_scheme2_slice_fourier(t, x, U, t_index=ti, Kx=Kx)
        # y_rec = decode_output_scheme2_slice_fourier(y_vec, y_meta)
        # difference between y_vec and y_rec is only due to truncation to Kx modes; the encoding/decoding is consistent.
        # print('error due to truncation', np.linalg.norm(y_rec - y_meta["u_slice"]))  # should be close to zero up to truncation error
        # X_list.append(coeffs[:min(Kx, coeffs.shape[0])]) # force maximal dimension Kx on both input/output for possible stepping. 
        X_list.append(x_vec)
        Y_list.append(y_vec) # y_vec is the encoded u(t1,x) in Fourier space, shape (2*Kx,)

        if meta_ref is None:
            meta_ref = {
                "task": "step",
                "Kx": int(Kx),
                "Kin": None if Kin is None else int(Kin),
                "t1_requested": float(t1),
                "t_index": int(ti),
                "t1_actual": float(t[ti]),
                "x_grid": x,
                "t_grid": t,
                "y_meta": y_meta,
                "d_in": int(coeffs.shape[0]),
                "d_out": int(y_vec.shape[0]),
            }

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta_ref


def build_dataset_task_continuous_cached(
    data_dir: str,
    indices: np.ndarray,
    Nt: int,
    Kx: int,
):
    """
    Task B: input coeffs -> y_vec (2*Nt*Kx) using cached Scheme (1).
    """
    X_list, Y_list = [], []
    meta_ref = None
    cache = None
    for idx in indices:
        idx = int(idx)
        _, _, _, coeffs = load_parameter(data_dir, idx)
        coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)

        t, x, U = load_solution(data_dir, idx)

        if cache is None:

            cache = Scheme1Cache(t_grid=t, x_grid=x, Nt=Nt, Kx=Kx)
            meta_ref = {
                "task": "continuous",
                "Nt": int(Nt),
                "Kx": int(Kx),
                "x_grid": x,
                "t_grid": t,
                "d_in": int(coeffs.shape[0]),
                "d_out": int(2 * Nt * Kx),
                # diagnostics:
                "Gt_minus_I_norm": cache.Gt_minus_I_norm,
            }

        y_vec = cache.encode(U)  # (2*Nt*Kx,)

        X_list.append(coeffs)
        Y_list.append(y_vec)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta_ref, cache


def load_raw_test_trajectories_1d(
    data_dir: str,
    indices: np.ndarray,
):
    """
    Load raw trajectories for visualization/evaluation only.

    Returns:
        raw_inputs: list[dict]
            each dict has keys:
                - x
                - input_coeffs
            If later you also want decoded input functions, you can extend this dict.
        raw_outputs: list[dict]
            each dict has keys:
                - t
                - x
                - U
    """
    raw_inputs = []
    raw_outputs = []

    for idx in indices:
        idx = int(idx)

        _, x_param, _, coeffs = load_parameter(data_dir, idx)
        t, x, U = load_solution(data_dir, idx)

        raw_inputs.append({
            "x": np.asarray(x_param) if x_param is not None else None,
            "input_coeffs": np.asarray(coeffs, dtype=np.float64).reshape(-1),
        })
        raw_outputs.append({
            "t": np.asarray(t, dtype=np.float64),
            "x": np.asarray(x, dtype=np.float64),
            "U": np.asarray(U, dtype=np.float64),
        })

    return raw_inputs, raw_outputs

# ============================================================
# 2) Optional PCA wrapper (same pattern as your old code)
# ============================================================

class PCATransform:
    def __init__(self, use: bool, n_components: int, path_out: str):
        self.use = bool(use)
        self.n_components = n_components
        self.path_out = path_out
        self.pca_out = None

    def fit(self, Y: np.ndarray):
        if not self.use:
            return Y
        ensure_dir(os.path.dirname(self.path_out))
        self.pca_out = PCA(n_components=self.n_components)
        Yr = self.pca_out.fit_transform(Y)

        joblib.dump(self.pca_out, self.path_out)
        return Yr

    def load_and_apply(self, Y: np.ndarray):
        if not self.use:
            return Y
        self.pca_out = joblib.load(self.path_out)
        return self.pca_out.transform(Y)

    def inverse_Y(self, Yhat: np.ndarray):
        if not self.use:
            return Yhat
        return self.pca_out.inverse_transform(Yhat)


# ============================================================
# 3) Training backends: (A) rf model, (B) a simple MLP
# ============================================================

def ridge_fit_multioutput_features(features: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    features: (n, F, D)
    Y:        (n, D)
    Solve for W: (F, D) minimizing sum_d ||Phi_d W_d - Y_d||^2 + lam ||W_d||^2
    where Phi_d = features[:,:,d].
    """
    n, F, D = features.shape
    W = torch.empty((F, D), dtype=features.dtype, device=features.device)
    I = torch.eye(F, dtype=features.dtype, device=features.device)

    # loop over outputs; acceptable when D is moderate (your two tasks typically are)
    for d in range(D):
        Phi = features[:, :, d]                  # (n, F)
        A = Phi.T @ Phi + lam * I               # (F, F)
        b = Phi.T @ Y[:, d]                     # (F,)
        W[:, d] = torch.linalg.solve(A, b)
    return W

def predict_multioutput_features(features: torch.Tensor, W: torch.Tensor):
    # features: (n,F,D), W: (F,D) -> Yhat (n,D)
    return torch.einsum("nfd,fd->nd", features, W)

# shared features-------

def ridge_fit_shared(Phi: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    Phi: (n, F) shared features
    Y:   (n, D) targets
    Returns W: (F, D)
    Solves (Phi^T Phi + lam I) W = Phi^T Y
    """
    n, F = Phi.shape
    _, D = Y.shape
    I = torch.eye(F, dtype=Phi.dtype, device=Phi.device)
    A = Phi.T @ Phi + lam * I              # (F, F)
    B = Phi.T @ Y                          # (F, D)
    W = torch.linalg.solve(A, B)           # (F, D)
    return W

def ridge_predict_shared(Phi: torch.Tensor, W: torch.Tensor):
    return Phi @ W                         # (n, D)


# ---- Simple MLP backend (vector->vector) ----

class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, depth: int = 5, dropout: float = 0.0):
        super().__init__()
        layers = []
        din = d_in
        for _ in range(depth):
            layers.append(nn.Linear(din, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            din = hidden
        layers.append(nn.Linear(din, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(model, Xtr, Ytr, Xte, Yte, lr=1e-4, epochs=1000, print_every=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        pred = model(Xtr)
        loss = ((pred - Ytr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % print_every == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                te = ((model(Xte) - Yte) ** 2).mean().item()
            print(f"epoch {ep:4d} | train_mse {loss.item():.4e} | test_mse {te:.4e}")
    return model

def eval_step_function_error(Yhat, data_dir, test_indices, t1, meta, eval_count=50):
    errs = []
    for i in range(min(eval_count, len(test_indices))):
        idx0 = int(test_indices[i])
        t_grid, x_grid, U = load_solution(data_dir, idx0)
        ti = nearest_time_index(t_grid, float(t1))

        y_meta_local = dict(meta["y_meta"])
        y_meta_local["t_index"] = int(ti)
        y_meta_local["t_value"] = float(t_grid[ti])

        u_pred = decode_output_scheme2_slice_fourier(Yhat[i], y_meta_local)
        u_true = U[ti, :]
        errs.append(rel_l2(u_pred, u_true))
    return float(np.mean(errs)), float(np.median(errs))

def eval_continuous_function_error(
    Yhat: np.ndarray,
    data_dir: str,
    test_indices: np.ndarray,
    Nt: int,
    Kx: int,
    eval_count: int = 50,
):
    errs = []
    m = min(eval_count, len(test_indices))

    for i in range(m):
        idx0 = int(test_indices[i])
        t_grid, x_grid, U_true = load_solution(data_dir, idx0)

        cache = Scheme1Cache(t_grid=t_grid, x_grid=x_grid, Nt=Nt, Kx=Kx)
        U_pred = cache.decode(Yhat[i], t_grid=t_grid)

        errs.append(weighted_global_rel_l2(U_pred, U_true, t_grid, x_grid))

    return float(np.mean(errs)), float(np.median(errs))

def analyze_step_output_basis_rank_curve(
    data_dir: str,
    indices: np.ndarray,
    t1: float,
    output_ranks: Sequence[int],
):
    """
    For each output basis rank (current Scheme 2 rank parameter),
    encode/decode u(t1, x) and compare directly to the true slice.
    """
    output_ranks = sorted(set(int(r) for r in output_ranks))
    if len(output_ranks) == 0:
        raise ValueError("output_ranks must not be empty")

    results = []
    for r in output_ranks:
        errs = []
        for idx in indices:
            idx = int(idx)
            t, x, U = load_solution(data_dir, idx)
            ti = nearest_time_index(t, float(t1))

            y_vec, y_meta = encode_output_scheme2_slice_fourier(
                t=t,
                x=x,
                U=U,
                t_index=ti,
                Kx=r,
            )
            u_rec = decode_output_scheme2_slice_fourier(y_vec, y_meta)
            u_true = U[ti, :]

            errs.append(rel_l2(u_rec, u_true))

        results.append({
            "output_rank": int(r),
            "function_rel_l2_mean_vs_raw": float(np.mean(errs)),
            "function_rel_l2_median_vs_raw": float(np.median(errs)),
        })

    return results

def analyze_step_output_pca_curve(
    data_dir: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    t1: float,
    output_rank: int,
    pca_ranks: Sequence[int],
):
    """
    PCA analysis on already-encoded step outputs at fixed output basis rank.
    """
    _, Ytr, _ = build_dataset_task_step(
        data_dir=data_dir,
        indices=train_indices,
        Kx=output_rank,
        t1=float(t1),
        Kin=None,
    )
    _, Yte, _ = build_dataset_task_step(
        data_dir=data_dir,
        indices=test_indices,
        Kx=output_rank,
        t1=float(t1),
        Kin=None,
    )

    max_rank = min(Ytr.shape[0], Ytr.shape[1])
    valid_ranks = [int(r) for r in pca_ranks if 1 <= int(r) <= max_rank]

    results = []
    for r in valid_ranks:
        pca = PCA(n_components=r)
        pca.fit(Ytr)

        Yte_red = pca.transform(Yte)
        Yte_rec = pca.inverse_transform(Yte_red)

        coeff_rel = rel_l2(Yte_rec, Yte)

        errs = []
        for i in range(len(test_indices)):
            idx0 = int(test_indices[i])
            t, x, U = load_solution(data_dir, idx0)
            ti = nearest_time_index(t, float(t1))

            _, y_meta = encode_output_scheme2_slice_fourier(
                t=t,
                x=x,
                U=U,
                t_index=ti,
                Kx=output_rank,
            )
            u_rec = decode_output_scheme2_slice_fourier(Yte_rec[i], y_meta)
            u_true = U[ti, :]
            errs.append(rel_l2(u_rec, u_true))

        results.append({
            "pca_rank": int(r),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "coeff_rel_l2": float(coeff_rel),
            "function_rel_l2_mean": float(np.mean(errs)),
            "function_rel_l2_median": float(np.median(errs)),
        })

    return results


def analyze_continuous_output_pca_curve(
    data_dir: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    Nt: int,
    Kx: int,
    pca_ranks: Sequence[int],
):
    """
    PCA analysis on already-encoded continuous outputs at fixed (Nt, Kx).
    """
    _, Ytr, _, _ = build_dataset_task_continuous_cached(
        data_dir=data_dir,
        indices=train_indices,
        Nt=Nt,
        Kx=Kx,
    )
    _, Yte, _, cache = build_dataset_task_continuous_cached(
        data_dir=data_dir,
        indices=test_indices,
        Nt=Nt,
        Kx=Kx,
    )

    max_rank = min(Ytr.shape[0], Ytr.shape[1])
    valid_ranks = [int(r) for r in pca_ranks if 1 <= int(r) <= max_rank]

    results = []
    for r in valid_ranks:
        pca = PCA(n_components=r)
        pca.fit(Ytr)

        Yte_red = pca.transform(Yte)
        Yte_rec = pca.inverse_transform(Yte_red)

        coeff_rel = rel_l2(Yte_rec, Yte)

        errs = []
        for i in range(len(test_indices)):
            idx0 = int(test_indices[i])
            t, x, U = load_solution(data_dir, idx0)
            U_rec = cache.decode(Yte_rec[i], t_grid=t)
            errs.append(weighted_global_rel_l2(U_rec, U, t, x))

        results.append({
            "pca_rank": int(r),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "coeff_rel_l2": float(coeff_rel),
            "function_rel_l2_mean": float(np.mean(errs)),
            "function_rel_l2_median": float(np.median(errs)),
        })

    return results


def plot_continuous_output_pca_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean", "coeff_rel_l2"),
    show_explained_variance: bool = True,
    title: str = "1D continuous output PCA analysis",
):
    ranks = [int(row["pca_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    if show_explained_variance:
        ev = [float(row["explained_variance_ratio_sum"]) for row in results]
        plt.plot(ranks, [(1 - i) for i in ev], marker="s", linestyle="--", label="1 - explained_variance_ratio_sum")

    plt.xlabel("PCA rank")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_step_output_pca_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean", "coeff_rel_l2"),
    show_explained_variance: bool = True,
    title: str = "1D step output PCA analysis",
):
    ranks = [int(row["pca_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    if show_explained_variance:
        ev = [float(row["explained_variance_ratio_sum"]) for row in results]
        plt.plot(ranks, [(1 - i) for i in ev], marker="s", linestyle="--", label="1 - explained_variance_ratio_sum")

    plt.xlabel("PCA rank")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_continuous_output_rank_grid(
    results: list[dict],
    metric_key: str = "function_rel_l2_mean_vs_raw",
    title: str = "1D continuous output rank grid analysis",
):
    nts = sorted(set(int(row["Nt"]) for row in results))
    kxs = sorted(set(int(row["Kx"]) for row in results))

    Z = np.full((len(nts), len(kxs)), np.nan, dtype=float)
    nt_to_i = {nt: i for i, nt in enumerate(nts)}
    kx_to_j = {kx: j for j, kx in enumerate(kxs)}

    for row in results:
        i = nt_to_i[int(row["Nt"])]
        j = kx_to_j[int(row["Kx"])]
        Z[i, j] = float(row[metric_key])

    plt.figure(figsize=(8, 6))
    im = plt.imshow(Z, origin="lower", aspect="auto")
    plt.xticks(range(len(kxs)), kxs)
    plt.yticks(range(len(nts)), nts)
    plt.xlabel("Kx")
    plt.ylabel("Nt")
    plt.title(title + f" ({metric_key})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def plot_step_output_basis_rank_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean_vs_raw",),
    title: str = "1D step output basis rank analysis",
):
    ranks = [int(row["output_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    plt.xlabel("output basis rank")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_continuous_output_rank_grid(
    data_dir: str,
    indices: np.ndarray,
    Nt_candidates: Sequence[int],
    Kx_candidates: Sequence[int],
):
    """
    Analyze representation error on a 2-parameter grid (Nt, Kx).

    Reference object:
        the true full U(t, x) loaded from disk.

    Error:
        weighted global relative L2 in (t, x).
    """
    Nt_candidates = sorted(set(int(v) for v in Nt_candidates))
    Kx_candidates = sorted(set(int(v) for v in Kx_candidates))

    if len(Nt_candidates) == 0 or len(Kx_candidates) == 0:
        raise ValueError("Nt_candidates and Kx_candidates must not be empty")

    results = []
    for Nt in Nt_candidates:
        for Kx in Kx_candidates:
            errs = []
            for idx in indices:
                idx = int(idx)
                t, x, U = load_solution(data_dir, idx)

                cache = Scheme1Cache(t_grid=t, x_grid=x, Nt=Nt, Kx=Kx)
                y_vec = cache.encode(U)
                U_rec = cache.decode(y_vec, t_grid=t)

                errs.append(weighted_global_rel_l2(U_rec, U, t, x))

            results.append({
                "Nt": int(Nt),
                "Kx": int(Kx),
                "encoded_dim": int(2 * Nt * Kx),
                "function_rel_l2_mean_vs_raw": float(np.mean(errs)),
                "function_rel_l2_median_vs_raw": float(np.median(errs)),
            })

    return results


# ============================================================
# 1d auto-selection helpers
# ============================================================

def auto_select_step_dimensions_for_training(
    *,
    data_dir: str,
    train_indices_full: np.ndarray,
    test_indices_full: np.ndarray,
    analysis_sample_size_train: int,
    analysis_sample_size_test: int,
    t1: float,
    input_rank_candidates: Sequence[int],
    output_rank_candidates: Sequence[int],
    output_basis_tol: float,
    output_basis_metric: str = "function_rel_l2_mean_vs_raw",
    pca_rank_candidates: Sequence[int] = (4, 8, 16, 32, 64),
    output_pca_tol: float = 1e-2,
    output_pca_metric: str = "function_rel_l2_mean",
):
    train_indices_analysis = np.asarray(train_indices_full[:analysis_sample_size_train], dtype=int)
    test_indices_analysis = np.asarray(test_indices_full[:analysis_sample_size_test], dtype=int)

    input_results = analyze_input_truncation_curve_1d(
        data_dir=data_dir,
        indices=test_indices_analysis,
        input_ranks=input_rank_candidates,
    )

    selected_input_rank, selected_input_row, input_status = choose_smallest_rank_under_tolerance(
        results=input_results,
        rank_key="input_rank",
        metric_key="parameter_rel_l2_mean_vs_full",
        tolerance=output_basis_tol,  # you may want a separate input tol later
    )

    output_results = analyze_step_output_basis_rank_curve(
        data_dir=data_dir,
        indices=test_indices_analysis,
        t1=t1,
        output_ranks=output_rank_candidates,
    )

    selected_output_rank, selected_output_row, output_status = choose_smallest_rank_under_tolerance(
        results=output_results,
        rank_key="output_rank",
        metric_key=output_basis_metric,
        tolerance=output_basis_tol,
    )

    pca_results = analyze_step_output_pca_curve(
        data_dir=data_dir,
        train_indices=train_indices_analysis,
        test_indices=test_indices_analysis,
        t1=t1,
        output_rank=int(selected_output_rank),
        pca_ranks=pca_rank_candidates,
    )

    selected_pca_rank, selected_pca_row, pca_status = choose_smallest_rank_under_tolerance(
        results=pca_results,
        rank_key="pca_rank",
        metric_key=output_pca_metric,
        tolerance=output_pca_tol,
    )

    return {
        "selected_input_rank": int(selected_input_rank),
        "selected_output_rank": int(selected_output_rank),
        "selected_pca_rank": int(selected_pca_rank),
        "input_selection": {
            "selected_row": selected_input_row,
            "selection_status": input_status,
            "results": input_results,
        },
        "output_selection": {
            "selected_row": selected_output_row,
            "selection_status": output_status,
            "results": output_results,
        },
        "pca_selection": {
            "selected_row": selected_pca_row,
            "selection_status": pca_status,
            "results": pca_results,
        },
        "analysis_train_indices": train_indices_analysis,
        "analysis_test_indices": test_indices_analysis,
    }


def auto_select_continuous_dimensions_for_training(
    *,
    data_dir: str,
    train_indices_full: np.ndarray,
    test_indices_full: np.ndarray,
    analysis_sample_size_train: int,
    analysis_sample_size_test: int,
    input_rank_candidates: Sequence[int],
    Nt_candidates: Sequence[int],
    Kx_candidates: Sequence[int],
    output_basis_tol: float,
    output_basis_metric: str = "function_rel_l2_mean_vs_raw",
    pca_rank_candidates: Sequence[int] = (4, 8, 16, 32, 64, 128),
    output_pca_tol: float = 1e-2,
    output_pca_metric: str = "function_rel_l2_mean",
):
    train_indices_analysis = np.asarray(train_indices_full[:analysis_sample_size_train], dtype=int)
    test_indices_analysis = np.asarray(test_indices_full[:analysis_sample_size_test], dtype=int)

    input_results = analyze_input_truncation_curve_1d(
        data_dir=data_dir,
        indices=test_indices_analysis,
        input_ranks=input_rank_candidates,
    )

    selected_input_rank, selected_input_row, input_status = choose_smallest_rank_under_tolerance(
        results=input_results,
        rank_key="input_rank",
        metric_key="parameter_rel_l2_mean_vs_full",
        tolerance=output_basis_tol,  # separate input tol can be added later
    )

    grid_results = analyze_continuous_output_rank_grid(
        data_dir=data_dir,
        indices=test_indices_analysis,
        Nt_candidates=Nt_candidates,
        Kx_candidates=Kx_candidates,
    )

    (selected_Nt, selected_Kx), selected_grid_row, grid_status = choose_smallest_pair_under_tolerance(
        results=grid_results,
        nt_key="Nt",
        kx_key="Kx",
        metric_key=output_basis_metric,
        tolerance=output_basis_tol,
    )

    pca_results = analyze_continuous_output_pca_curve(
        data_dir=data_dir,
        train_indices=train_indices_analysis,
        test_indices=test_indices_analysis,
        Nt=int(selected_Nt),
        Kx=int(selected_Kx),
        pca_ranks=pca_rank_candidates,
    )

    selected_pca_rank, selected_pca_row, pca_status = choose_smallest_rank_under_tolerance(
        results=pca_results,
        rank_key="pca_rank",
        metric_key=output_pca_metric,
        tolerance=output_pca_tol,
    )

    return {
        "selected_input_rank": int(selected_input_rank),
        "selected_Nt": int(selected_Nt),
        "selected_Kx": int(selected_Kx),
        "selected_pca_rank": int(selected_pca_rank),
        "input_selection": {
            "selected_row": selected_input_row,
            "selection_status": input_status,
            "results": input_results,
        },
        "grid_selection": {
            "selected_row": selected_grid_row,
            "selection_status": grid_status,
            "results": grid_results,
        },
        "pca_selection": {
            "selected_row": selected_pca_row,
            "selection_status": pca_status,
            "results": pca_results,
        },
        "analysis_train_indices": train_indices_analysis,
        "analysis_test_indices": test_indices_analysis,
    }

# ============================================================
# 4) Full pipeline runner (mirrors your old script structure)
# ============================================================

def run_training_pipeline(
    data_dir: str,
    cache_dir: str,
    task: str,                      # "step" or "continuous" or "stepper"
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    Kx: int,                        # dimensions for Fourier encoding in space (actual output dim is 2*Kx due to sine/cosine)
    Kin:int | None = None,
    t1: float | None = None,        # for step
    Nt: int | None = None,          # for continuous
    model_type: str = "rf",         # "rf" or "nn"
    use_pca: bool = True,
    pca_components: int = 1280,
    pca_application = "out", # "in", "out", "both", "none"
    normalize_in: float = 1.0,
    normalize_out: float = 1.0,
    device: str = "cpu",
    force_rebuild: bool = False,
    t_from: float | None = None,
    t_to: float | None = None,
    shared_kernel: bool = True,
    d_features: int = 8192,
    lengthscale: float = 0.5,
    lam: float = 1e-4,
    eval_count: int = 50,
):
    # ---------- dataset caching ----------
    ensure_dir(cache_dir)

    tag = (
    f"{task}_Kx{Kx}_Kin{Kin}_Nt{Nt}_t1{t1}_"
    f"tfrom{t_from}_tto{t_to}_shared{shared_kernel}")
    train_cache = os.path.join(cache_dir, f"{tag}_train.npz")
    test_cache  = os.path.join(cache_dir, f"{tag}_test.npz")

    if os.path.exists(train_cache) and force_rebuild == False:
        d = load_npz(train_cache)
        Xtr, Ytr, meta = d["X"], d["Y"], d["meta"].item()
    else:
        if task == "step":
            Xtr, Ytr, meta = build_dataset_task_step(data_dir, train_indices, Kx=Kx, t1=float(t1))
        elif task == "continuous":
            Xtr, Ytr, meta, _ = build_dataset_task_continuous_cached(data_dir, train_indices, Nt=Nt, Kx=Kx)
        else:
            raise ValueError("task must be 'step' or 'continuous'")
        save_npz(train_cache, X=Xtr, Y=Ytr, meta=meta)

    if os.path.exists(test_cache) and force_rebuild == False:
        d = load_npz(test_cache)
        Xte, Yte = d["X"], d["Y"]
    else:
        if task == "step":
            Xte, Yte, _ = build_dataset_task_step(data_dir, test_indices, Kx=Kx, t1=float(t1))
        elif task == "continuous":
            Xte, Yte, _, _ = build_dataset_task_continuous_cached(data_dir, test_indices, Nt=Nt, Kx=Kx)

        save_npz(test_cache, X=Xte, Y=Yte, meta=meta)
    Yte_pre_pca = Yte.copy()
    Xte_full = Xte.copy()
    if Kin is None:
        Xtr = Xtr.copy()
        Xte = Xte.copy()
    else:
        Xtr = Xtr[:, :Kin].copy()
        Xte = Xte[:, :Kin].copy()

    Xte_pre_pca = Xte.copy()
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    print("shape check:",
          "Xtr", Xtr.shape, "Ytr", Ytr.shape,
          "Xte", Xte.shape, "Yte", Yte.shape)

    # ---------- PCA ----------
    pca = PCATransform(
        use=use_pca,
        n_components=pca_components,
        path_out=os.path.join(cache_dir, f"{tag}_pca_out.joblib"),
    )
    if use_pca:
        Ytr_p = pca.fit(Ytr)
        Yte_p = pca.load_and_apply(Yte)
        Xtr_p, Ytr_p, Xte_p, Yte_p = Xtr, Ytr_p, Xte, Yte_p
    else:
        Xtr_p, Ytr_p, Xte_p, Yte_p = Xtr, Ytr, Xte, Yte

    # ---------- normalize + torch ----------
    Xtr_t = torch.tensor(Xtr_p, dtype=torch.float32, device=device) * float(normalize_in)
    Ytr_t = torch.tensor(Ytr_p, dtype=torch.float32, device=device) * float(normalize_out)
    Xte_t = torch.tensor(Xte_p, dtype=torch.float32, device=device) * float(normalize_in)
    Yte_t = torch.tensor(Yte_p, dtype=torch.float32, device=device) * float(normalize_out)

    # ---------- train ----------
    trained = {"model_type": model_type, "shared_kernel": shared_kernel}
    if model_type == "nn":
        model = MLP(d_in=Xtr_t.shape[1], d_out=Ytr_t.shape[1], hidden=256, depth=5, dropout=0.0).to(device)
        model = train_mlp(model, Xtr_t, Ytr_t, Xte_t, Yte_t, lr=1e-4, epochs=1000, print_every=100)
        with torch.no_grad():
            Yhat_t = model(Xte_t)

    elif model_type == "rf":
        # Option A: plug in your NN_cff_vec + train_feature_model here.
        # Option B: keep it self-contained with ridge on top of NN_cff_vec feature map.

        # NOTE: setup also here!
        d_features = d_features
        # lam = 1e-4 # NOTE keep small. large normalization produces noise in result
        sigma_in = np.ones(Xtr_t.shape[1])
        sigma_out = np.ones(Ytr_t.shape[1])
        if shared_kernel:
            feat_model = RFF(d_in=d_in, d_features=d_features, lengthscale=lengthscale, trainable_kernel=False, device=device).to(device)
            feat_model.resample()

            with torch.no_grad():
                Phi_tr = feat_model(Xtr_t)   # (n, F)
                Phi_te = feat_model(Xte_t)   # (n, F)

            W = ridge_fit_shared(Phi_tr, Ytr_t, lam=lam)
            Yhat_t = ridge_predict_shared(Phi_te, W)

        else:
            feat_model = NN_cff_vec(
                d_in=d_in,
                d_features=d_features,
                d_out=Ytr_t.shape[1],
                lengthscale=lengthscale,
                sigma_in=sigma_in,
                sigma_out=sigma_out,
                trainable_kernel=False,
                projection_type="Cauchy",
                device=device,
            ).to(device)

            # NOTE: NN_cff_vec must have resample() called at least once
            feat_model.resample()

            with torch.no_grad():
                Phi_tr = feat_model(Xtr_t)   # (n, F, D)
                Phi_te = feat_model(Xte_t)

            W = ridge_fit_multioutput_features(Phi_tr, Ytr_t, lam=lam)  # (F, D)
            Yhat_t = predict_multioutput_features(Phi_te, W)

    else:
        raise ValueError("model_type must be 'rf' or 'nn'")

    # ---------- undo normalization + PCA ----------
    Yhat = (Yhat_t.detach().cpu().numpy()) / float(normalize_out)
    Ytrue = (Yte_t.detach().cpu().numpy()) / float(normalize_out)
    if use_pca:
        Yhat = pca.inverse_Y(Yhat)
        Ytrue = pca.inverse_Y(Ytrue)
    coeff_rel_l2 = rel_l2(Yhat, Ytrue)


    vec_mse = float(np.mean((Yhat - Ytrue) ** 2))
    print("coefficient-space rel L2:", coeff_rel_l2)
    print("vector-space test MSE:", vec_mse)
    function_rel_l2_mean = None
    function_rel_l2_median = None

    if task == "step":
        function_rel_l2_mean, function_rel_l2_median = eval_step_function_error(
            Yhat, data_dir, test_indices, t1, meta, eval_count=eval_count
        )
        print(
            "function-space rel L2 (step) mean/median:",
            function_rel_l2_mean,
            function_rel_l2_median,
        )
    else:
        function_rel_l2_mean, function_rel_l2_median = eval_continuous_function_error(
            Yhat, data_dir, test_indices, Nt=Nt, Kx=Kx, eval_count=eval_count
        )
        print(
            "function-space rel L2 (continuous) mean/median:",
            function_rel_l2_mean,
            function_rel_l2_median,
        )
    i = 0
    idx0 = int(test_indices[i])
    t_grid, x_grid, U = load_solution(data_dir, idx0)

    if task == "step":
        # compare predicted u(t1, x) to true u(t1, x)
        ti = nearest_time_index(t_grid, float(t1))
        y_meta = meta["y_meta"]
        # Overwrite meta's t_index/t_value to match ti if train/test grids differ (rare, but safe)
        y_meta_local = dict(y_meta)
        y_meta_local["t_index"] = int(ti)
        y_meta_local["t_value"] = float(t_grid[ti])

        u_pred = decode_output_scheme2_slice_fourier(Yhat[i], y_meta_local)
        u_true = U[ti, :]
        print("function-space rel error (step):", rel_l2(u_pred, u_true))


    else:
        # continuous: decode to U(t,x) and compute weighted L2 relative error
        y_meta = {
            "Nt": meta["Nt"],
            "Kx": meta["Kx"],
            "N_x": len(meta["x_grid"]),
            "t0": float(meta["t_grid"][0]),
            "t1": float(meta["t_grid"][-1]),
        }
        U_pred = decode_output_scheme1_legendre_time_fourier_space(Yhat[i], y_meta, t_grid)

        wt = trapezoid_weights(t_grid)
        wx = np.full(len(x_grid), 1.0 / len(x_grid))
        err2 = np.sum((wt[:, None] * (U - U_pred) ** 2) * wx[None, :])
        ref2 = np.sum((wt[:, None] * (U) ** 2) * wx[None, :]) + 1e-12
        print("function-space rel L2 (continuous):", float(np.sqrt(err2 / ref2)))

    return {
        "vec_mse": vec_mse,
        "coeff_rel_l2": coeff_rel_l2,
        "function_rel_l2_mean": function_rel_l2_mean,
        "function_rel_l2_median": function_rel_l2_median,
        "Yhat": Yhat,
        "Ytrue": Ytrue,
        "meta": meta,
        "pca": pca,
        "trained": trained,
        "d_features": int(d_features),
        "lengthscale": float(lengthscale),
        "lam": float(lam),
    }

def sweep_feature_sizes(
    *,
    data_dir: str,
    cache_dir: str,
    task: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_sizes: Sequence[int],
    Kx: int,
    Kin: int | None = None,
    t1: float | None = None,
    Nt: int | None = None,
    model_type: str = "rf",
    use_pca: bool = True,
    pca_components: int = 1280,
    normalize_in: float = 1.0,
    normalize_out: float = 1.0,
    device: str = "cpu",
    force_rebuild: bool = False,
    shared_kernel: bool = True,
    lengthscale: float = 0.5,
    lam: float = 1e-4,
    eval_count: int = 50,
):
    rows = []

    for d_features in feature_sizes:
        print("\n" + "=" * 80)
        print(f"Feature-size sweep: d_features={int(d_features)}")
        print("=" * 80)

        out = run_training_pipeline(
            data_dir=data_dir,
            cache_dir=cache_dir,
            task=task,
            train_indices=train_indices,
            test_indices=test_indices,
            Kx=Kx,
            Kin=Kin,
            t1=t1,
            Nt=Nt,
            model_type=model_type,
            use_pca=use_pca,
            pca_components=pca_components,
            normalize_in=normalize_in,
            normalize_out=normalize_out,
            device=device,
            force_rebuild=force_rebuild,
            shared_kernel=shared_kernel,
            d_features=int(d_features),
            lengthscale=lengthscale,
            lam=lam,
            eval_count=eval_count,
        )

        rows.append({
            "task": task,
            "shared_kernel": bool(shared_kernel),
            "d_features": int(d_features),
            "vec_mse": float(out["vec_mse"]),
            "coeff_rel_l2": float(out["coeff_rel_l2"]),
            "function_rel_l2_mean": float(out["function_rel_l2_mean"]),
            "function_rel_l2_median": float(out["function_rel_l2_median"]),
        })

    return rows

def plot_feature_size_sweep(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean", "coeff_rel_l2"),
    title: str = "Feature-size sweep",
):
    feature_sizes = [int(row["d_features"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(feature_sizes, values, marker="o", label=metric_key)

    plt.xlabel("d_features")
    plt.ylabel("metric")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# visualization for task B, this is more complicated
def visualize_continuous_result(
    *,
    cache: Scheme1Cache,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    U_true: np.ndarray,          # (T, N_x) ground truth for one sample
    y_true_vec: np.ndarray,      # (2*Nt*Kx,)
    y_hat_vec: np.ndarray,       # (2*Nt*Kx,)
    title_prefix: str = "",
):
    # decode both
    U_hat = cache.decode(y_hat_vec, t_grid=t_grid)
    U_rec_true = cache.decode(y_true_vec, t_grid=t_grid)  # this is truncated representation of the truth

    # --- scalar error curves over time ---
    # relative L2 in x for each time
    eps = 1e-12
    rel_t_pred = np.linalg.norm(U_hat - U_true, axis=1) / (np.linalg.norm(U_true, axis=1) + eps)
    rel_t_repr = np.linalg.norm(U_rec_true - U_true, axis=1) / (np.linalg.norm(U_true, axis=1) + eps)

    # --- pick a few time slices to plot ---
    import matplotlib.pyplot as plt
    pick_ts = [0, len(t_grid)//4, len(t_grid)//2, 3*len(t_grid)//4, len(t_grid)-1]

    for ti in pick_ts:
        plt.figure()
        plt.plot(x_grid, U_true[ti, :], label="true")
        plt.plot(x_grid, U_rec_true[ti, :], label="true (truncated repr)", alpha=0.7)
        plt.plot(x_grid, U_hat[ti, :], label="pred (decoded)", alpha=0.8)
        plt.legend()
        plt.title(f"{title_prefix} t={t_grid[ti]:.3f} slice")
        plt.show()

    # --- error over time plot ---
    plt.figure()
    plt.semilogy(t_grid, rel_t_pred, label="pred vs true")
    plt.semilogy(t_grid, rel_t_repr, label="repr(truth) vs true", alpha=0.8)
    plt.xlabel("t")
    plt.ylabel("relative L2 over x")
    plt.title(f"{title_prefix} error vs time")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    # --- global relative errors (time-space) ---
    wt = trapezoid_weights(t_grid)  # (T,)
    wx = np.full(len(x_grid), 1.0 / len(x_grid))
    err2_pred = np.sum((wt[:, None] * (U_hat - U_true) ** 2) * wx[None, :])
    err2_repr = np.sum((wt[:, None] * (U_rec_true - U_true) ** 2) * wx[None, :])
    ref2 = np.sum((wt[:, None] * (U_true) ** 2) * wx[None, :]) + eps

    print("Global rel L2 (pred):", float(np.sqrt(err2_pred / ref2)))
    print("Global rel L2 (repr bound):", float(np.sqrt(err2_repr / ref2)))

# ============================================================
# 5) Example invocation
# ============================================================

if __name__ == "__main__":
    here = Path(__file__).resolve().parent          # folder that contains this file
    data_root = here.parent / "data"    
    data_dir = data_root / "viscous_burgers_utx_1d"
    cache_dir = data_root / "cache" / "burgers_1d"
    data_dir = str(data_dir)
    cache_dir = str(cache_dir)

    t1 = 0.5 # time step size for step task

    # test: if the shock can be expressed by the expansion well at t almost 1. 

    # EXPERIMENT NOTE:
    Kx = 128 # number of Fourier modes for output encoding--- actual dimension Kx*2
    Nt = 32
    # due to the shock structure we need a larger tail for this problem. 
    # TODO: check the distribution/decay of the parameters of Ys. 
    # choose indices
    train_size = 9000
    test_size = 1000
    assert train_size + test_size <= 10000, "total samples exceed available data"
    train_indices = np.arange(0, train_size)
    test_indices  = np.arange(train_size, train_size + test_size)
    if_PCA = False
    pca_components = 320 # TODO this is placeholder. choose the dimension automatically

    tasks =  ['step'] # options: ['distribution','step','continuous']
    # --------- test for wave reconstrunction: shock case
    if 'distribution' in tasks:
        Kx_list = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
        curve = expansion_error_curve_for_one_snapshot(
            data_dir=data_dir,
            idx=test_indices[0],
            t1=t1,
            Kx_list=Kx_list,
            load_solution_fn=load_solution,
            encode_fn=encode_output_scheme2_slice_fourier,
            decode_fn=decode_output_scheme2_slice_fourier,
            use_energy_tail=True,
        )
        plot_expansion_error_curve(curve)

    # conclusion: dimension \asymp 10^2 is needed for loyal reconstruction of shock structure. 

    if 'step_dim_analysis' in tasks:
        step_auto = auto_select_step_dimensions_for_training(
            data_dir=data_dir,
            train_indices_full=train_indices,
            test_indices_full=test_indices,
            analysis_sample_size_train=1000,
            analysis_sample_size_test=200,
            t1=t1,
            input_rank_candidates=[4, 6, 8, 10, 12, 16, 20],
            output_rank_candidates=[4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128],
            output_basis_tol=5e-2,
            output_basis_metric="function_rel_l2_mean_vs_raw",
            pca_rank_candidates=[4, 8, 16, 32, 64, 128],
            output_pca_tol=1e-2,
            output_pca_metric="function_rel_l2_mean",
        )

        print("\nStep-task automatic dimension selection:")
        print("selected_input_rank:", step_auto["selected_input_rank"])
        print("selected_output_rank:", step_auto["selected_output_rank"])
        print("selected_pca_rank:", step_auto["selected_pca_rank"])

        print_rank_curve_table(step_auto["input_selection"]["results"], "1D input truncation analysis")
        print_rank_curve_table(step_auto["output_selection"]["results"], "1D step output basis analysis")
        print_rank_curve_table(step_auto["pca_selection"]["results"], "1D step output PCA analysis")

        plot_input_truncation_curve_1d(step_auto["input_selection"]["results"])
        plot_step_output_basis_rank_curve(step_auto["output_selection"]["results"])
        plot_step_output_pca_curve(step_auto["pca_selection"]["results"])


    if 'continuous_dim_analysis' in tasks:
        cont_auto = auto_select_continuous_dimensions_for_training(
            data_dir=data_dir,
            train_indices_full=train_indices,
            test_indices_full=test_indices,
            analysis_sample_size_train=1000,
            analysis_sample_size_test=200,
            input_rank_candidates=[4, 6, 8, 10, 12, 16, 20],
            Nt_candidates=[4, 8, 12, 16, 20, 24],
            Kx_candidates=[8, 16, 24, 32, 48, 64, 96, 128],
            output_basis_tol=5e-2,
            output_basis_metric="function_rel_l2_mean_vs_raw",
            pca_rank_candidates=[8, 16, 32, 64, 128, 256, 512],
            output_pca_tol=1e-3,
            output_pca_metric="function_rel_l2_mean",
        )

        print("\nContinuous-task automatic dimension selection:")
        print("selected_input_rank:", cont_auto["selected_input_rank"])
        print("selected_Nt:", cont_auto["selected_Nt"])
        print("selected_Kx:", cont_auto["selected_Kx"])
        print("selected_pca_rank:", cont_auto["selected_pca_rank"])

        print_rank_curve_table(cont_auto["input_selection"]["results"], "1D input truncation analysis")
        print_rank_curve_table(cont_auto["grid_selection"]["results"], "1D continuous output grid analysis")
        print_rank_curve_table(cont_auto["pca_selection"]["results"], "1D continuous output PCA analysis")

        plot_input_truncation_curve_1d(cont_auto["input_selection"]["results"])
        plot_continuous_output_rank_grid(cont_auto["grid_selection"]["results"])
        plot_continuous_output_pca_curve(cont_auto["pca_selection"]["results"])

        Kx = cont_auto["selected_Kx"]
        Nt = cont_auto["selected_Nt"]
        pca_components = cont_auto["selected_pca_rank"]
        if_PCA = cont_auto["selected_pca_rank"] > 0
    if 'step' in tasks:
        # --- Task A: u0 -> u(t1, x) ---
        outA = run_training_pipeline(
            data_dir=data_dir,
            cache_dir=cache_dir,
            task="step",
            train_indices=train_indices,
            test_indices=test_indices,
            Kx=Kx, 
            t1=t1,
            Nt=None,
            model_type="rf",
            use_pca=if_PCA,
            pca_components=pca_components,
            normalize_in=1.0,
            normalize_out=1.0,
            device="cpu",
            force_rebuild=True, # activate when changes in training set incurred
        )
        print("Task A (step) results:", outA)
        Yhat_val = outA["Yhat"]
        Ytrue_val = outA["Ytrue"]
        t_grid = outA["meta"]["t_grid"]
        pca = outA["pca"]
        for i in range(min(10, Yhat_val.shape[0])):
            print(f"Example {i}: Yhat norm {np.linalg.norm(Yhat_val[i]):.4e}, Ytrue norm {np.linalg.norm(Ytrue_val[i]):.4e}")
            print('(Yhat-Ytrue)/Y_true', (Yhat_val[i] - Ytrue_val[i])/np.abs(Ytrue_val[i]))
            # reconstruct and plot the predicted vs true u(t1,x) for this example
            y_hat = decode_output_scheme2_slice_fourier(Yhat_val[i], outA["meta"]['y_meta'])
            y_true = decode_output_scheme2_slice_fourier(Ytrue_val[i], outA["meta"]['y_meta'])
            y_true_truncated = decode_output_scheme2_slice_fourier(Ytrue_val[i][:128], outA["meta"]['y_meta'])
            # TODO think of better evaluation methods
            plt.plot(y_hat, label="Yhat")
            plt.plot(y_true, label="Ytrue", alpha=0.7)
            plt.plot(y_true_truncated, label=f"Ytrue in the first {Kx} basis", alpha=0.7)
            plt.legend()
            plt.title(f"Example {i} - Task A (step)")
            plt.show()
    if 'continuous' in tasks:
        # --- Task B: u0 -> U(t,x) (Legendre time, Fourier space) ---
        
        outB = run_training_pipeline(
            data_dir=data_dir,
            cache_dir=cache_dir,
            task="continuous",
            train_indices=train_indices,
            test_indices=test_indices,
            Kx=Kx,
            t1=None,
            Nt=Nt,
            model_type="rf",
            use_pca=True,
            pca_components=pca_components,
            normalize_in=1.0,
            normalize_out=1.0,
            device="cpu",
        )
        # PCA test on effectiveness of further dimension reduction
        ranks = (256, 512)  # adjust; must be <= 2*Nt*Kx
        pca_compression_study_on_Y(
            outB=outB,
            data_dir=data_dir,
            test_indices=test_indices,
            ranks=ranks,
            n_show=3,   # 先小一点看趋势
        )

        # Recreate cache for decoding/visualization from meta (cheap, deterministic)
        meta = outB["meta"]
        I = np.eye(Nt)
        I_fro = np.linalg.norm(I, ord="fro")
        cacheB = Scheme1Cache(t_grid=meta["t_grid"], x_grid=meta["x_grid"], Nt=meta["Nt"], Kx=meta["Kx"])
        print("||Gt - I||:", cacheB.Gt_minus_I_norm)
        print("||Gt - I||/||I||:", cacheB.Gt_minus_I_norm / I_fro)
        print('conditional number of Gt', cacheB.Gt_cond)

        # Pick one test sample for detailed visualization
        i = 0
        idx0 = int(test_indices[i])
        t_grid, x_grid, U_true = load_solution(data_dir, idx0)

        y_true_vec = outB["Ytrue"][i]
        y_hat_vec = outB["Yhat"][i]

        visualize_continuous_result(
            cache=cacheB,
            t_grid=t_grid,
            x_grid=x_grid,
            U_true=U_true,
            y_true_vec=y_true_vec,
            y_hat_vec=y_hat_vec,
            title_prefix="Task B",
        )

#     Kout_list = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]  # <= d_in_full
#     rows = sweep_input_truncation(
#         data_dir=data_dir,
#         cache_dir=cache_dir,
#         t1=1.0,
#         Kx_out=64,
#         Kout_list=Kout_list,
#         train_indices=train_indices,
#         test_indices=test_indices,
#         model_type="rf",
#         device="cpu",
# )
    # Yhat_val = outA["Yhat"]
    # Ytrue_val = outA["Ytrue"]
    # t_grid = outA["meta"]["t_grid"]
    # pca = outA["pca"]
    # for i in range(min(10, Yhat_val.shape[0])):
    #     print(f"Example {i}: Yhat norm {np.linalg.norm(Yhat_val[i]):.4e}, Ytrue norm {np.linalg.norm(Ytrue_val[i]):.4e}")
    #     print('(Yhat-Ytrue)/Y_true', (Yhat_val[i] - Ytrue_val[i])/np.abs(Ytrue_val[i]))
    #     # reconstruct and plot the predicted vs true u(t1,x) for this example
    #     y_hat = decode_output_scheme2_slice_fourier(Yhat_val[i], outA["meta"]['y_meta'])
    #     y_true = decode_output_scheme2_slice_fourier(Ytrue_val[i], outA["meta"]['y_meta'])
    #     y_true_truncated = decode_output_scheme2_slice_fourier(Ytrue_val[i][:128], outA["meta"]['y_meta'])
    #     # TODO think of better evaluation methods
    #     plt.plot(y_hat, label="Yhat")
    #     plt.plot(y_true, label="Ytrue", alpha=0.7)
    #     plt.plot(y_true_truncated, label=f"Ytrue in the first {Kx} basis", alpha=0.7)
    #     plt.legend()
    #     plt.title(f"Example {i} - Task A (step)")
    #     plt.show()