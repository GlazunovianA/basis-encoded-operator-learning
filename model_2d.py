from __future__ import annotations

import os
import sys
import random
import importlib
import argparse
from pathlib import Path
from typing import Sequence, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from test_representation_system.ed_2d import decode_field_l2_2d, project_on_L2_basis_2d, evaluate_series_L2
from util_scaling import z_flatten, z_reconstruction, sigma_generate
from G_N import RFF, NN_cff_vec
from neural_network import FullyConnectedNN

# NOTE remember to delete the cache for refactoring every time. 

# ============================================================
# 0) Small helpers
# ============================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_npz(path: str, **kwargs):
    ensure_dir(os.path.dirname(path))
    np.savez(path, **kwargs)


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def flatten_coeff_tensor(A: np.ndarray) -> np.ndarray:
    """
    A: (n, N1, N2) -> flattened with your z-order flattening.
    For the current pipeline we still assume square coefficient tensors,
    but input/output squares may now have different sizes.
    """
    return z_flatten(A)


def reconstruct_coeff_tensor(Z: np.ndarray) -> np.ndarray:
    """
    Z: (n, N^2) -> (n, N, N) using your inverse flattening.
    """
    return z_reconstruction(Z)


def maybe_truncate_coeff_tensor(A: np.ndarray, N_ex_row: int, N_ex_col: int | None = None) -> np.ndarray:
    if N_ex_col is None:
        N_ex_col = N_ex_row
    return A[:, :N_ex_row, :N_ex_col]


# ============================================================
# 0) dimension selection helpers
# ============================================================

def choose_smallest_rank_under_tolerance(
    results: list[dict],
    rank_key: str,
    metric_key: str,
    tolerance: float,
):
    """
    results:
        list of dicts containing rank_key and metric_key

    returns:
        selected_rank, selected_row, selection_status
    """
    if len(results) == 0:
        raise ValueError("results must not be empty")

    rows_sorted = sorted(results, key=lambda row: int(row[rank_key]))

    for row in rows_sorted:
        metric_value = float(row[metric_key])
        if metric_value <= float(tolerance):
            return int(row[rank_key]), row, "threshold_met"

    return int(rows_sorted[-1][rank_key]), rows_sorted[-1], "threshold_not_met"


def auto_select_output_rank(
    *,
    data_dir: str,
    equation: str,
    analysis_indices: np.ndarray,
    basis_u: str,
    output_rank_candidates: Sequence[int],
    reference_output_rank: int,
    output_basis_tol: float,
    output_basis_metric: str = "function_rel_l2_mean_vs_ref",
    N_quad: int = 50,
):
    """
    Select the smallest output expansion rank whose basis reconstruction
    error is below the prescribed tolerance.

    Reference object:
        largest tested/reference basis rank reconstruction.
    """
    all_ranks = sorted(set([int(reference_output_rank)] + [int(r) for r in output_rank_candidates]))
    results = analyze_output_basis_rank_curve(
        data_dir=data_dir,
        equation=equation,
        indices=analysis_indices,
        basis_u=basis_u,
        output_ranks=all_ranks,
        N_quad=N_quad,
    )

    candidate_results = [
        row for row in results
        if int(row["output_rank"]) in set(int(r) for r in output_rank_candidates)
    ]

    selected_rank, selected_row, selection_status = choose_smallest_rank_under_tolerance(
        results=candidate_results,
        rank_key="output_rank",
        metric_key=output_basis_metric,
        tolerance=output_basis_tol,
    )

    return {
        "selected_output_rank": int(selected_rank),
        "selected_row": selected_row,
        "selection_status": selection_status,
        "metric_used": output_basis_metric,
        "tolerance": float(output_basis_tol),
        "results": results,
    }


def auto_select_output_pca_rank(
    *,
    data_dir: str,
    equation: str,
    train_indices_analysis: np.ndarray,
    test_indices_analysis: np.ndarray,
    basis_u: str, 
    selected_output_rank: int,
    pca_rank_candidates: Sequence[int],
    output_pca_tol: float,
    output_pca_metric: str = "function_rel_l2_mean",
    N_quad: int = 50,
):
    """
    Select the smallest PCA rank whose reconstruction error on the
    basis-encoded outputs is below the prescribed tolerance.
    """
    Ytr_mat = build_output_only_dataset_for_rank(
        data_dir=data_dir,
        equation=equation,
        indices=train_indices_analysis,
        N_ex_output=selected_output_rank,
        basis_u=basis_u,
        N_quad=N_quad,
    )
    Yte_mat = build_output_only_dataset_for_rank(
        data_dir=data_dir,
        equation=equation,
        indices=test_indices_analysis,
        N_ex_output=selected_output_rank,
        basis_u=basis_u,
        N_quad=N_quad,
    )

    results = analyze_output_pca_curve(
        Ytr_mat=Ytr_mat,
        Yte_mat=Yte_mat,
        basis_u=basis_u,
        N_quad=N_quad,
        pca_ranks=pca_rank_candidates,
        eval_count=min(50, len(Yte_mat)),
    )

    selected_rank, selected_row, selection_status = choose_smallest_rank_under_tolerance(
        results=results,
        rank_key="pca_rank",
        metric_key=output_pca_metric,
        tolerance=output_pca_tol,
    )

    return {
        "selected_pca_rank": int(selected_rank),
        "selected_row": selected_row,
        "selection_status": selection_status,
        "metric_used": output_pca_metric,
        "tolerance": float(output_pca_tol),
        "results": results,
    }


def auto_select_dimensions_for_training(
    *,
    data_dir: str,
    equation: str,
    basis_u: str,
    train_indices_full: np.ndarray,
    test_indices_full: np.ndarray,
    analysis_sample_size_train: int,
    analysis_sample_size_test: int,
    output_rank_candidates: Sequence[int],
    reference_output_rank: int,
    output_basis_tol: float,
    output_basis_metric: str,
    pca_rank_candidates: Sequence[int],
    output_pca_tol: float,
    output_pca_metric: str,
    N_quad: int = 50,
):
    """
    Full automatic dimension selection:

    1) choose output basis rank
    2) choose PCA rank conditional on chosen output basis rank
    """
    train_indices_analysis = np.asarray(train_indices_full[:analysis_sample_size_train], dtype=int)
    test_indices_analysis = np.asarray(test_indices_full[:analysis_sample_size_test], dtype=int)

    basis_selection = auto_select_output_rank(
        data_dir=data_dir,
        equation=equation,
        analysis_indices=test_indices_analysis,
        basis_u=basis_u,
        output_rank_candidates=output_rank_candidates,
        reference_output_rank=reference_output_rank,
        output_basis_tol=output_basis_tol,
        output_basis_metric=output_basis_metric,
        N_quad=N_quad,
    )

    selected_output_rank = int(basis_selection["selected_output_rank"])

    pca_selection = auto_select_output_pca_rank(
        data_dir=data_dir,
        equation=equation,
        train_indices_analysis=train_indices_analysis,
        test_indices_analysis=test_indices_analysis,
        basis_u=basis_u,
        selected_output_rank=selected_output_rank,
        pca_rank_candidates=pca_rank_candidates,
        output_pca_tol=output_pca_tol,
        output_pca_metric=output_pca_metric,
        N_quad=N_quad,
    )

    return {
        "selected_output_rank": selected_output_rank,
        "selected_pca_rank": int(pca_selection["selected_pca_rank"]),
        "basis_selection": basis_selection,
        "pca_selection": pca_selection,
        "analysis_train_indices": train_indices_analysis,
        "analysis_test_indices": test_indices_analysis,
    }


# ============================================================
# 1) Data loading / encoding for 2D operator task
# ============================================================

def load_fem_array(path: str):
    """
    Expects .npy with columns [x, y, value]
    """
    arr = np.load(path)
    x, y, values = arr[:, 0], arr[:, 1], arr[:, 2]
    return x, y, values


def load_raw_test_fields(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
):
    """
    Load raw FEM input/solution fields for visualization only.

    Returns:
        raw_inputs: list[dict]
            each dict has keys: x, y, values
        raw_solutions: list[dict]
            each dict has keys: x, y, values
    """
    raw_inputs = []
    raw_solutions = []

    for idx in indices:
        idx = int(idx)

        param_path = os.path.join(data_dir, equation, f"fem_parameter_{idx}.npy")
        sol_path = os.path.join(data_dir, equation, f"fem_solution_{idx}.npy")

        xp, yp, vp = load_fem_array(param_path)
        xu, yu, vu = load_fem_array(sol_path)

        raw_inputs.append({
            "x": xp,
            "y": yp,
            "values": vp,
        })
        raw_solutions.append({
            "x": xu,
            "y": yu,
            "values": vu,
        })

    return raw_inputs, raw_solutions

def load_input_coeff_tensor(
    data_dir: str,
    equation: str,
    idx: int,
    N_ex_input: int,
) -> np.ndarray:
    """
    Load precomputed input coefficient tensor from disk.

    Expected file:
        data/{equation}/input_coeff_tensor_{idx}.npy

    Returns:
        coeffs_param: shape (N_ex_input, N_ex_input)
    """
    coeff_path = os.path.join(data_dir, equation, f"input_coeff_tensor_{idx}.npy")
    coeffs_param = np.load(coeff_path)

    coeffs_param = np.asarray(coeffs_param, dtype=float)
    if coeffs_param.ndim != 2:
        raise ValueError(
            f"Expected 2D input coefficient tensor at {coeff_path}, got shape {coeffs_param.shape}"
        )

    if coeffs_param.shape[0] < N_ex_input or coeffs_param.shape[1] < N_ex_input:
        raise ValueError(
            f"Input coeff tensor at {coeff_path} has shape {coeffs_param.shape}, "
            f"smaller than requested N_ex_input={N_ex_input}"
        )

    return coeffs_param[:N_ex_input, :N_ex_input]

def build_dataset_2d_operator(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
    N_ex_input: int,
    N_ex_output: int,
    basis_a: str,
    basis_u: str,
    N_quad: int = 50,
):
    """
    Build dataset for operator learning:
        a(x,y) -> u(x,y)

    Returns:
        X: (n, N_ex_input,  N_ex_input)
        Y: (n, N_ex_output, N_ex_output)
        meta: dict
    """
    X_list, Y_list = [], []
    meta_ref = None

    for idx in indices:
        idx = int(idx)

        param_path = os.path.join(data_dir, equation, f"fem_parameter_{idx}.npy")
        sol_path = os.path.join(data_dir, equation, f"fem_solution_{idx}.npy")

        xp, yp, vp = load_fem_array(param_path)
        xu, yu, vu = load_fem_array(sol_path)

        # coeffs_param, meta_param = encode_single_field_to_coeffs(
        #     xp, yp, vp, N_ex=N_ex_input, basis_type=basis_a, N_quad=N_quad
        # )

        coeffs_param = load_input_coeff_tensor(
            data_dir=data_dir,
            equation=equation,
            idx=idx,
            N_ex_input=N_ex_input,
        )
        meta_param = {
            "source": "precomputed_input_coeff_tensor",
            "N_ex": int(N_ex_input),
            "basis_type": basis_a,
        }
        coeffs_sol, meta_sol = encode_single_field_to_coeffs(
            xu, yu, vu, N_ex=N_ex_output, basis_type=basis_u, N_quad=N_quad
        )

        X_list.append(coeffs_param)
        Y_list.append(coeffs_sol)

        if meta_ref is None:
            meta_ref = {
                "task": "operator_2d",
                "equation": equation,
                "N_ex_input": int(N_ex_input),
                "N_ex_output": int(N_ex_output),
                "basis_a": basis_a,
                "basis_u": basis_u,
                "N_quad": int(N_quad),
                "d_in_matrix": (int(N_ex_input), int(N_ex_input)),
                "d_out_matrix": (int(N_ex_output), int(N_ex_output)),
                "d_in": int(N_ex_input * N_ex_input),
                "d_out": int(N_ex_output * N_ex_output),
                "param_meta": meta_param,
                "sol_meta": meta_sol,
            }

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta_ref


# ============================================================
# 2) Optional PCA wrapper
# ============================================================

class PCATransform:
    def __init__(self, use: bool, n_components: int, path_out: str):
        self.use = bool(use)
        self.n_components = int(n_components)
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
# 3) Training backends
# ============================================================

def ridge_fit_multioutput_features(features: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    features: (n, F, D)
    Y:        (n, D)
    Returns W: (F, D)
    """
    n, F, D = features.shape
    W = torch.empty((F, D), dtype=features.dtype, device=features.device)
    I = torch.eye(F, dtype=features.dtype, device=features.device)

    for d in range(D):
        Phi = features[:, :, d]
        A = Phi.T @ Phi + lam * I
        b = Phi.T @ Y[:, d]
        W[:, d] = torch.linalg.solve(A, b)
    return W

# update of the above::
def ridge_fit_multioutput_features_regime_aware(features: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    features: (n, F, D)
    Y:        (n, D)
    Returns:
        W: (F, D)
    """
    features64 = features.to(torch.float64)
    Y64 = Y.to(torch.float64)

    n, F, D = features64.shape
    W64 = torch.empty((F, D), dtype=torch.float64, device=features.device)

    if F <= n:
        I_F = torch.eye(F, dtype=torch.float64, device=features.device)
        for d in range(D):
            Phi_d = features64[:, :, d]             # (n, F)
            A = Phi_d.T @ Phi_d + lam * I_F         # (F, F)
            b = Phi_d.T @ Y64[:, d:d+1]             # (F, 1)
            L = torch.linalg.cholesky(A)
            w_d = torch.cholesky_solve(b, L)        # (F, 1)
            W64[:, d] = w_d[:, 0]
    else:
        I_n = torch.eye(n, dtype=torch.float64, device=features.device)
        for d in range(D):
            Phi_d = features64[:, :, d]             # (n, F)
            K = Phi_d @ Phi_d.T + lam * I_n         # (n, n)
            y_d = Y64[:, d:d+1]                     # (n, 1)
            L = torch.linalg.cholesky(K)
            alpha_d = torch.cholesky_solve(y_d, L)  # (n, 1)
            w_d = Phi_d.T @ alpha_d                 # (F, 1)
            W64[:, d] = w_d[:, 0]

    return W64.to(features.dtype)


def predict_multioutput_features(features: torch.Tensor, W: torch.Tensor):
    return torch.einsum("nfd,fd->nd", features, W)


def ridge_fit_shared(Phi: torch.Tensor, Y: torch.Tensor, lam: float):
    _, F = Phi.shape
    I = torch.eye(F, dtype=Phi.dtype, device=Phi.device)
    A = Phi.T @ Phi + lam * I
    B = Phi.T @ Y
    W = torch.linalg.solve(A, B)
    return W

def ridge_fit_shared_regime_aware(Phi: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    Phi: (n, F)
    Y:   (n, D)
    Returns:
        W: (F, D)
    """
    Phi64 = Phi.to(torch.float64)
    Y64 = Y.to(torch.float64)

    n, F = Phi64.shape

    if F <= n:
        I_F = torch.eye(F, dtype=Phi64.dtype, device=Phi64.device)
        A = Phi64.T @ Phi64 + lam * I_F
        B = Phi64.T @ Y64
        L = torch.linalg.cholesky(A)
        W64 = torch.cholesky_solve(B, L)
    else:
        I_n = torch.eye(n, dtype=Phi64.dtype, device=Phi64.device)
        K = Phi64 @ Phi64.T + lam * I_n
        L = torch.linalg.cholesky(K)
        Alpha = torch.cholesky_solve(Y64, L)   # (n, D)
        W64 = Phi64.T @ Alpha                  # (F, D)

    return W64.to(Phi.dtype)


def ridge_predict_shared(Phi: torch.Tensor, W: torch.Tensor):
    return Phi @ W


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


# ============================================================
# 4) Evaluation helpers for 2D
# ============================================================

from scipy.interpolate import LinearNDInterpolator, griddata

def encode_single_field_to_coeffs(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    N_ex: int,
    basis_type: str,
    N_quad: int = 50,
):
    coeffs, f_rec, Xq, Yq, f_quad = project_on_L2_basis_2d(
        x, y, values, N_basis = N_ex, N_quad=N_quad, basis_type=basis_type
    )
    return coeffs, {
        "f_rec": f_rec,
        "Xq": Xq,
        "Yq": Yq,
        "f_quad": f_quad,
        "basis_type": basis_type,
        "N_quad": int(N_quad),
        "N_ex": int(N_ex),
    }


def interpolate_scattered_field_to_points(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    values_raw: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
):
    """
    Interpolate scattered FEM data onto target points.

    Inputs:
        x_raw, y_raw, values_raw: raw FEM scattered data
        x_target, y_target: target coordinates with same shape

    Returns:
        values_target with same shape as x_target / y_target
    """
    interpolator = LinearNDInterpolator(
        list(zip(np.asarray(x_raw), np.asarray(y_raw))),
        np.asarray(values_raw),
        fill_value=np.nan,
    )
    values_target = interpolator(x_target, y_target)
    return np.asarray(values_target, dtype=float)

def plot_field_imshow(ax, field, title, transpose=False):
    arr = field.T if transpose else field
    im = ax.imshow(
        arr,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        cmap="coolwarm",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def coefficient_errors(pred: np.ndarray, true: np.ndarray):
    mse = float(np.mean((pred - true) ** 2))
    rel = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
    return {"mse": mse, "rel_l2": rel}


def evaluate_function_error_2d(
    pred_coeffs: np.ndarray,
    true_coeffs: np.ndarray,
    basis_type: str,
    N_quad: int = 50,
):
    """
    Compare reconstructed function values on quadrature grid.
    """
    f_pred = evaluate_series_L2(pred_coeffs, N_quad=N_quad, basis_type=basis_type)
    f_true = evaluate_series_L2(true_coeffs, N_quad=N_quad, basis_type=basis_type)

    abs_err = f_pred - f_true
    rel = np.linalg.norm(abs_err) / (np.linalg.norm(f_true) + 1e-12)
    return {
        "f_pred": f_pred,
        "f_true": f_true,
        "abs_err": abs_err,
        "rel_l2": float(rel),
    }

def evaluate_test_set_function_error_2d( pred_tensor: np.ndarray, true_tensor: np.ndarray, basis_type: str, N_quad: int = 50, eval_count: int = 50, ): 
    errs = [] 
    m = min(eval_count, len(pred_tensor)) 
    for i in range(m): 
        out = evaluate_function_error_2d( 
            pred_coeffs=pred_tensor[i], 
            true_coeffs=true_tensor[i], 
            basis_type=basis_type, N_quad=N_quad, ) 
        errs.append(out["rel_l2"]) 
    return float(np.mean(errs)), float(np.median(errs))


def evaluate_series_to_plot_points(
    coeffs: np.ndarray,
    basis_type: str,
    domain: list[list[float]] | None = None,
    N_quad: int = 50,
    N_quad_for_orth: int = 200,
):
    field, Xq, Yq, meta = decode_field_l2_2d(
        coeffs=coeffs,
        basis_type=basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )
    return {
        "field": field,
        "Xq": Xq,
        "Yq": Yq,
        "meta": meta,
    }


# def interpolate_raw_field_to_grid(
#     x_raw: np.ndarray,
#     y_raw: np.ndarray,
#     values_raw: np.ndarray,
#     Xq: np.ndarray,
#     Yq: np.ndarray,
#     method: str = "linear",
# ):
#     points = np.column_stack([x_raw, y_raw])
#     query = np.column_stack([Xq.ravel(), Yq.ravel()])

#     # Main interpolation: linear on triangulation
#     linear_interp = LinearNDInterpolator(points, values_raw, fill_value=np.nan)
#     values_on_grid = linear_interp(query)

#     # values_on_grid = griddata(points, values_raw, query, method=method)
#     if np.any(np.isnan(values_on_grid)):
#         values_nearest = griddata(points, values_raw, query, method="nearest")
#         nan_mask = np.isnan(values_on_grid)
#         values_on_grid[nan_mask] = values_nearest[nan_mask]

#     return values_on_grid.reshape(Xq.shape)


def interpolate_raw_field_to_grid(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    values_raw: np.ndarray,
    Xq: np.ndarray,
    Yq: np.ndarray,
    method: str = "linear",
    fallback_to_nearest: bool = True,
) -> np.ndarray:
    points = np.column_stack([x_raw, y_raw])
    query = np.column_stack([Xq.ravel(), Yq.ravel()])

    values_on_grid = griddata(points, values_raw, query, method=method)

    if fallback_to_nearest and np.any(np.isnan(values_on_grid)):
        values_nearest = griddata(points, values_raw, query, method="nearest")
        mask = np.isnan(values_on_grid)
        values_on_grid[mask] = values_nearest[mask]

    return values_on_grid.reshape(Xq.shape)

def plot_field_tri(ax, Xq: np.ndarray, Yq: np.ndarray, field: np.ndarray, title: str):
    tcf = ax.tricontourf(Xq.ravel(), Yq.ravel(), field.ravel(), levels=50, cmap="coolwarm")
    ax.set_xlim(float(np.min(Xq)), float(np.max(Xq)))
    ax.set_ylim(float(np.min(Yq)), float(np.max(Yq)))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(tcf, ax=ax)


def visualize_prediction_2d(
    pred_coeffs: np.ndarray,
    true_coeffs_projected: np.ndarray,
    true_coeffs_pre_pca: np.ndarray | None,
    output_basis_type: str,
    input_basis_type: str,
    N_quad: int = 50,
    title_prefix: str = "",
    input_coeffs: np.ndarray | None = None,
    raw_input_x: np.ndarray | None = None,
    raw_input_y: np.ndarray | None = None,
    raw_input_values: np.ndarray | None = None,
    raw_solution_x: np.ndarray | None = None,
    raw_solution_y: np.ndarray | None = None,
    raw_solution_values: np.ndarray | None = None,
    domain: list[list[float]] | None = [[0.0, 1.0], [0.0, 1.0]],
    N_quad_for_orth: int = 200,
):
    pred_eval = evaluate_series_to_plot_points(
        coeffs=pred_coeffs,
        basis_type=output_basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )

    true_proj_eval = evaluate_series_to_plot_points(
        coeffs=true_coeffs_projected,
        basis_type=output_basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )

    true_pre_pca_eval = None
    if true_coeffs_pre_pca is not None:
        true_pre_pca_eval = evaluate_series_to_plot_points(
            coeffs=true_coeffs_pre_pca,
            basis_type=output_basis_type,
            domain=domain,
            N_quad=N_quad,
            N_quad_for_orth=N_quad_for_orth,
        )

    # input_eval = None
    # # if input_coeffs is not None:
    # #     input_eval = evaluate_series_to_plot_points(
    # #         coeffs=input_coeffs,
    # #         basis_type=input_basis_type,
    # #         domain=domain,
    # #         N_quad=N_quad,
    # #         N_quad_for_orth=N_quad_for_orth,
    # #     )
    # input_meta = None
    # if (
    #     input_coeffs is not None
    #     and raw_input_x is not None
    #     and raw_input_y is not None
    #     and raw_input_values is not None
    # ):
    #     _, input_meta = encode_single_field_to_coeffs(
    #         x=raw_input_x,
    #         y=raw_input_y,
    #         values=raw_input_values,
    #         N_ex=input_coeffs.shape[0],
    #         basis_type=input_basis_type,
    #         N_quad=N_quad,
    #     )

    #     input_eval = {
    #         "field": input_meta["f_rec"],
    #         "Xq": input_meta["Xq"],
    #         "Yq": input_meta["Yq"],
    #     }
    input_eval = None
    raw_input_on_grid = None
    reencoded_input_eval = None

    if input_coeffs is not None:
        input_eval = evaluate_series_to_plot_points(
            coeffs=input_coeffs,
            basis_type=input_basis_type,
            domain=domain,
            N_quad=N_quad,
            N_quad_for_orth=N_quad_for_orth,
        )

    if (
        raw_input_x is not None
        and raw_input_y is not None
        and raw_input_values is not None
    ):
        if input_eval is not None:
            raw_input_on_grid = interpolate_raw_field_to_grid(
                x_raw=raw_input_x,
                y_raw=raw_input_y,
                values_raw=raw_input_values,
                Xq=input_eval["Xq"],
                Yq=input_eval["Yq"],
                method="linear",
            )

        # optional: separately check consistency with online re-encoding
        if input_coeffs is not None:
            _, input_meta = encode_single_field_to_coeffs(
                x=raw_input_x,
                y=raw_input_y,
                values=raw_input_values,
                N_ex=input_coeffs.shape[0],
                basis_type=input_basis_type,
                N_quad=N_quad,
            )
            reencoded_input_eval = {
                "field": input_meta["f_rec"],
                "Xq": input_meta["Xq"],
                "Yq": input_meta["Yq"],
                "raw_on_encoding_grid": input_meta["f_quad"],
            }

    decoded_from_coeffs = evaluate_series_to_plot_points(
        coeffs=input_coeffs,
        basis_type=input_basis_type,
        domain=domain,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )
    print("input coeff decode consistency:",
        rel_l2(raw_input_on_grid, input_eval["field"]))
    
    raw_solution_on_grid = None
    pred_minus_raw = None
    raw_minus_projected = None
    if (
        raw_solution_x is not None
        and raw_solution_y is not None
        and raw_solution_values is not None
    ):
        raw_solution_on_grid = interpolate_raw_field_to_grid(
            x_raw=raw_solution_x,
            y_raw=raw_solution_y,
            values_raw=raw_solution_values,
            Xq=pred_eval["Xq"],
            Yq=pred_eval["Yq"],
            method="linear",
        )
        pred_minus_raw = pred_eval["field"] - raw_solution_on_grid
        raw_minus_projected = raw_solution_on_grid - true_proj_eval["field"]
    pred_minus_projected = pred_eval["field"] - true_proj_eval["field"]

    plt.figure(figsize=(18, 10))
    panel = 1

    # if input_eval is not None:
    #     ax = plt.subplot(2, 4, panel)
    #     plot_field_tri(
    #         ax,
    #         input_eval["Xq"],
    #         input_eval["Yq"],
    #         input_eval["field"],
    #         f"{title_prefix} encoded input",
    #     )
    #     panel += 1
    # raw_input_on_grid = None
    # input_minus_raw = None
    # if (
    #     raw_input_x is not None
    #     and raw_input_y is not None
    #     and raw_input_values is not None
    # ):
    #     ax = plt.subplot(2, 4, panel)
    #     tcf = ax.tricontourf(raw_input_x, raw_input_y, raw_input_values, levels=50, cmap="coolwarm")
    #     ax.set_aspect("equal")
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_title(f"{title_prefix} raw input")
    #     plt.colorbar(tcf, ax=ax)
    #     panel += 1

    #     if input_eval is not None:
    #         raw_input_on_grid = interpolate_raw_field_to_grid(
    #             x_raw=raw_input_x,
    #             y_raw=raw_input_y,
    #             values_raw=raw_input_values,
    #             Xq=input_eval["Xq"],
    #             Yq=input_eval["Yq"],
    #             method="linear", # this automatically creates larger error on high gradients
    #         )
    #         input_minus_raw = input_eval["field"] - raw_input_on_grid

    #         if panel <= 8:
    #             ax = plt.subplot(2, 4, panel)
    #             plot_field_tri(
    #                 ax,
    #                 input_eval["Xq"],
    #                 input_eval["Yq"],
    #                 input_minus_raw,
    #                 f"{title_prefix} |encoded input - raw input|",
    #             )
    #             panel += 1
    if input_meta is not None:
    
        ax = plt.subplot(2, 4, panel)
        plot_field_tri(
            ax,
            input_meta["Xq"],
            input_meta["Yq"],
            input_meta["f_rec"],
            f"{title_prefix} encoded input",
        )
        panel += 1

        ax = plt.subplot(2, 4, panel)
        plot_field_tri(
            ax,
            input_meta["Xq"],
            input_meta["Yq"],
            input_meta["f_quad"],
            f"{title_prefix} raw input",
        )
        panel += 1


        # if panel <= 8:
        #     ax = plt.subplot(2, 4, panel)
        #     plot_field_tri(
        #         ax,
        #         input_meta["Xq"],
        #         input_meta["Yq"],
        #         np.abs(input_minus_raw),
        #         f"{title_prefix} |encoded - raw input|",
        #     )
        #     panel += 1
        if input_eval is not None and raw_input_on_grid is not None and panel <= 8:
            ax = plt.subplot(2, 4, panel)
            plot_field_tri(ax, input_eval["Xq"], input_eval["Yq"], np.abs(input_eval["field"] - raw_input_on_grid), f"{title_prefix} |encoded - raw|")
            panel += 1

# # ----- test: if the representation on input is precise
#         x_raw_interp = interpolate_raw_field_to_grid(raw_input_x, raw_input_y, raw_input_values, input_eval["Xq"],
#             input_eval["Yq"])
#         x_diff = x_raw_interp - input_eval["field"]
#         plot_field_tri(
#             ax,
#             input_eval["Xq"],
#             input_eval["Yq"],
#             x_diff,
#             f"{title_prefix} encoded input diff",
#         )
#         panel += 1
# # ------ test passed
    if (
        raw_solution_x is not None
        and raw_solution_y is not None
        and raw_solution_values is not None
    ):
        ax = plt.subplot(2, 4, panel)
        tcf = ax.tricontourf(raw_solution_x, raw_solution_y, raw_solution_values, levels=50, cmap="coolwarm")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix} raw solution")
        plt.colorbar(tcf, ax=ax)
        panel += 1

    ax = plt.subplot(2, 4, panel)
    plot_field_tri(
        ax,
        true_proj_eval["Xq"],
        true_proj_eval["Yq"],
        true_proj_eval["field"],
        f"{title_prefix} target (projected)",
    )
    panel += 1

    ax = plt.subplot(2, 4, panel)
    plot_field_tri(
        ax,
        pred_eval["Xq"],
        pred_eval["Yq"],
        pred_eval["field"],
        f"{title_prefix} prediction",
    )
    panel += 1

    ax = plt.subplot(2, 4, panel)
    plot_field_tri(
        ax,
        pred_eval["Xq"],
        pred_eval["Yq"],
        np.abs(pred_minus_projected),
        f"{title_prefix} |pred - projected|",
    )
    panel += 1

    if pred_minus_raw is not None:
        ax = plt.subplot(2, 4, panel)
        plot_field_tri(
            ax,
            pred_eval["Xq"],
            pred_eval["Yq"],
            np.abs(pred_minus_raw),
            f"{title_prefix} |pred - raw solution|",
        )
        panel += 1

    if raw_minus_projected is not None and panel <= 8:
        ax = plt.subplot(2, 4, panel)
        plot_field_tri(
            ax,
            pred_eval["Xq"],
            pred_eval["Yq"],
            np.abs(raw_minus_projected),
            f"{title_prefix} raw_minus_projected",
        )
        panel += 1

    plt.tight_layout()
    plt.show()

    err_proj = rel_l2(pred_eval["field"], true_proj_eval["field"])
    print("function-space rel L2 vs projected target:", err_proj)

    if true_pre_pca_eval is not None:
        err_pre_pca = rel_l2(pred_eval["field"], true_pre_pca_eval["field"])
        print("function-space rel L2 vs pre-PCA target:", err_pre_pca)

    if raw_solution_on_grid is not None:
        err_raw = rel_l2(pred_eval["field"], raw_solution_on_grid)
        print("function-space rel L2 vs raw solution (after interpolation to decoder grid):", err_raw)

    # if raw_input_on_grid is not None and input_eval is not None:
    #     err_input_raw = rel_l2(input_eval["field"], raw_input_on_grid)
    #     print("input representation rel L2 vs raw input (after interpolation to decoder grid):", err_input_raw)

    if input_meta is not None:
        err_input_raw = rel_l2(input_meta["f_rec"], input_meta["f_quad"])
        print("input representation rel L2 on encoding grid:", err_input_raw)
# NOTE we could visualize the full error here too. lets do that in frame 7

# ============================================================
# 5) Pure analysis helpers
# ============================================================

def build_output_only_dataset_for_rank(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
    N_ex_output: int,
    basis_u: str,
    N_quad: int = 50,
) -> np.ndarray:
    """
    Returns only Y tensor of shape (n, N_ex_output, N_ex_output).
    We still need to call the common dataset builder, but input side is irrelevant here.
    """
    _, Y, _ = build_dataset_2d_operator(
        data_dir=data_dir,
        equation=equation,
        indices=indices,
        N_ex_input=1,
        N_ex_output=N_ex_output,
        basis_a=basis_u,
        basis_u=basis_u,
        N_quad=N_quad,
    )
    return Y


def analyze_output_basis_rank_curve(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
    basis_u: str,
    output_ranks: Sequence[int],
    N_quad: int = 50,
):
    """
    Study only the output basis rank N_ex_output.

    Reference object:
        reconstruction from the largest tested output rank.

    This does NOT compare against the exact raw FEM field directly.
    It compares smaller output-rank reconstructions to the largest-rank
    basis reconstruction in the tested family.
    """
    output_ranks = sorted(set(int(r) for r in output_ranks))
    if len(output_ranks) == 0:
        raise ValueError("output_ranks must not be empty")

    ref_rank = max(output_ranks)
    Y_ref = build_output_only_dataset_for_rank(
        data_dir=data_dir,
        equation=equation,
        indices=indices,
        N_ex_output=ref_rank,
        basis_u=basis_u,
        N_quad=N_quad,
    )

    f_ref_list = [
        evaluate_series_L2(Y_ref[i], N_quad=N_quad, basis_type=basis_u)
        for i in range(len(Y_ref))
    ]

    results = []
    for r in output_ranks:
        Y_r = build_output_only_dataset_for_rank(
            data_dir=data_dir,
            equation=equation,
            indices=indices,
            N_ex_output=r,
            basis_u=basis_u,
            N_quad=N_quad,
        )

        coeff_errs = []
        func_errs = []
        for i in range(len(Y_r)):
            f_r = evaluate_series_L2(Y_r[i], N_quad=N_quad, basis_type=basis_u)
            f_ref = f_ref_list[i]

            func_errs.append(rel_l2(f_r, f_ref))

            Y_r_pad = np.zeros_like(Y_ref[i])
            Y_r_pad[:r, :r] = Y_r[i]
            coeff_errs.append(rel_l2(Y_r_pad, Y_ref[i]))

        results.append({
            "output_rank": int(r),
            "coeff_rel_l2_mean_vs_ref": float(np.mean(coeff_errs)),
            "coeff_rel_l2_median_vs_ref": float(np.median(coeff_errs)),
            "function_rel_l2_mean_vs_ref": float(np.mean(func_errs)),
            "function_rel_l2_median_vs_ref": float(np.median(func_errs)),
        })

    return results


def analyze_output_pca_curve(
    Ytr_mat: np.ndarray,
    Yte_mat: np.ndarray,
    basis_u: str,
    N_quad: int = 50,
    pca_ranks: Sequence[int] = (4, 8, 16, 32, 64),
    eval_count: int = 50,
):
    """
    Study only PCA compression of the already basis-encoded outputs.

    Input:
        Ytr_mat, Yte_mat are basis-encoded output tensors of shape
        (n, N_ex_output, N_ex_output)

    Output:
        reconstruction quality as PCA rank varies
    """
    Ytr = flatten_coeff_tensor(Ytr_mat)
    Yte = flatten_coeff_tensor(Yte_mat)

    max_rank = min(Ytr.shape[0], Ytr.shape[1])
    valid_ranks = [int(r) for r in pca_ranks if 1 <= int(r) <= max_rank]

    results = []
    for r in valid_ranks:
        pca = PCA(n_components=r)
        pca.fit(Ytr)

        Yte_red = pca.transform(Yte)
        Yte_rec = pca.inverse_transform(Yte_red)

        Yte_rec_mat = reconstruct_coeff_tensor(Yte_rec)
        Yte_true_mat = reconstruct_coeff_tensor(Yte)

        coeff_rel = rel_l2(Yte_rec, Yte)
        func_mean, func_med = evaluate_test_set_function_error_2d(
            pred_tensor=Yte_rec_mat,
            true_tensor=Yte_true_mat,
            basis_type=basis_u,
            N_quad=N_quad,
            eval_count=eval_count,
        )

        results.append({
            "pca_rank": int(r),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "coeff_rel_l2": float(coeff_rel),
            "function_rel_l2_mean": float(func_mean),
            "function_rel_l2_median": float(func_med),
        })

    return results

''' these are only sanity checks. '''
def build_input_only_dataset_for_rank(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
    N_ex_input: int,
    basis_a: str,
    N_quad: int = 50,
) -> np.ndarray:
    """
    Returns only X tensor of shape (n, N_ex_input, N_ex_input).

    This is the input-side analogue of build_output_only_dataset_for_rank.
    """
    X, _, _ = build_dataset_2d_operator(
        data_dir=data_dir,
        equation=equation,
        indices=indices,
        N_ex_input=N_ex_input,
        N_ex_output=1,
        basis_a=basis_a,
        basis_u=basis_a,
        N_quad=N_quad,
    )
    return X


def analyze_input_basis_rank_curve(
    data_dir: str,
    equation: str,
    indices: np.ndarray,
    basis_a: str,
    input_ranks: Sequence[int],
    N_quad: int = 50,
    N_quad_for_orth: int = 200,
    domain: list[list[float]] | None = [[0.0, 1.0], [0.0, 1.0]],
):
    """
    Study only the input basis rank N_ex_input.

    Reference object:
        the raw saved input field itself (after interpolation onto the decoder grid).

    This is different from the output-rank analysis: here we explicitly compare
    the encoded-decoded input representation to the raw saved field, because
    this is the actual fidelity question on the learner's input side.
    """
    input_ranks = sorted(set(int(r) for r in input_ranks))
    if len(input_ranks) == 0:
        raise ValueError("input_ranks must not be empty")

    raw_inputs, _ = load_raw_test_fields(
        data_dir=data_dir,
        equation=equation,
        indices=indices,
    )

    results = []
    for r in input_ranks:
        X_r = build_input_only_dataset_for_rank(
            data_dir=data_dir,
            equation=equation,
            indices=indices,
            N_ex_input=r,
            basis_a=basis_a,
            N_quad=N_quad,
        )

        coeff_norms = []
        func_errs = []

        for i in range(len(X_r)):
            x_eval = evaluate_series_to_plot_points(
                coeffs=X_r[i],
                basis_type=basis_a,
                domain=domain,
                N_quad=N_quad,
                N_quad_for_orth=N_quad_for_orth,
            )

            raw_input = raw_inputs[i]
            raw_on_grid = interpolate_raw_field_to_grid(
                x_raw=raw_input["x"],
                y_raw=raw_input["y"],
                values_raw=raw_input["values"],
                Xq=x_eval["Xq"],
                Yq=x_eval["Yq"],
                method="linear",
            )

            func_errs.append(rel_l2(x_eval["field"], raw_on_grid))
            coeff_norms.append(float(np.linalg.norm(X_r[i])))

        results.append({
            "input_rank": int(r),
            "input_coeff_norm_mean": float(np.mean(coeff_norms)),
            "input_coeff_norm_median": float(np.median(coeff_norms)),
            "function_rel_l2_mean_vs_raw": float(np.mean(func_errs)),
            "function_rel_l2_median_vs_raw": float(np.median(func_errs)),
        })

    return results


def plot_input_basis_rank_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean_vs_raw",),
    title: str = "Input basis rank analysis",
):
    ranks = [int(row["input_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    plt.xlabel("input rank")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_input_rank_analysis(
    *,
    data_dir: str,
    equation: str,
    test_indices: np.ndarray,
    basis_a: str,
    input_ranks: Sequence[int],
    N_quad: int = 50,
    N_quad_for_orth: int = 200,
):
    results = analyze_input_basis_rank_curve(
        data_dir=data_dir,
        equation=equation,
        indices=test_indices,
        basis_a=basis_a,
        input_ranks=input_ranks,
        N_quad=N_quad,
        N_quad_for_orth=N_quad_for_orth,
    )
    print_rank_curve_table(results, "Input basis rank analysis")
    return results

# ============================================================
# 6) Full 2D pipeline runner
# ============================================================

def run_training_pipeline_2d(
    *,
    data_dir: str,
    cache_dir: str,
    equation: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    N_ex_input: int,
    N_ex_output: int,
    basis_a: str,
    basis_u: str,
    N_quad: int = 50,
    model_type: str = "rf",
    use_pca: bool = True,
    pca_components: int = 32,
    normalize_in: float = 1.0,
    normalize_out: float = 1.0,
    device: str = "cpu",
    force_rebuild: bool = False,
    shared_kernel: bool = False,
    d_features: int = 2048,
    lengthscale: float = 1.0,
    lam: float = 1e-4,
    sigma_mode: str = "uniform",
    sigma_decay: float = 1.0,
    mlp_hidden: int = 256,
    mlp_depth: int = 5,
    mlp_epochs: int = 1000,
    mlp_lr: float = 1e-4,
):
    ensure_dir(cache_dir)

    tag = (
        f"operator2d_eq{equation}"
        f"_Nin{N_ex_input}_Nout{N_ex_output}"
        f"_ba{basis_a}_bu{basis_u}"
        f"_pca{use_pca}_{pca_components}"
        f"_shared{shared_kernel}"
    )
    train_cache = os.path.join(cache_dir, f"{tag}_train.npz")
    test_cache = os.path.join(cache_dir, f"{tag}_test.npz")

    if os.path.exists(train_cache) and not force_rebuild:
        d = load_npz(train_cache)
        Xtr_mat, Ytr_mat, meta = d["X"], d["Y"], d["meta"].item()
    else:
        Xtr_mat, Ytr_mat, meta = build_dataset_2d_operator(
            data_dir=data_dir,
            equation=equation,
            indices=train_indices,
            N_ex_input=N_ex_input,
            N_ex_output=N_ex_output,
            basis_a=basis_a,
            basis_u=basis_u,
            N_quad=N_quad,
        )
        save_npz(train_cache, X=Xtr_mat, Y=Ytr_mat, meta=meta)

    if os.path.exists(test_cache) and not force_rebuild:
        d = load_npz(test_cache)
        Xte_mat, Yte_mat = d["X"], d["Y"]
    else:
        Xte_mat, Yte_mat, _ = build_dataset_2d_operator(
            data_dir=data_dir,
            equation=equation,
            indices=test_indices,
            N_ex_input=N_ex_input,
            N_ex_output=N_ex_output,
            basis_a=basis_a,
            basis_u=basis_u,
            N_quad=N_quad,
        )
        save_npz(test_cache, X=Xte_mat, Y=Yte_mat, meta=meta)

    Yte_mat_pre_pca = Yte_mat.copy()

    raw_test_input_fields, raw_test_solution_fields = load_raw_test_fields(
        data_dir=data_dir,
        equation=equation,
        indices=test_indices,
    )

    Xtr = flatten_coeff_tensor(Xtr_mat)
    Ytr = flatten_coeff_tensor(Ytr_mat)
    Xte = flatten_coeff_tensor(Xte_mat)
    Yte = flatten_coeff_tensor(Yte_mat)

    print(
        "shape check:",
        "Xtr", Xtr.shape, "Ytr", Ytr.shape,
        "Xte", Xte.shape, "Yte", Yte.shape
    )

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

    Xtr_t = torch.tensor(Xtr_p, dtype=torch.float32, device=device) * float(normalize_in)
    Ytr_t = torch.tensor(Ytr_p, dtype=torch.float32, device=device) * float(normalize_out)
    Xte_t = torch.tensor(Xte_p, dtype=torch.float32, device=device) * float(normalize_in)
    Yte_t = torch.tensor(Yte_p, dtype=torch.float32, device=device) * float(normalize_out)

    trained = {"model_type": model_type, "shared_kernel": shared_kernel}

    if model_type == "nn":
        model = MLP(
            d_in=Xtr_t.shape[1],
            d_out=Ytr_t.shape[1],
            hidden=mlp_hidden,
            depth=mlp_depth,
            dropout=0.0,
        ).to(device)
        model = train_mlp(
            model, Xtr_t, Ytr_t, Xte_t, Yte_t,
            lr=mlp_lr, epochs=mlp_epochs, print_every=100
        )
        with torch.no_grad():
            Yhat_t = model(Xte_t)
        trained["model"] = model

    elif model_type == "rf":
        d_in_eff = Xtr_t.shape[1]
        d_out_eff = Ytr_t.shape[1]

        if sigma_mode == "decay":
            sigma_in = sigma_generate(
                d_in_eff, PCA=use_pca, decay_type="exponential", decay_rate=sigma_decay
            )
            sigma_out = sigma_generate(
                d_out_eff, PCA=use_pca, decay_type="exponential", decay_rate=sigma_decay
            )
        else:
            sigma_in = np.ones(d_in_eff)
            sigma_out = np.ones(d_out_eff)

        if shared_kernel:
            feat_model = RFF(
                d_in=d_in_eff,
                d_features=d_features,
                lengthscale=lengthscale,
                trainable_kernel=False,
                device=device,
            ).to(device)
            feat_model.resample()

            with torch.no_grad():
                Phi_tr = feat_model(Xtr_t)
                Phi_te = feat_model(Xte_t)

            W = ridge_fit_shared_regime_aware(Phi_tr, Ytr_t, lam=lam)
            Yhat_t = ridge_predict_shared(Phi_te, W)
            trained["W"] = W.detach().cpu()

        else:
            feat_model = NN_cff_vec(
                d_in=d_in_eff,
                d_features=d_features,
                d_out=d_out_eff,
                lengthscale=lengthscale,
                sigma_in=sigma_in,
                sigma_out=sigma_out,
                trainable_kernel=False,
                projection_type="Cauchy",
                device=device,
            ).to(device)
            feat_model.resample()

            with torch.no_grad():
                Phi_tr = feat_model(Xtr_t)
                Phi_te = feat_model(Xte_t)

            # W = ridge_fit_multioutput_features(Phi_tr, Ytr_t, lam=lam)
            W = ridge_fit_multioutput_features_regime_aware(Phi_tr, Ytr_t, lam=lam)
            Yhat_t = predict_multioutput_features(Phi_te, W)
            trained["W"] = W.detach().cpu()

        trained["feature_model"] = feat_model

    else:
        raise ValueError("model_type must be 'rf' or 'nn'")

    Yhat = Yhat_t.detach().cpu().numpy() / float(normalize_out)
    Ytrue = Yte_t.detach().cpu().numpy() / float(normalize_out)

    if use_pca:
        Yhat = pca.inverse_Y(Yhat)
        Ytrue = pca.inverse_Y(Ytrue)

    vec_mse = float(np.mean((Yhat - Ytrue) ** 2))
    print("vector-space test MSE:", vec_mse)

    Yhat_mat = reconstruct_coeff_tensor(Yhat)
    Ytrue_mat = reconstruct_coeff_tensor(Ytrue)

    coeff_err = coefficient_errors(Yhat, Ytrue)
    print("coefficient-space rel L2:", coeff_err["rel_l2"])

    mean_fe, med_fe = evaluate_test_set_function_error_2d(
        pred_tensor=Yhat_mat,
        true_tensor=Ytrue_mat,
        basis_type=basis_u,
        N_quad=N_quad,
        eval_count=min(50, len(Yhat_mat)),
    )
    print("function-space rel L2 mean/median over test subset:", mean_fe, med_fe)

    return {
        "vec_mse": vec_mse,
        "coeff_rel_l2": coeff_err["rel_l2"],
        "function_rel_l2_mean": mean_fe,
        "function_rel_l2_median": med_fe,
        "Xte": Xte,
        "Xte_mat": Xte_mat,
        "Yhat": Yhat,
        "Ytrue": Ytrue,
        "Yhat_mat": Yhat_mat,
        "Ytrue_mat": Ytrue_mat,
        "Ytrue_mat_pre_pca": Yte_mat_pre_pca,
        "raw_test_input_fields": raw_test_input_fields,
        "raw_test_solution_fields": raw_test_solution_fields,
        "meta": meta,
        "pca": pca,
        "trained": trained,
    }


# ============================================================
# 7) Example sweep helpers
# ============================================================

def print_rank_curve_table(results: list[dict], title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for row in results:
        print(row)


def run_output_rank_analysis(
    *,
    data_dir: str,
    equation: str,
    test_indices: np.ndarray,
    basis_u: str,
    output_ranks: Sequence[int],
    N_quad: int = 50,
):
    results = analyze_output_basis_rank_curve(
        data_dir=data_dir,
        equation=equation,
        indices=test_indices,
        basis_u=basis_u,
        output_ranks=output_ranks,
        N_quad=N_quad,
    )
    print_rank_curve_table(results, "Output basis rank analysis")
    return results


def run_output_pca_analysis(
    *,
    data_dir: str,
    equation: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    basis_u: str,
    N_ex_output: int,
    N_quad: int = 50,
    pca_ranks: Sequence[int] = (4, 8, 16, 32, 64),
):
    Ytr_mat = build_output_only_dataset_for_rank(
        data_dir=data_dir,
        equation=equation,
        indices=train_indices,
        N_ex_output=N_ex_output,
        basis_u=basis_u,
        N_quad=N_quad,
    )
    Yte_mat = build_output_only_dataset_for_rank(
        data_dir=data_dir,
        equation=equation,
        indices=test_indices,
        N_ex_output=N_ex_output,
        basis_u=basis_u,
        N_quad=N_quad,
    )

    results = analyze_output_pca_curve(
        Ytr_mat=Ytr_mat,
        Yte_mat=Yte_mat,
        basis_u=basis_u,
        N_quad=N_quad,
        pca_ranks=pca_ranks,
        eval_count=min(50, len(Yte_mat)),
    )
    print_rank_curve_table(results, f"PCA analysis at fixed output rank N_ex_output={N_ex_output}")
    return results


def plot_output_basis_rank_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean_vs_ref",),
    title: str = "Output basis rank analysis",
):
    ranks = [int(row["output_rank"]) for row in results]

    plt.figure(figsize=(7, 5))
    for metric_key in metric_keys:
        values = [float(row[metric_key]) for row in results]
        plt.plot(ranks, values, marker="o", label=metric_key)

    plt.xlabel("output rank")
    plt.ylabel("error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_output_pca_rank_curve(
    results: list[dict],
    metric_keys: Sequence[str] = ("function_rel_l2_mean", "coeff_rel_l2"),
    show_explained_variance: bool = True,
    title: str = "Output PCA rank analysis",
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


def report_dimension_selection(auto_dim: dict | None, plot: bool = True):
    if not auto_dim:
        print("\nAutomatic dimension selection: skipped.")
        return

    required_keys = {"selected_output_rank", "selected_pca_rank", "basis_selection", "pca_selection"}
    if not required_keys.issubset(set(auto_dim.keys())):
        print("\nAutomatic dimension selection: incomplete data, reporting skipped.")
        return

    print("\nAutomatic dimension selection:")
    print("selected N_ex_output:", auto_dim["selected_output_rank"])
    print("selected PCA rank:", auto_dim["selected_pca_rank"])
    print("basis selection status:", auto_dim["basis_selection"]["selection_status"])
    print("pca selection status:", auto_dim["pca_selection"]["selection_status"])

    print_rank_curve_table(auto_dim["basis_selection"]["results"], "Automatic output-rank selection results")
    print_rank_curve_table(auto_dim["pca_selection"]["results"], "Automatic PCA-rank selection results")

    if plot:
        plot_output_basis_rank_curve(auto_dim["basis_selection"]["results"])
        plot_output_pca_rank_curve(auto_dim["pca_selection"]["results"])


# ============================================================
# 8) Example main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run 2D operator model with config file.")
    parser.add_argument("--config", type=str, required=True, help="Config suffix, e.g. darcy -> cfgs.cfg_darcy")
    args = parser.parse_args()

    config_module = importlib.import_module(f"cfgs.cfg_{args.config}")
    cfg = config_module.cfg

    print("Loaded config:")
    print(cfg)
    print("Learning rate:", getattr(cfg, "LR", None))
    print("Model:", cfg.train)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    equation = cfg.EQ
    force_recompute = cfg.FORCE_RECOMPUTE

    if hasattr(cfg, "N_ex_input") or hasattr(cfg, "N_ex_output"):
        N_ex_input = getattr(cfg, "N_ex_input")
        N_ex_output = getattr(cfg, "N_ex_output")
    else:
        N_ex_input = cfg.N_ex
        N_ex_output = cfg.N_ex

    train_sample_size = getattr(cfg, "train_sample_size", 900)
    test_sample_size = getattr(cfg, "test_sample_size", 100)

    here = Path(__file__).resolve().parent
    data_dir = str(here / "data")
    cache_dir = str(here / "data" / "cache" / equation)

    train_indices = np.arange(0, train_sample_size)
    test_indices = np.arange(train_sample_size, train_sample_size + test_sample_size)

    normalize_const_in = getattr(cfg, "normalize_const_in", 1.0)
    normalize_const_out = getattr(cfg, "normalize_const_out", 1.0)

    auto_dim = None

    if getattr(cfg, "AUTO_DIMENSION_SELECTION", True):
        auto_dim = auto_select_dimensions_for_training(
            data_dir=data_dir,
            equation=equation,
            basis_u=cfg.basis_u,
            train_indices_full=train_indices,
            test_indices_full=test_indices,
            analysis_sample_size_train=getattr(cfg, "ANALYSIS_SAMPLE_SIZE_TRAIN", 300),
            analysis_sample_size_test=getattr(cfg, "ANALYSIS_SAMPLE_SIZE_TEST", 100),
            output_rank_candidates=getattr(cfg, "OUTPUT_RANK_CANDIDATES", [4, 8, 12, 16, 20, 24, 32]),
            reference_output_rank=getattr(cfg, "REFERENCE_OUTPUT_RANK", 32),
            output_basis_tol=getattr(cfg, "OUTPUT_BASIS_TOL", 5e-2),
            output_basis_metric=getattr(cfg, "OUTPUT_BASIS_ERROR_METRIC", "function_rel_l2_mean_vs_ref"),
            pca_rank_candidates=getattr(cfg, "PCA_RANK_CANDIDATES", [4, 8, 16, 32, 64]),
            output_pca_tol=getattr(cfg, "OUTPUT_PCA_TOL", 1e-2),
            output_pca_metric=getattr(cfg, "OUTPUT_PCA_ERROR_METRIC", "function_rel_l2_mean"),
            N_quad=getattr(cfg, "N_quad", 50),
        )

        N_ex_output = int(auto_dim["selected_output_rank"])
        pca_components = int(auto_dim["selected_pca_rank"])

        print("\nAutomatic dimension selection:")
        print("selected N_ex_output:", N_ex_output)
        print("selected PCA rank:", pca_components)
        print("basis selection status:", auto_dim["basis_selection"]["selection_status"])
        print("pca selection status:", auto_dim["pca_selection"]["selection_status"])
    else:
        pca_components = cfg.PCA_COMPONENTS
        N_ex_output = cfg.N_ex_output

    report_dimension_selection(
        auto_dim,
        plot=getattr(cfg, "PLOT_DIMENSION_SELECTION", True),
    )

    if getattr(cfg, "RUN_INPUT_RANK_ANALYSIS", False):
        input_rank_results = run_input_rank_analysis(
            data_dir=data_dir,
            equation=equation,
            test_indices=test_indices[:getattr(cfg, "ANALYSIS_SAMPLE_SIZE_TEST", 100)],
            basis_a=cfg.basis_a,
            input_ranks=getattr(cfg, "INPUT_RANK_CANDIDATES", [4, 8, 12, 16, 20, 24, 32]),
            N_quad=getattr(cfg, "N_quad", 50),
            N_quad_for_orth=getattr(cfg, "N_quad_for_orth", 200),
        )
        if getattr(cfg, "PLOT_INPUT_RANK_ANALYSIS", True):
            plot_input_basis_rank_curve(
                input_rank_results,
                metric_keys=("function_rel_l2_mean_vs_raw", "function_rel_l2_median_vs_raw"),
                title="Input basis rank analysis",
            )

    out = run_training_pipeline_2d(
        data_dir=data_dir,
        cache_dir=cache_dir,
        equation=equation,
        train_indices=train_indices,
        test_indices=test_indices,
        N_ex_input=N_ex_input,
        N_ex_output=N_ex_output,
        basis_a=cfg.basis_a,
        basis_u=cfg.basis_u,
        N_quad=getattr(cfg, "N_quad", 50),
        model_type=cfg.train,
        use_pca=cfg.USE_PCA,
        pca_components=pca_components,
        normalize_in=normalize_const_in,
        normalize_out=normalize_const_out,
        device=getattr(cfg, "device", "cpu"),
        force_rebuild=force_recompute,
        shared_kernel=getattr(cfg, "shared_kernel", False),
        d_features=getattr(cfg, "d_features", 2048),
        lengthscale=cfg.lengthscale,
        lam=getattr(cfg, "LAMBDA_REG", 1e-4),
        sigma_mode=getattr(cfg, "sigma_mode", "uniform"),
        sigma_decay=getattr(cfg, "scaling", 1.0),
        mlp_hidden=getattr(cfg, "mlp_hidden", 256),
        mlp_depth=getattr(cfg, "mlp_depth", 5),
        mlp_epochs=getattr(cfg, "EPOCHS", 1000),
        mlp_lr=getattr(cfg, "LR", 1e-4),
    )

    print("Pipeline results:")
    print("vec_mse:", out["vec_mse"])
    print("coeff_rel_l2:", out["coeff_rel_l2"])
    print("function_rel_l2_mean:", out["function_rel_l2_mean"])
    print("function_rel_l2_median:", out["function_rel_l2_median"])

    compare_pre_pca = getattr(cfg, "compare_pre_pca", False)
    n_show = min(getattr(cfg, "sample_num", 5), len(out["Yhat_mat"]))

    if getattr(cfg, "VISUALIZE_PREDICTIONS", True):
        for i in range(n_show):
            raw_input = out["raw_test_input_fields"][i]
            raw_solution = out["raw_test_solution_fields"][i]

            print(f"Sample {i}:")
            visualize_prediction_2d(
                pred_coeffs=out["Yhat_mat"][i],
                true_coeffs_projected=out["Ytrue_mat"][i],
                true_coeffs_pre_pca=out["Ytrue_mat_pre_pca"][i],
                output_basis_type=cfg.basis_u,
                N_quad=getattr(cfg, "N_quad", 100),
                title_prefix=f"sample {i}",
                input_coeffs=out["Xte_mat"][i],
                input_basis_type=cfg.basis_a,
                raw_input_x=raw_input["x"],
                raw_input_y=raw_input["y"],
                raw_input_values=raw_input["values"],
                raw_solution_x=raw_solution["x"],
                raw_solution_y=raw_solution["y"],
                raw_solution_values=raw_solution["values"],
            )

    # run timed tests
    timing_encoded = time_surrogate_inference_encoded(
    out=out,
    normalize_const_in=normalize_const_in,
    normalize_const_out=normalize_const_out,
    device=getattr(cfg, "device", "cpu"),
    )
    print(timing_encoded)
    time_end=time_surrogate_end_to_end(
        out=out, 
        cfg=cfg,
    )
    print(time_end)



#### =============== tests ===============
def feature_size_test():
    parser = argparse.ArgumentParser(description="Run 2D operator model with config file.")
    parser.add_argument("--config", type=str, required=True, help="Config suffix, e.g. darcy -> cfgs.cfg_darcy")
    args = parser.parse_args()

    config_module = importlib.import_module(f"cfgs.cfg_{args.config}")
    cfg = config_module.cfg

    print("Loaded config:")
    print(cfg)
    print("Learning rate:", getattr(cfg, "LR", None))
    print("Model:", cfg.train)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    equation = cfg.EQ
    force_recompute = cfg.FORCE_RECOMPUTE

    if hasattr(cfg, "N_ex_input") or hasattr(cfg, "N_ex_output"):
        N_ex_input = getattr(cfg, "N_ex_input")
        N_ex_output = getattr(cfg, "N_ex_output")
    else:
        N_ex_input = cfg.N_ex
        N_ex_output = cfg.N_ex

    train_sample_size = getattr(cfg, "train_sample_size", 900)
    test_sample_size = getattr(cfg, "test_sample_size", 100)

    here = Path(__file__).resolve().parent
    data_dir = str(here / "data")
    cache_dir = str(here / "data" / "cache" / equation)

    train_indices = np.arange(0, train_sample_size)
    test_indices = np.arange(train_sample_size, train_sample_size + test_sample_size)

    normalize_const_in = getattr(cfg, "normalize_const_in", 1.0)
    normalize_const_out = getattr(cfg, "normalize_const_out", 1.0)

    auto_dim = None

    if getattr(cfg, "AUTO_DIMENSION_SELECTION", False):
        auto_dim = auto_select_dimensions_for_training(
            data_dir=data_dir,
            equation=equation,
            basis_u=cfg.basis_u,
            train_indices_full=train_indices,
            test_indices_full=test_indices,
            analysis_sample_size_train=getattr(cfg, "ANALYSIS_SAMPLE_SIZE_TRAIN", 300),
            analysis_sample_size_test=getattr(cfg, "ANALYSIS_SAMPLE_SIZE_TEST", 100),
            output_rank_candidates=getattr(cfg, "OUTPUT_RANK_CANDIDATES", [4, 8, 12, 16, 20, 24, 32]),
            reference_output_rank=getattr(cfg, "REFERENCE_OUTPUT_RANK", 32),
            output_basis_tol=getattr(cfg, "OUTPUT_BASIS_TOL", 5e-2),
            output_basis_metric=getattr(cfg, "OUTPUT_BASIS_ERROR_METRIC", "function_rel_l2_mean_vs_ref"),
            pca_rank_candidates=getattr(cfg, "PCA_RANK_CANDIDATES", [4, 8, 16, 32, 64]),
            output_pca_tol=getattr(cfg, "OUTPUT_PCA_TOL", 1e-2),
            output_pca_metric=getattr(cfg, "OUTPUT_PCA_ERROR_METRIC", "function_rel_l2_mean"),
            N_quad=getattr(cfg, "N_quad", 50),
        )

        N_ex_output = int(auto_dim["selected_output_rank"])
        pca_components = int(auto_dim["selected_pca_rank"])

        print("\nAutomatic dimension selection:")
        print("selected N_ex_output:", N_ex_output)
        print("selected PCA rank:", pca_components)
        print("basis selection status:", auto_dim["basis_selection"]["selection_status"])
        print("pca selection status:", auto_dim["pca_selection"]["selection_status"])
    else:
        pca_components = cfg.PCA_COMPONENTS

    report_dimension_selection(
        auto_dim,
        plot=getattr(cfg, "PLOT_DIMENSION_SELECTION", True),
    )
    for feature_dim in [128, 256, 512, 1024, 2048, 4096]:
        out = run_training_pipeline_2d(
            data_dir=data_dir,
            cache_dir=cache_dir,
            equation=equation,
            train_indices=train_indices,
            test_indices=test_indices,
            N_ex_input=N_ex_input,
            N_ex_output=N_ex_output,
            basis_a=cfg.basis_a,
            basis_u=cfg.basis_u,
            N_quad=getattr(cfg, "N_quad", 50),
            model_type=cfg.train,
            use_pca=cfg.USE_PCA,
            pca_components=pca_components,
            normalize_in=normalize_const_in,
            normalize_out=normalize_const_out,
            device=getattr(cfg, "device", "cpu"),
            force_rebuild=force_recompute,
            shared_kernel=getattr(cfg, "shared_kernel", False),
            d_features=feature_dim,
            lengthscale=cfg.lengthscale,
            lam=getattr(cfg, "LAMBDA_REG", 1e-4),
            sigma_mode=getattr(cfg, "sigma_mode", "uniform"),
            sigma_decay=getattr(cfg, "scaling", 1.0),
            mlp_hidden=getattr(cfg, "mlp_hidden", 256),
            mlp_depth=getattr(cfg, "mlp_depth", 5),
            mlp_epochs=getattr(cfg, "EPOCHS", 1000),
            mlp_lr=getattr(cfg, "LR", 1e-4),
        )

        print("Pipeline results:")
        print('feature dimension', feature_dim)
        print("vec_mse:", out["vec_mse"])
        print("coeff_rel_l2:", out["coeff_rel_l2"])
        print("function_rel_l2_mean:", out["function_rel_l2_mean"])
        print("function_rel_l2_median:", out["function_rel_l2_median"])


import time

def predict_with_trained_bundle(
    *,
    trained: dict,
    X_batch: np.ndarray,
    pca,
    normalize_in: float,
    normalize_out: float,
    device: str,
):
    """
    X_batch: encoded + flattened input vectors, shape (n, d_in)

    Returns:
        Yhat_mat: predicted output coefficient tensors, shape (n, N_ex_output, N_ex_output)
        elapsed_core: prediction time for model forward + ridge application
        elapsed_post: inverse PCA + tensor reconstruction time
    """
    model_type = trained["model_type"]
    shared_kernel = trained["shared_kernel"]

    X_t = torch.tensor(X_batch, dtype=torch.float32, device=device) * float(normalize_in)

    t0 = time.perf_counter()

    if model_type == "nn":
        model = trained["model"]
        model.eval()
        with torch.no_grad():
            Yhat_t = model(X_t)

    elif model_type == "rf":
        feat_model = trained["feature_model"]
        W = trained["W"].to(device)

        with torch.no_grad():
            if shared_kernel:
                Phi = feat_model(X_t)                 # (n, F)
                Yhat_t = ridge_predict_shared(Phi, W)
            else:
                Phi = feat_model(X_t)                 # (n, F, D)
                Yhat_t = predict_multioutput_features(Phi, W)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    elapsed_core = time.perf_counter() - t0

    t1 = time.perf_counter()

    Yhat = Yhat_t.detach().cpu().numpy() / float(normalize_out)

    if pca.use:
        Yhat = pca.inverse_Y(Yhat)

    Yhat_mat = reconstruct_coeff_tensor(Yhat)

    elapsed_post = time.perf_counter() - t1

    return Yhat_mat, float(elapsed_core), float(elapsed_post)


def time_surrogate_inference_encoded(
    *,
    out: dict,
    normalize_const_in: float = 1.0,
    normalize_const_out: float = 1.0,
    device: str = "cpu",
    repeats: int = 20,
    warmup: int = 3,
):
    """
    Time surrogate inference on already encoded test inputs.
    This is the cleanest online-query timing for the learned model itself.
    """
    Xte = out["Xte"]
    trained = out["trained"]
    pca = out["pca"]

    # Warmup
    for _ in range(warmup):
        _ = predict_with_trained_bundle(
            trained=trained,
            X_batch=Xte,
            pca=pca,
            normalize_in=normalize_const_in,
            normalize_out=normalize_const_out,
            device=device,
        )

    core_times = []
    post_times = []
    total_times = []

    for _ in range(repeats):
        _, elapsed_core, elapsed_post = predict_with_trained_bundle(
            trained=trained,
            X_batch=Xte,
            pca=pca,
            normalize_in=normalize_const_in,
            normalize_out=normalize_const_out,
            device=device,
        )
        core_times.append(elapsed_core)
        post_times.append(elapsed_post)
        total_times.append(elapsed_core + elapsed_post)

    n_samples = len(Xte)

    return {
        "n_samples": int(n_samples),
        "repeats": int(repeats),
        "core_time_mean_total_seconds": float(np.mean(core_times)),
        "core_time_median_total_seconds": float(np.median(core_times)),
        "post_time_mean_total_seconds": float(np.mean(post_times)),
        "post_time_median_total_seconds": float(np.median(post_times)),
        "total_time_mean_total_seconds": float(np.mean(total_times)),
        "total_time_median_total_seconds": float(np.median(total_times)),
        "core_time_mean_per_sample_seconds": float(np.mean(core_times) / n_samples),
        "post_time_mean_per_sample_seconds": float(np.mean(post_times) / n_samples),
        "total_time_mean_per_sample_seconds": float(np.mean(total_times) / n_samples),
    }

def encode_raw_input_field_to_flat_vector(
    *,
    raw_input_field: dict,
    N_ex_input: int,
    basis_a: str,
    N_quad: int = 50,
):
    coeffs, _ = encode_single_field_to_coeffs(
        raw_input_field["x"],
        raw_input_field["y"],
        raw_input_field["values"],
        N_ex=N_ex_input,
        basis_type=basis_a,
        N_quad=N_quad,
    )
    coeffs_batch = coeffs[None, ...]
    X_flat = flatten_coeff_tensor(coeffs_batch)
    return X_flat, coeffs

# timing the trained model for each step it takes to produce a solution
def time_surrogate_end_to_end(
    *,
    out: dict,
    cfg,
    repeats: int = 10,
    warmup: int = 2,
    batch_size: int = 16,
):
    raw_inputs = out["raw_test_input_fields"]
    trained = out["trained"]
    pca = out["pca"]

    N_ex_input = int(getattr(cfg, "N_ex_input", getattr(cfg, "N_ex", None)))
    basis_a = cfg.basis_a
    basis_u = cfg.basis_u
    N_quad = int(getattr(cfg, "N_quad", 50))
    normalize_const_in = float(getattr(cfg, "normalize_const_in", 1.0))
    normalize_const_out = float(getattr(cfg, "normalize_const_out", 1.0))
    device = getattr(cfg, "device", "cpu")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def iter_batches(seq, batch_size):
        for start in range(0, len(seq), batch_size):
            yield seq[start:start + batch_size]

    # warmup
    for _ in range(warmup):
        for raw_batch in iter_batches(raw_inputs[: min(len(raw_inputs), max(batch_size, 3))], batch_size):
            X_batch_list = []
            for raw_input in raw_batch:
                X_flat, _ = encode_raw_input_field_to_flat_vector(
                    raw_input_field=raw_input,
                    N_ex_input=N_ex_input,
                    basis_a=basis_a,
                    N_quad=N_quad,
                )
                X_batch_list.append(X_flat[0])

            X_batch = np.stack(X_batch_list, axis=0)

            Yhat_mat_batch, _, _ = predict_with_trained_bundle(
                trained=trained,
                X_batch=X_batch,
                pca=pca,
                normalize_in=normalize_const_in,
                normalize_out=normalize_const_out,
                device=device,
            )

            Yhat_mat_batch = np.asarray(Yhat_mat_batch)
            if Yhat_mat_batch.ndim == 2:
                Yhat_mat_batch = Yhat_mat_batch[None, ...]
            elif Yhat_mat_batch.ndim != 3:
                raise ValueError(
                    f"Unexpected reconstructed batch shape in warmup: {Yhat_mat_batch.shape}"
                )

            for i in range(Yhat_mat_batch.shape[0]):
                _ = evaluate_series_to_plot_points(
                    coeffs=Yhat_mat_batch[i],
                    basis_type=basis_u,
                    N_quad=N_quad,
                )

    encode_times = []
    predict_times = []
    decode_times = []
    total_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()

        encode_acc = 0.0
        predict_acc = 0.0
        decode_acc = 0.0

        for raw_batch in iter_batches(raw_inputs, batch_size):
            t_encode_0 = time.perf_counter()
            X_batch_list = []

            for raw_input in raw_batch:
                X_flat, _ = encode_raw_input_field_to_flat_vector(
                    raw_input_field=raw_input,
                    N_ex_input=N_ex_input,
                    basis_a=basis_a,
                    N_quad=N_quad,
                )
                X_batch_list.append(X_flat[0])

            X_batch = np.stack(X_batch_list, axis=0)
            encode_acc += time.perf_counter() - t_encode_0

            Yhat_mat_batch, elapsed_core, elapsed_post = predict_with_trained_bundle(
                trained=trained,
                X_batch=X_batch,
                pca=pca,
                normalize_in=normalize_const_in,
                normalize_out=normalize_const_out,
                device=device,
            )
            predict_acc += elapsed_core + elapsed_post

            t_decode_0 = time.perf_counter()

            Yhat_mat_batch = np.asarray(Yhat_mat_batch)
            if Yhat_mat_batch.ndim == 2:
                Yhat_mat_batch = Yhat_mat_batch[None, ...]
            elif Yhat_mat_batch.ndim != 3:
                raise ValueError(
                    f"Unexpected reconstructed batch shape before decoding: {Yhat_mat_batch.shape}"
                )

            for i in range(Yhat_mat_batch.shape[0]):
                _ = evaluate_series_to_plot_points(
                    coeffs=Yhat_mat_batch[i],
                    basis_type=basis_u,
                    N_quad=N_quad,
                )

            decode_acc += time.perf_counter() - t_decode_0

        total_elapsed = time.perf_counter() - t0

        encode_times.append(encode_acc)
        predict_times.append(predict_acc)
        decode_times.append(decode_acc)
        total_times.append(total_elapsed)

    n_samples = len(raw_inputs)

    return {
        "n_samples": int(n_samples),
        "repeats": int(repeats),
        "batch_size": int(batch_size),
        "encode_mean_total_seconds": float(np.mean(encode_times)),
        "predict_mean_total_seconds": float(np.mean(predict_times)),
        "decode_mean_total_seconds": float(np.mean(decode_times)),
        "total_mean_total_seconds": float(np.mean(total_times)),
        "encode_mean_per_sample_seconds": float(np.mean(encode_times) / n_samples),
        "predict_mean_per_sample_seconds": float(np.mean(predict_times) / n_samples),
        "decode_mean_per_sample_seconds": float(np.mean(decode_times) / n_samples),
        "total_mean_per_sample_seconds": float(np.mean(total_times) / n_samples),
    }


if __name__ == "__main__":
    main()
    feature_size_test()