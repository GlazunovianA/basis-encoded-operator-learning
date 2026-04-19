import os
import numpy as np
import torch

from io_pipeline import load_solution
from encoding_decoding import (
    encode_output_scheme2_slice_fourier,
    decode_output_scheme2_slice_fourier,
    encode_output_scheme1_legendre_time_fourier_space,
    decode_output_scheme1_legendre_time_fourier_space,
    trapezoid_weights,
)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # .../neuralsurrogate_reproduce
sys.path.insert(0, str(ROOT))

from G_N import NN_cff_vec, RFF 



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


# -------------------------
# Ridge helpers (shared + non-shared)
# -------------------------
def ridge_fit_shared(Phi: torch.Tensor, Y: torch.Tensor, lam: float):
    # Phi: (n,F), Y: (n,D) -> W: (F,D)
    n, F = Phi.shape
    I = torch.eye(F, dtype=Phi.dtype, device=Phi.device)
    A = Phi.T @ Phi + float(lam) * I
    B = Phi.T @ Y
    return torch.linalg.solve(A, B)

def ridge_predict_shared(Phi: torch.Tensor, W: torch.Tensor):
    return Phi @ W

def ridge_fit_multioutput_features(features: torch.Tensor, Y: torch.Tensor, lam: float):
    """
    features: (n, F, D), Y: (n, D) -> W: (F, D)
    """
    n, F, D = features.shape
    W = torch.empty((F, D), dtype=features.dtype, device=features.device)
    I = torch.eye(F, dtype=features.dtype, device=features.device)
    for d in range(D):
        Phi = features[:, :, d]        # (n, F)
        A = Phi.T @ Phi + lam * I      # (F, F)
        b = Phi.T @ Y[:, d]            # (F,)
        W[:, d] = torch.linalg.solve(A, b)
    return W

def predict_multioutput_features(features: torch.Tensor, W: torch.Tensor):
    # features: (n,F,D), W: (F,D) -> (n,D)
    return torch.einsum("nfd,fd->nd", features, W)


# -------------------------
# Dataset builders
# -------------------------
def build_dataset_task_step(data_dir: str, indices: np.ndarray, Kx: int, t1: float):
    X_list, Y_list = [], []
    meta_ref = None
    for idx in indices:
        idx = int(idx)
        t, x, U = load_solution(data_dir, idx)
        ti = nearest_time_index(t, t1)
        y_vec, y_meta = encode_output_scheme2_slice_fourier(t, x, U, t_index=ti, Kx=Kx)

        # NOTE: input for Task A is not shown here because in your current code it uses load_parameter(...)
        # Keep your existing Task A builder if you already have it.
        raise NotImplementedError("Use your existing Task A builder here.")

    # return X, Y, meta_ref




def _step_pairs_from_mode(t_grid: np.ndarray, dt: float, mode: str, t_from: float | None, t_to: float | None,
                          t_starts: list[float] | None, t_window: tuple[float, float] | None):
    """
    Returns list of (i_from, i_to) index pairs on this t_grid.
    """
    if mode == "single":
        assert t_from is not None and t_to is not None
        i_from = nearest_time_index(t_grid, t_from)
        i_to = nearest_time_index(t_grid, t_to)
        return [(i_from, i_to)]

    if mode == "multi":
        pairs = []
        if t_starts is not None:
            for ts in t_starts:
                i_from = nearest_time_index(t_grid, ts)
                i_to = nearest_time_index(t_grid, ts + dt)
                pairs.append((i_from, i_to))
            return pairs

        # auto-generate from a window [a,b]
        assert t_window is not None, "For stepper_mode='multi' provide t_starts or t_window=(a,b)"
        a, b = t_window
        # choose all grid points within [a, b-dt]
        for i_from, t0 in enumerate(t_grid):
            if t0 < a:
                continue
            if t0 > b - dt:
                break
            i_to = nearest_time_index(t_grid, float(t0 + dt))
            pairs.append((i_from, i_to))
        return pairs

    raise ValueError(f"Unknown stepper_mode={mode}")


def build_dataset_task_stepper(
    data_dir: str,
    indices: np.ndarray,
    Kx: int,
    dt: float,
    stepper_mode: str = "single",           # "single" or "multi"
    t_from: float | None = None,            # for single
    t_to: float | None = None,              # for single
    t_starts: list[float] | None = None,    # for multi
    t_window: tuple[float, float] | None = None,  # for multi
    include_time: bool = False,             # extra option: append t as input feature
):
    """
    Learns c(t) -> c(t+dt). Input X is coeffs at t (and optionally time), target Y is coeffs at t+dt.
    Output coeff dimension is Kx (your N_basis / Kx reinterpretation).
    """
    X_list, Y_list = [], []
    meta_ref = None

    for idx in indices:
        idx = int(idx)
        t, x, U = load_solution(data_dir, idx)

        pairs = _step_pairs_from_mode(t, dt, stepper_mode, t_from, t_to, t_starts, t_window)
        for (i_from, i_to) in pairs:
            c_from, meta_from = encode_output_scheme2_slice_fourier(t, x, U, t_index=i_from, Kx=Kx)
            c_to, meta_to = encode_output_scheme2_slice_fourier(t, x, U, t_index=i_to, Kx=Kx)

            if include_time:
                X_list.append(np.concatenate([c_from, [float(t[i_from])]], axis=0))
            else:
                X_list.append(c_from)
            Y_list.append(c_to)

            if meta_ref is None:
                meta_ref = {
                    "task": "stepper",
                    "Kx": int(Kx),
                    "dt_requested": float(dt),
                    "stepper_mode": stepper_mode,
                    "include_time": bool(include_time),
                    "x_grid": x,
                    "t_grid": t,
                    "y_meta": meta_to,         # convenient decode meta for outputs
                    "y_meta_from": meta_from,
                    "y_meta_to": meta_to,
                }

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta_ref


def build_dataset_task_stepper_fast(
    data_dir: str,
    indices: np.ndarray,
    Kx: int,
    dt: float,
    t_starts: list[float],          # explicit list to avoid explosion
):
    from encoding_decoding import eval_basis_matrix
    X_list, Y_list = [], []
    meta_ref = None

    A = None
    base_meta = None

    for idx in indices:
        idx = int(idx)
        t, x, U = load_solution(data_dir, idx)

        # init cached projector once (x is shared across dataset)
        if A is None:
            N_x = len(x)
            w = np.full(N_x, 1.0 / N_x)
            Phi = eval_basis_matrix("fourier_periodic", int(Kx), x)   # (Kx, N_x)
            A = Phi * w[None, :]  # (Kx, N_x)
            base_meta = {
                "scheme": "slice_real_fourier_coeffs",
                "basis": "fourier_periodic",
                "N_basis": int(Kx),
                "N_x": int(N_x),
                "x_grid": np.asarray(x),
                "quad": "uniform_on_01",
                "w_x": w,
            }

        for ts in t_starts:
            i_from = nearest_time_index(t, float(ts))
            i_to   = nearest_time_index(t, float(ts + dt))

            u_from = U[i_from, :]
            u_to   = U[i_to, :]

            c_from = (A @ u_from).astype(np.float64)
            c_to   = (A @ u_to).astype(np.float64)

            X_list.append(c_from)
            Y_list.append(c_to)

            if meta_ref is None:
                meta_ref = {
                    "task": "stepper",
                    "Kx": int(Kx),
                    "dt_requested": float(dt),
                    "t_starts": list(t_starts),
                    "x_grid": x,
                    "t_grid": t,
                    "y_meta": {**base_meta, "t_index": int(i_to), "t_value": float(t[i_to])},
                }

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y, meta_ref


# -------------------------
# PCA wrapper (as in your current code)
# -------------------------
import joblib
from sklearn.decomposition import PCA

class PCATransform:
    def __init__(self, use: bool, n_components: int, path_in: str, path_out: str):
        self.use = bool(use)
        self.n_components = n_components
        self.path_in = path_in
        self.path_out = path_out
        self.pca_in = None
        self.pca_out = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if not self.use:
            return X, Y
        ensure_dir(os.path.dirname(self.path_in))
        ensure_dir(os.path.dirname(self.path_out))
        self.pca_in = PCA(n_components=self.n_components)
        self.pca_out = PCA(n_components=self.n_components)
        Xr = self.pca_in.fit_transform(X)
        Yr = self.pca_out.fit_transform(Y)
        joblib.dump(self.pca_in, self.path_in)
        joblib.dump(self.pca_out, self.path_out)
        return Xr, Yr

    def load_and_apply(self, X: np.ndarray, Y: np.ndarray):
        if not self.use:
            return X, Y
        self.pca_in = joblib.load(self.path_in)
        self.pca_out = joblib.load(self.path_out)
        return self.pca_in.transform(X), self.pca_out.transform(Y)

    def inverse_Y(self, Yhat: np.ndarray):
        if not self.use:
            return Yhat
        return self.pca_out.inverse_transform(Yhat)


# -------------------------
# Training entrypoint
# -------------------------
def train_model(
    *,
    data_dir: str,
    cache_dir: str,
    task: str,                      # "step" | "continuous" | "stepper"
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    Kx: int,
    t1: float | None = None,
    Nt: int | None = None,

    # stepper options (the "other option" you asked for is here)
    dt: float | None = None,
    stepper_mode: str = "single",               # "single" or "multi"
    t_from: float | None = None,
    t_to: float | None = None,
    t_starts: list[float] | None = None,
    t_window: tuple[float, float] | None = None,
    include_time: bool = False,

    # backend options
    model_type: str = "rf",
    shared_kernel: bool = True,

    # PCA / normalization
    use_pca: bool = False,
    pca_components: int = 32,
    normalize_in: float = 1.0,
    normalize_out: float = 1.0,

    # rf hyperparams (match your current defaults)
    d_features: int = 8192,
    lengthscale: float = 0.5,
    lam: float = 1e-4,

    device: str = "cpu",
    force_rebuild: bool = False,
):
    ensure_dir(cache_dir)

    tag = f"{task}_Kx{Kx}_Nt{Nt}_t1{t1}_dt{dt}_mode{stepper_mode}_itime{include_time}_shared{shared_kernel}"
    train_cache = os.path.join(cache_dir, f"{tag}_train.npz")
    test_cache = os.path.join(cache_dir, f"{tag}_test.npz")

    # -------- dataset --------
    if os.path.exists(train_cache) and not force_rebuild:
        d = load_npz(train_cache)
        Xtr, Ytr, meta = d["X"], d["Y"], d["meta"].item()
    else:
        if task == "stepper":
            assert dt is not None
            import time
            t0 = time.time()
            print("building dataset...")
            Xtr, Ytr, meta = build_dataset_task_stepper_fast(
                data_dir=data_dir,
                indices=train_indices,
                Kx=Kx,
                dt=float(dt),
                t_starts=t_starts,
            )
        else:
            raise NotImplementedError("Keep your existing 'step' and 'continuous' dataset builders here.")
        save_npz(train_cache, X=Xtr, Y=Ytr, meta=meta)

    if os.path.exists(test_cache) and not force_rebuild:
        d = load_npz(test_cache)
        Xte, Yte = d["X"], d["Y"]
    else:
        if task == "stepper":
            assert dt is not None
            import time
            t0 = time.time()
            print("loading dataset...")
            Xte, Yte, _ = build_dataset_task_stepper_fast(
                data_dir=data_dir,
                indices=test_indices,
                Kx=Kx,
                dt=float(dt),
                t_starts=t_starts,
            )
        else:
            raise NotImplementedError("Keep your existing 'step' and 'continuous' dataset builders here.")
        save_npz(test_cache, X=Xte, Y=Yte, meta=meta)

    # -------- PCA --------
    pca = PCATransform(
        use=use_pca,
        n_components=pca_components,
        path_in=os.path.join(cache_dir, f"{tag}_pca_in.joblib"),
        path_out=os.path.join(cache_dir, f"{tag}_pca_out.joblib"),
    )
    if use_pca:
        Xtr_p, Ytr_p = pca.fit(Xtr, Ytr)
        Xte_p, Yte_p = pca.load_and_apply(Xte, Yte)
    else:
        Xtr_p, Ytr_p, Xte_p, Yte_p = Xtr, Ytr, Xte, Yte

    # -------- torch --------
    Xtr_t = torch.tensor(Xtr_p, dtype=torch.float32, device=device) * float(normalize_in)
    Ytr_t = torch.tensor(Ytr_p, dtype=torch.float32, device=device) * float(normalize_out)
    Xte_t = torch.tensor(Xte_p, dtype=torch.float32, device=device) * float(normalize_in)
    Yte_t = torch.tensor(Yte_p, dtype=torch.float32, device=device) * float(normalize_out)

    d_in = Xtr_t.shape[1]
    d_out = Ytr_t.shape[1]

    trained = {"model_type": model_type, "shared_kernel": shared_kernel}

    # -------- model --------
    if model_type != "rf":
        raise NotImplementedError("Only rf backend is wired here; keep your nn branch if you want.")

    sigma_in = np.ones(d_in)
    sigma_out = np.ones(d_out)
    t0 = time.time()
    print("computing features...")
    
    if shared_kernel:
        feat_model = RFF(
            d_in=d_in,
            d_features=d_features,
            lengthscale=lengthscale,
            trainable_kernel=False,
            device=device,
        ).to(device)
        feat_model.resample()
        with torch.no_grad():
            Phi_tr = feat_model(Xtr_t)  # (n,F)
            Phi_te = feat_model(Xte_t)
        print("Phi_tr", Phi_tr.shape)
        print("features computed in", time.time() - t0, "sec")

        t0 = time.time()
        print("solving ridge...")
        W = ridge_fit_shared(Phi_tr, Ytr_t, lam=lam)
        Yhat_t = ridge_predict_shared(Phi_te, W)
        print("ridge solved in", time.time() - t0, "sec")

    else:
        feat_model = NN_cff_vec(
            d_in=d_in,
            d_features=d_features,
            d_out=d_out,
            lengthscale=lengthscale,
            sigma_in=sigma_in,
            sigma_out=sigma_out,
            trainable_kernel=False,
            projection_type="Cauchy",
            device=device,
        ).to(device)
        feat_model.resample()
        with torch.no_grad():
            Phi_tr = feat_model(Xtr_t)  # (n,F,D)
            Phi_te = feat_model(Xte_t)

        W = ridge_fit_multioutput_features(Phi_tr, Ytr_t, lam=lam)
        Yhat_t = predict_multioutput_features(Phi_te, W)

    trained.update({
        "feat_model": feat_model,
        "W": W,
        "lam": lam,
        "d_features": d_features,
        "lengthscale": lengthscale,
        "d_in": d_in,
        "d_out": d_out,
        "include_time": include_time,
    })

    # -------- inverse transforms --------
    Yhat = (Yhat_t.detach().cpu().numpy()) / float(normalize_out)
    Ytrue = (Yte_t.detach().cpu().numpy()) / float(normalize_out)
    if use_pca:
        Yhat = pca.inverse_Y(Yhat)
        Ytrue = pca.inverse_Y(Ytrue)

    vec_mse = float(np.mean((Yhat - Ytrue) ** 2))

    return {
        "vec_mse": vec_mse,
        "Yhat": Yhat,
        "Ytrue": Ytrue,
        "meta": meta,
        "pca": pca,
        "trained": trained,
    }


def rollout_stepper(
    *,
    c0: np.ndarray,
    n_steps: int,
    trained: dict,
    pca: PCATransform,
    use_pca: bool,
    normalize_in: float,
    normalize_out: float,
    device: str,
):
    """
    Roll out in coefficient space: c_{m+1} = f(c_m[,t_m]).
    If trained["include_time"]==True, caller must pass c0 with time appended at each step via this function.
    Here we implement the time-appended update internally using dt inferred from caller loop.
    """
    feat_model = trained["feat_model"]
    W = trained["W"]
    shared_kernel = trained["shared_kernel"]
    include_time = trained.get("include_time", False)

    # c state dimension (without time)
    d_state = trained["d_out"]
    c = np.asarray(c0, dtype=np.float64).reshape(1, d_state)
    C = [c[0].copy()]

    # TODO if include_time=True, we cannot infer time progression without user passing it;
    # so we keep it simple: require c0 already includes time as last entry is NOT supported here.
    # Prefer: keep include_time=False for now, or manage time in main.
    if include_time:
        raise ValueError("rollout_stepper: include_time=True requires time-managed input; handle in main explicitly.")

    for _ in range(n_steps):
        c_in = c
        if use_pca:
            c_in = pca.pca_in.transform(c_in)

        x_t = torch.tensor(c_in, dtype=torch.float32, device=device) * float(normalize_in)
        with torch.no_grad():
            Phi = feat_model(x_t)
            if shared_kernel:
                y_t = Phi @ W
            else:
                y_t = torch.einsum("nfd,fd->nd", Phi, W)

        y = (y_t.detach().cpu().numpy()) / float(normalize_out)
        if use_pca:
            y = pca.inverse_Y(y)

        c = y
        C.append(c[0].copy())

    return np.stack(C, axis=0)