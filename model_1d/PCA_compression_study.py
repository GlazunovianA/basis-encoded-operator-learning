import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from pathlib import Path

from model_burgers import (
    load_solution,
    ensure_dir,
    Scheme1Cache,
    weighted_global_rel_l2,
)
from test_new_scheme.patch_t_encoding import Scheme1CacheDiscreteTimeONB


def save_true_output_bundle(
    path: str,
    *,
    Ytrue: np.ndarray,
    meta: dict,
    indices: np.ndarray,
):
    ensure_dir(os.path.dirname(path))
    np.savez(
        path,
        Ytrue=np.asarray(Ytrue, dtype=float),
        meta=np.array(meta, dtype=object),
        indices=np.asarray(indices, dtype=int),
    )


def load_true_output_bundle(path: str):
    d = np.load(path, allow_pickle=True)
    return {
        "Ytrue": d["Ytrue"],
        "meta": d["meta"].item(),
        "indices": d["indices"],
    }


def _build_meta_from_cache(cache, t: np.ndarray, x: np.ndarray, cache_name: str) -> dict:
    return {
        "cache_name": cache_name,
        "Nt": cache.Nt,
        "Kx": cache.Kx,
        "T_grid": cache.T,
        "Nx_grid": cache.N_x,
        "Kx_full": cache.K_full,
        "t_grid": np.asarray(t, dtype=float),
        "x_grid": np.asarray(x, dtype=float),
        "d_out": 2 * cache.Nt * cache.Kx,
    }


def _decode_with_cache(cache, y_vec: np.ndarray, t_grid: np.ndarray | None = None) -> np.ndarray:
    if isinstance(cache, Scheme1CacheDiscreteTimeONB):
        return cache.decode(y_vec)
    return cache.decode(y_vec, t_grid=t_grid)


def _collect_encoded_outputs(
    data_dir: str,
    indices: np.ndarray,
    *,
    cache_cls,
    Nt: int | None = None,
    Kx: int | None = None,
    compute_errors: bool = False,
    n_print: int = 10,
):
    indices = np.asarray(indices, dtype=int)

    encoded_output_vectors = []
    reconstruction_errors = []

    cache = None
    meta = None

    for i, idx in enumerate(indices):
        t, x, U_true = load_solution(data_dir, int(idx))

        if cache is None:
            cache = cache_cls(t_grid=t, x_grid=x, Nt=Nt, Kx=Kx)
            meta = _build_meta_from_cache(cache, t, x, cache_cls.__name__)

            print(
                f"{cache_cls.__name__} using T={cache.T}, Nx={cache.N_x}, "
                f"Nt={cache.Nt}, Kx={cache.Kx}, Kx_full={cache.K_full}"
            )
            print(
                f"Time Gram diagnostics: ||Gt-I||={cache.Gt_minus_I_norm:.6e}, "
                f"cond(Gt)={cache.Gt_cond:.6e}"
            )

        y_vec = cache.encode(U_true)
        encoded_output_vectors.append(y_vec)

        if compute_errors:
            U_rec = _decode_with_cache(cache, y_vec, t_grid=t)
            err = weighted_global_rel_l2(U_rec, U_true, t, x)
            reconstruction_errors.append(err)

            if i < n_print:
                print(f"[ex {i}] baseline representation error = {err:.6e}")

    Ytrue = np.stack(encoded_output_vectors, axis=0)

    result = {
        "Ytrue": Ytrue,
        "meta": meta,
        "indices": indices,
    }

    if compute_errors:
        errs = np.asarray(reconstruction_errors, dtype=float)
        result["all_errors"] = errs
        result["mean"] = float(np.mean(errs))
        result["median"] = float(np.median(errs))
        result["max"] = float(np.max(errs))

    return result


def build_true_output_data(
    data_dir: str,
    indices: np.ndarray,
    Nt: int | None = None,
    Kx: int | None = None,
):
    bundle = _collect_encoded_outputs(
        data_dir=data_dir,
        indices=indices,
        cache_cls=Scheme1Cache,
        Nt=Nt,
        Kx=Kx,
        compute_errors=False,
    )
    return bundle["Ytrue"], bundle["meta"]


def study_baseline_representation_error(
    data_dir: str,
    indices: np.ndarray,
    Nt: int | None = None,
    Kx: int | None = None,
    n_show: int = 10,
):
    bundle = _collect_encoded_outputs(
        data_dir=data_dir,
        indices=indices,
        cache_cls=Scheme1Cache,
        Nt=Nt,
        Kx=Kx,
        compute_errors=True,
        n_print=n_show,
    )
    return {
        "bundle": {
            "Ytrue": bundle["Ytrue"],
            "meta": bundle["meta"],
            "indices": bundle["indices"],
        },
        "mean": bundle["mean"],
        "median": bundle["median"],
        "max": bundle["max"],
        "all_errors": bundle["all_errors"],
    }


def study_baseline_discrete_time_onb(
    data_dir: str,
    indices: np.ndarray,
    Nt: int | None,
    Kx: int | None,
    n_show: int = 10,
):
    bundle = _collect_encoded_outputs(
        data_dir=data_dir,
        indices=indices,
        cache_cls=Scheme1CacheDiscreteTimeONB,
        Nt=Nt,
        Kx=Kx,
        compute_errors=True,
        n_print=n_show,
    )
    return {
        "bundle": {
            "Ytrue": bundle["Ytrue"],
            "meta": bundle["meta"],
            "indices": bundle["indices"],
        },
        "mean": bundle["mean"],
        "median": bundle["median"],
        "max": bundle["max"],
        "all_errors": bundle["all_errors"],
    }


def pca_representation_study_true_Y(
    *,
    Ytrue: np.ndarray,
    meta: dict,
    data_dir: str,
    indices: np.ndarray,
    ranks=(16, 32, 64, 128, 256, 512),
    n_show=10,
):
    Nt = int(meta["Nt"])
    Kx = int(meta["Kx"])
    t_grid = np.asarray(meta["t_grid"], dtype=float)
    x_grid = np.asarray(meta["x_grid"], dtype=float)
    cache_name = meta["cache_name"]

    cache_cls_map = {
        "Scheme1Cache": Scheme1Cache,
        "Scheme1CacheDiscreteTimeONB": Scheme1CacheDiscreteTimeONB,
    }
    if cache_name not in cache_cls_map:
        raise ValueError(f"Unknown cache_name in meta: {cache_name}")

    cache = cache_cls_map[cache_name](t_grid=t_grid, x_grid=x_grid, Nt=Nt, Kx=Kx)

    Ytrue = np.asarray(Ytrue, dtype=float)
    indices = np.asarray(indices, dtype=int)

    n_samples, D = Ytrue.shape
    expected_D = 2 * Nt * Kx
    if D != expected_D:
        raise ValueError(f"Ytrue has dimension {D}, expected {expected_D}")

    max_rank = min(n_samples, D, max(ranks))
    valid_ranks = [int(r) for r in ranks if int(r) <= max_rank]
    if not valid_ranks:
        raise ValueError(f"No valid PCA ranks. max admissible rank is {max_rank}")

    pca = PCA(n_components=max_rank)
    pca.fit(Ytrue)

    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)

    plt.figure()
    plt.semilogy(np.arange(1, len(evr) + 1), 1.0 - cumevr, marker="o")
    plt.xlabel("PCA rank r")
    plt.ylabel("1 - cumulative explained variance")
    plt.title("True Y-space PCA residual energy")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

    rows = []
    show_ids = list(range(min(n_show, n_samples)))

    for i in show_ids:
        idx0 = int(indices[i])
        t, x, U_true = load_solution(data_dir, idx0)

        U_repr_full = _decode_with_cache(cache, Ytrue[i], t_grid=t)
        base_repr_err = weighted_global_rel_l2(U_repr_full, U_true, t, x)

        Zi_full = pca.transform(Ytrue[i:i + 1])

        print(f"[ex {i}] baseline representation error = {base_repr_err:.6e}")

        for r in valid_ranks:
            Zi_r = Zi_full[:, :r]
            Yi_r = pca.mean_ + Zi_r @ pca.components_[:r, :]

            U_repr_pca = _decode_with_cache(cache, Yi_r[0], t_grid=t)
            repr_err_pca = weighted_global_rel_l2(U_repr_pca, U_true, t, x)

            rows.append(
                {
                    "example": i,
                    "dataset_index": idx0,
                    "rank": r,
                    "repr_err": repr_err_pca,
                    "base_repr_err": base_repr_err,
                    "extra_repr_err": repr_err_pca - base_repr_err,
                }
            )

    results_df = pd.DataFrame(rows)

    if len(results_df) == 0:
        return {
            "pca": pca,
            "results_df": results_df,
            "grouped_df": pd.DataFrame(),
        }

    grouped_df = results_df.groupby("rank").mean(numeric_only=True).reset_index()

    plt.figure()
    plt.plot(grouped_df["rank"], grouped_df["extra_repr_err"], marker="o")
    plt.xlabel("PCA rank r")
    plt.ylabel("Δ global rel L2 error")
    plt.title("Extra function-space error from PCA compression of true Y")
    plt.grid(True, ls="--", alpha=0.3)
    plt.show()

    plt.figure()
    plt.plot(grouped_df["rank"], grouped_df["repr_err"], marker="o", label="repr error after PCA")
    plt.plot(grouped_df["rank"], grouped_df["base_repr_err"], marker="o", label="baseline repr error")
    plt.xlabel("PCA rank r")
    plt.ylabel("global rel L2 error")
    plt.title("Function-space representation error vs PCA rank")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    print(grouped_df)

    return {
        "pca": pca,
        "results_df": results_df,
        "grouped_df": grouped_df,
    }


''' time slice PCA study: how much error is introduced by PCA compression at time t? '''

# ============================================================
# New: time-slice PCA study on U(t_k, x)
# ============================================================

def rel_l2_x(u_hat: np.ndarray, u_true: np.ndarray) -> float:
    u_hat = np.asarray(u_hat, dtype=float)
    u_true = np.asarray(u_true, dtype=float)
    return float(np.linalg.norm(u_hat - u_true) / (np.linalg.norm(u_true) + 1e-12))


def choose_representative_time_indices(
    t_grid: np.ndarray,
    *,
    explicit_times: tuple[float, ...] | None = None,
    num_quantiles: int = 6,
    include_last: bool = True,
):
    t_grid = np.asarray(t_grid, dtype=float)

    if explicit_times is not None:
        time_indices = []
        for t0 in explicit_times:
            j = int(np.argmin(np.abs(t_grid - float(t0))))
            time_indices.append(j)
        time_indices = np.array(sorted(set(time_indices)), dtype=int)
        return time_indices

    raw = np.linspace(0, len(t_grid) - 1, num_quantiles)
    time_indices = np.unique(np.round(raw).astype(int))

    if include_last and time_indices[-1] != len(t_grid) - 1:
        time_indices = np.concatenate([time_indices, np.array([len(t_grid) - 1], dtype=int)])

    return np.array(sorted(set(time_indices.tolist())), dtype=int)


def collect_true_solution_slices(
    data_dir: str,
    indices: np.ndarray,
):
    """
    Returns
    -------
    all_solutions : ndarray, shape (n_samples, N_t, N_x)
    t_grid : ndarray, shape (N_t,)
    x_grid : ndarray, shape (N_x,)
    """
    indices = np.asarray(indices, dtype=int)

    solution_snapshots = []
    t_ref = None
    x_ref = None

    for idx in indices:
        t, x, U_true = load_solution(data_dir, int(idx))

        if t_ref is None:
            t_ref = np.asarray(t, dtype=float)
            x_ref = np.asarray(x, dtype=float)
        else:
            if len(t) != len(t_ref) or not np.allclose(t, t_ref):
                raise ValueError("Time grid mismatch across samples")
            if len(x) != len(x_ref) or not np.allclose(x, x_ref):
                raise ValueError("Space grid mismatch across samples")

        solution_snapshots.append(np.asarray(U_true, dtype=float))

    all_solutions = np.stack(solution_snapshots, axis=0)   # (n_samples, N_t, N_x)
    return all_solutions, t_ref, x_ref


def fit_slice_pca_and_measure_errors(
    slice_matrix: np.ndarray,
    ranks=(4, 8, 16, 32, 64, 128),
):
    """
    slice_matrix shape: (n_samples, N_x)
    PCA is fitted across samples for one fixed time slice.
    """
    slice_matrix = np.asarray(slice_matrix, dtype=float)
    n_samples, n_x = slice_matrix.shape

    max_rank = min(n_samples, n_x, max(ranks))
    valid_ranks = [int(r) for r in ranks if int(r) <= max_rank]
    if not valid_ranks:
        raise ValueError(f"No valid ranks for slice PCA. max admissible rank is {max_rank}")

    pca = PCA(n_components=max_rank)
    Z_full = pca.fit_transform(slice_matrix)

    rows = []
    for i in range(n_samples):
        ui_true = slice_matrix[i]
        zi_full = Z_full[i:i + 1]

        for r in valid_ranks:
            zi_r = zi_full[:, :r]
            ui_hat = pca.mean_ + zi_r @ pca.components_[:r, :]
            err = rel_l2_x(ui_hat[0], ui_true)

            rows.append(
                {
                    "example": i,
                    "rank": r,
                    "slice_rel_l2_x": err,
                }
            )

    results_df = pd.DataFrame(rows)
    grouped_df = results_df.groupby("rank").agg(
        slice_rel_l2_x_mean=("slice_rel_l2_x", "mean"),
        slice_rel_l2_x_median=("slice_rel_l2_x", "median"),
        slice_rel_l2_x_max=("slice_rel_l2_x", "max"),
    ).reset_index()

    evr = pca.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    residual_energy = 1.0 - cumevr

    residual_energy_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(residual_energy) + 1, dtype=int),
            "residual_energy": residual_energy,
        }
    )

    return {
        "pca": pca,
        "results_df": results_df,
        "grouped_df": grouped_df,
        "residual_energy_df": residual_energy_df,
        "valid_ranks": valid_ranks,
    }


def time_slice_pca_study_true_solution(
    *,
    data_dir: str,
    indices: np.ndarray,
    ranks=(4, 8, 16, 32, 64, 128),
    explicit_times: tuple[float, ...] | None = None,
    num_quantiles: int = 6,
    n_show: int = 0,
):
    """
    For representative time points t_k:
      - collect U_i(t_k, :)
      - perform PCA on x-slices across samples
      - measure rank-r slice reconstruction error
    """
    indices = np.asarray(indices, dtype=int)

    all_solutions, t_grid, x_grid = collect_true_solution_slices(
        data_dir=data_dir,
        indices=indices,
    )
    n_samples, n_t, n_x = all_solutions.shape

    time_indices = choose_representative_time_indices(
        t_grid,
        explicit_times=explicit_times,
        num_quantiles=num_quantiles,
        include_last=True,
    )

    all_rows = []
    summary_rows = []
    residual_rows = []
    per_time_objects = {}

    print(f"time-slice PCA study on {n_samples} samples, N_t={n_t}, N_x={n_x}")
    print(f"chosen time indices: {time_indices.tolist()}")
    print(f"chosen times: {[float(t_grid[j]) for j in time_indices]}")

    for j in time_indices:
        t_value = float(t_grid[j])
        slice_matrix = all_solutions[:, j, :]   # (n_samples, N_x)

        study = fit_slice_pca_and_measure_errors(
            slice_matrix=slice_matrix,
            ranks=ranks,
        )

        results_df = study["results_df"].copy()
        grouped_df = study["grouped_df"].copy()
        residual_energy_df = study["residual_energy_df"].copy()

        results_df["time_index"] = int(j)
        results_df["time_value"] = t_value

        grouped_df["time_index"] = int(j)
        grouped_df["time_value"] = t_value

        residual_energy_df["time_index"] = int(j)
        residual_energy_df["time_value"] = t_value

        all_rows.append(results_df)
        summary_rows.append(grouped_df)
        residual_rows.append(residual_energy_df)

        per_time_objects[int(j)] = study

        print(f"[time index {j:4d}, t={t_value:.6f}]")
        print(grouped_df)

        if n_show > 0:
            show_count = min(n_show, n_samples)
            for i in range(show_count):
                ui_true = slice_matrix[i]
                zi_full = study["pca"].transform(slice_matrix[i:i + 1])

                print(f"  example {i}:")
                for r in study["valid_ranks"]:
                    zi_r = zi_full[:, :r]
                    ui_hat = study["pca"].mean_ + zi_r @ study["pca"].components_[:r, :]
                    err = rel_l2_x(ui_hat[0], ui_true)
                    print(f"    rank={r:4d} | slice rel L2_x = {err:.6e}")

    all_results_df = pd.concat(all_rows, axis=0, ignore_index=True)
    summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
    residual_energy_df = pd.concat(residual_rows, axis=0, ignore_index=True)

    return {
        "all_solutions_shape": all_solutions.shape,
        "t_grid": t_grid,
        "x_grid": x_grid,
        "time_indices": time_indices,
        "results_df": all_results_df,
        "summary_df": summary_df,
        "residual_energy_df": residual_energy_df,
        "per_time_objects": per_time_objects,
    }


def plot_time_slice_pca_summary(
    summary_df: pd.DataFrame,
    residual_energy_df: pd.DataFrame,
    *,
    selected_ranks: tuple[int, ...] | None = None,
):
    if len(summary_df) == 0:
        return

    summary_df = summary_df.copy()
    residual_energy_df = residual_energy_df.copy()

    available_ranks = sorted(summary_df["rank"].unique().tolist())
    if selected_ranks is None:
        selected_ranks = tuple(available_ranks[:min(4, len(available_ranks))])

    plt.figure()
    for r in selected_ranks:
        part = summary_df[summary_df["rank"] == r].sort_values("time_value")
        if len(part) == 0:
            continue
        plt.plot(
            part["time_value"],
            part["slice_rel_l2_x_mean"],
            marker="o",
            label=f"rank={r}",
        )
    plt.xlabel("time t")
    plt.ylabel("mean slice rel L2_x error")
    plt.title("Time-slice PCA error vs time")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    chosen_times = sorted(summary_df["time_value"].unique().tolist())
    plt.figure()
    for t0 in chosen_times:
        part = summary_df[np.isclose(summary_df["time_value"], t0)].sort_values("rank")
        plt.plot(
            part["rank"],
            part["slice_rel_l2_x_mean"],
            marker="o",
            label=f"t={t0:.3f}",
        )
    plt.xlabel("PCA rank r")
    plt.ylabel("mean slice rel L2_x error")
    plt.title("Time-slice PCA error vs rank")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure()
    for t0 in chosen_times:
        part = residual_energy_df[np.isclose(residual_energy_df["time_value"], t0)].sort_values("rank")
        plt.semilogy(
            part["rank"],
            part["residual_energy"],
            marker="o",
            label=f"t={t0:.3f}",
        )
    plt.xlabel("PCA rank r")
    plt.ylabel("1 - cumulative explained variance")
    plt.title("Residual PCA energy of time slices")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data_root = here.parent / "data"
    data_dir = str(data_root / "viscous_burgers_utx_1d")
    cache_dir = str(data_root / "cache" / "burgers_1d")

    train_size = 8000
    test_size = 1200
    test_indices = np.arange(train_size, train_size + test_size)

    baseline = study_baseline_discrete_time_onb(
        data_dir=data_dir,
        indices=test_indices,
        Nt=None,
        Kx=None,
        n_show=10,
    )

    bundle_max_precision = baseline["bundle"]

    study = pca_representation_study_true_Y(
        Ytrue=bundle_max_precision["Ytrue"],
        meta=bundle_max_precision["meta"],
        data_dir=data_dir,
        indices=bundle_max_precision["indices"],
        ranks=(16, 32, 48, 64, 96, 128, 256, 512, 1024),
        n_show=10,
    )
    
    slice_study = time_slice_pca_study_true_solution(
        data_dir=data_dir,
        indices=bundle_max_precision["indices"],
        ranks=(4, 8, 16, 32, 64, 128, 256),
        explicit_times=None,
        num_quantiles=6,
        n_show=0,
    )

    plot_time_slice_pca_summary(
        summary_df=slice_study["summary_df"],
        residual_energy_df=slice_study["residual_energy_df"],
        selected_ranks=(8, 16, 32, 64, 128),
    )

    print(slice_study["summary_df"])
    # print(baseline)