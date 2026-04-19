import numpy as np
from matplotlib import pyplot as plt

def rel_l2(a, b, eps=1e-12):
    return float(np.linalg.norm(a-b) / (np.linalg.norm(b) + eps))

def nearest_time_index(t: np.ndarray, t1: float) -> int:
    return int(np.argmin(np.abs(t - float(t1))))

def expansion_error_curve_for_one_snapshot(
    data_dir: str,
    idx: int,
    t1: float,
    Kx_list,
    load_solution_fn,
    encode_fn,
    decode_fn,
    use_energy_tail: bool = True,
):
    """
    Returns dict with arrays for plotting.
    Assumption (hypothesis): your encode_fn stores the true slice in y_meta["u_slice"].
    If not, we fall back to U[ti,:].
    """
    t, x, U = load_solution_fn(data_dir, int(idx))
    ti = nearest_time_index(t, t1)
    u_true = np.asarray(U[ti, :], dtype=np.float64)

    rel_errs = []
    abs_errs = []
    energy_tails = []

    # If you want an energy-tail metric consistent with your encode/decode FFT conventions,
    # it's better to compute it via your own internals. Here we compute from the encoded vector:
    # energy_head = ||y_vec||^2, energy_tail unknown unless we also compute full FFT.
    # So below I compute energy tail by doing full rFFT directly as an *auxiliary diagnostic*.
    if use_energy_tail:
        # This uses numpy's rfft with norm="ortho". If your encoder uses a different norm,
        # the tail ratio is still invariant up to a constant (ratio cancels) as long as it's consistent.
        uhat_full = np.fft.rfft(u_true, norm="ortho")
        full_energy = float(np.sum(np.abs(uhat_full)**2) + 1e-18)

    for Kx in Kx_list:
        y_vec, y_meta = encode_fn(t, x, U, t_index=ti, Kx=int(Kx))
        u_rec = decode_fn(y_vec, y_meta)

        u_ref = np.asarray(y_meta.get("u_slice", u_true), dtype=np.float64)

        abs_e = float(np.linalg.norm(u_rec - u_ref))
        rel_e = rel_l2(u_rec, u_ref)

        abs_errs.append(abs_e)
        rel_errs.append(rel_e)

        if use_energy_tail:
            head_energy = float(np.sum(np.abs(uhat_full[: int(Kx)])**2))
            energy_tails.append(float(1.0 - head_energy / full_energy))

    out = {
        "idx": int(idx),
        "t1": float(t1),
        "ti": int(ti),
        "t1_actual": float(t[ti]),
        "Kx": np.asarray(list(Kx_list), dtype=int),
        "rel_l2": np.asarray(rel_errs, dtype=float),
        "abs_l2": np.asarray(abs_errs, dtype=float),
    }
    if use_energy_tail:
        out["energy_tail"] = np.asarray(energy_tails, dtype=float)
    return out


def plot_expansion_error_curve(curve, title_suffix=""):
    Kx = curve["Kx"]
    plt.figure()
    plt.semilogy(Kx, curve["rel_l2"], marker="o")
    plt.xlabel("Kx (number of rFFT modes kept)")
    plt.ylabel("relative L2 error ||u_Kx - u|| / ||u||")
    plt.title(f"Truncation error vs Kx (idx={curve['idx']}, t={curve['t1_actual']:.4f}) {title_suffix}")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

    if "energy_tail" in curve:
        plt.figure()
        plt.semilogy(Kx, curve["energy_tail"], marker="o")
        plt.xlabel("Kx")
        plt.ylabel("energy tail 1 - sum_{k<Kx}|û_k|^2 / sum_k|û_k|^2")
        plt.title(f"Spectral energy tail vs Kx (idx={curve['idx']}, t={curve['t1_actual']:.4f}) {title_suffix}")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.show()