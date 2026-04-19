import numpy as np
from matplotlib import pyplot as plt

def rfft_spectrum(u: np.ndarray):
    """Return k indices (1..Kmax) and amplitude/energy spectra excluding k=0."""
    u = np.asarray(u, dtype=np.float64)
    uhat = np.fft.rfft(u, norm="ortho")
    amp = np.abs(uhat)
    eng = amp**2
    k = np.arange(len(uhat))  # includes k=0
    return k[1:], amp[1:], eng[1:]  # drop k=0

def fit_power_law(k, y, kmin, kmax):
    """
    Fit log y = a + b log k on k in [kmin,kmax].
    Returns slope b, intercept a, R^2.
    """
    mask = (k >= kmin) & (k <= kmax) & np.isfinite(y) & (y > 0)
    kk = k[mask]
    yy = y[mask]
    if len(kk) < 5:
        return np.nan, np.nan, np.nan, mask
    X = np.log(kk)
    Y = np.log(yy)
    b, a = np.polyfit(X, Y, 1)   # Y ~ b X + a
    Yhat = b*X + a
    ss_res = np.sum((Y - Yhat)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2) + 1e-18
    r2 = 1.0 - ss_res/ss_tot
    return b, a, r2, mask

def analyze_decay_for_indices(
    data_dir: str,
    indices,
    t1: float,
    load_solution_fn,
    k_fit_window=(8, 128),
    mode="energy",          # "energy" fits |uhat|^2; "amplitude" fits |uhat|
    max_plot=6,
):
    slopes = []
    r2s = []
    examples = []

    for idx in indices:
        t, x, U = load_solution_fn(data_dir, int(idx))
        ti = int(np.argmin(np.abs(t - float(t1))))
        u = U[ti, :]

        k, amp, eng = rfft_spectrum(u)
        y = eng if mode == "energy" else amp

        b, a, r2, mask = fit_power_law(k, y, k_fit_window[0], min(k_fit_window[1], k[-1]))
        slopes.append(b)
        r2s.append(r2)

        if len(examples) < max_plot:
            examples.append((int(idx), float(t[ti]), k, y, b, a, r2, mask))

    slopes = np.asarray(slopes, dtype=float)
    r2s = np.asarray(r2s, dtype=float)

    # Plot a few examples
    for (idx, t_actual, k, y, b, a, r2, mask) in examples:
        plt.figure()
        plt.loglog(k, y, alpha=0.8, label=f"spectrum ({mode})")
        if np.isfinite(b):
            kfit = k[mask]
            yfit = np.exp(a) * (kfit**b)
            plt.loglog(kfit, yfit, lw=2, label=f"fit: slope={b:.3f}, R2={r2:.3f}")
        plt.xlabel("k")
        plt.ylabel("|û_k|^2" if mode=="energy" else "|û_k|")
        plt.title(f"idx={idx}, t={t_actual:.4f}, fit window={k_fit_window}")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.show()

    return {
        "slopes": slopes,
        "r2": r2s,
        "mode": mode,
        "k_fit_window": k_fit_window,
    }

import numpy as np
from matplotlib import pyplot as plt

def rfft_spectrum(u: np.ndarray):
    """Return k indices (1..Kmax) and amplitude/energy spectra excluding k=0."""
    u = np.asarray(u, dtype=np.float64)
    uhat = np.fft.rfft(u, norm="ortho")
    amp = np.abs(uhat)
    eng = amp**2
    k = np.arange(len(uhat))  # includes k=0
    return k[1:], amp[1:], eng[1:]  # drop k=0

def fit_power_law(k, y, kmin, kmax):
    """
    Fit log y = a + b log k on k in [kmin,kmax].
    Returns slope b, intercept a, R^2.
    """
    mask = (k >= kmin) & (k <= kmax) & np.isfinite(y) & (y > 0)
    kk = k[mask]
    yy = y[mask]
    if len(kk) < 5:
        return np.nan, np.nan, np.nan, mask
    X = np.log(kk)
    Y = np.log(yy)
    b, a = np.polyfit(X, Y, 1)   # Y ~ b X + a
    Yhat = b*X + a
    ss_res = np.sum((Y - Yhat)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2) + 1e-18
    r2 = 1.0 - ss_res/ss_tot
    return b, a, r2, mask

def analyze_decay_for_indices(
    data_dir: str,
    indices,
    t1: float,
    load_solution_fn,
    k_fit_window=(8, 128),
    mode="energy",          # "energy" fits |uhat|^2; "amplitude" fits |uhat|
    max_plot=6,
):
    slopes = []
    r2s = []
    examples = []

    for idx in indices:
        t, x, U = load_solution_fn(data_dir, int(idx))
        ti = int(np.argmin(np.abs(t - float(t1))))
        u = U[ti, :]

        k, amp, eng = rfft_spectrum(u)
        y = eng if mode == "energy" else amp

        b, a, r2, mask = fit_power_law(k, y, k_fit_window[0], min(k_fit_window[1], k[-1]))
        slopes.append(b)
        r2s.append(r2)

        if len(examples) < max_plot:
            examples.append((int(idx), float(t[ti]), k, y, b, a, r2, mask))

    slopes = np.asarray(slopes, dtype=float)
    r2s = np.asarray(r2s, dtype=float)

    # Plot a few examples
    for (idx, t_actual, k, y, b, a, r2, mask) in examples:
        plt.figure()
        plt.loglog(k, y, alpha=0.8, label=f"spectrum ({mode})")
        if np.isfinite(b):
            kfit = k[mask]
            yfit = np.exp(a) * (kfit**b)
            plt.loglog(kfit, yfit, lw=2, label=f"fit: slope={b:.3f}, R2={r2:.3f}")
        plt.xlabel("k")
        plt.ylabel("|û_k|^2" if mode=="energy" else "|û_k|")
        plt.title(f"idx={idx}, t={t_actual:.4f}, fit window={k_fit_window}")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        plt.show()

    return {
        "slopes": slopes,
        "r2": r2s,
        "mode": mode,
        "k_fit_window": k_fit_window,
    }