import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import time

from io_pipeline import load_solution
from trainer_stepper import train_model, rollout_stepper, nearest_time_index, rel_l2
from encoding_decoding import encode_output_scheme2_slice_fourier, decode_output_scheme2_slice_fourier

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data_root = here.parent / "data"
    data_dir = str(data_root / "viscous_burgers_utx_1d")
    cache_dir = str(data_root / "cache" / "burgers_1d")

    # -----------------------
    # rollout settings
    # -----------------------
    Kx = 64
    dt = 0.1
    t_start = 0.1
    t_end = 1.0
    n_steps = int(round((t_end - t_start) / dt))

    train_size = 900
    test_size = 100
    train_indices = np.arange(0, train_size)
    test_indices = np.arange(train_size, train_size + test_size)

    # -----------------------
    # train stepper
    # -----------------------
    out = train_model(
        data_dir=data_dir,
        cache_dir=cache_dir,
        task="stepper",
        train_indices=train_indices,
        test_indices=test_indices,
        Kx=Kx,

        dt=dt,
        stepper_mode="multi",            # <-- this is the "other option"
        t_window=(t_start, 0.9),         # train all steps in [0.1, 0.9] with fixed dt
        # alternatively: stepper_mode="single", t_from=t_start, t_to=t_start+dt

        include_time=False,              # start with False

        model_type="rf",
        shared_kernel=True,

        use_pca=False,
        pca_components=32,
        normalize_in=1.0,
        normalize_out=1.0,

        d_features=8192,
        lengthscale=0.5,
        lam=1e-4,

        t_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        device="cpu",
        force_rebuild=True,
    )
    print("one-step vec MSE (held-out pairs):", out["vec_mse"])

    # -----------------------
    # rollout on one test trajectory
    # -----------------------
    idx0 = int(test_indices[0])
    t_grid, x_grid, U = load_solution(data_dir, idx0)

    i0 = nearest_time_index(t_grid, t_start)
    c0, meta0 = encode_output_scheme2_slice_fourier(t_grid, x_grid, U, t_index=i0, Kx=Kx)

    C_roll = rollout_stepper(
        c0=c0,
        n_steps=n_steps,
        trained=out["trained"],
        pca=out["pca"],
        use_pca=False,
        normalize_in=1.0,
        normalize_out=1.0,
        device="cpu",
    )

    # compare at final time
    t_final = t_start + n_steps * dt
    i_final = nearest_time_index(t_grid, t_final)

    c_true_final, meta_final = encode_output_scheme2_slice_fourier(t_grid, x_grid, U, t_index=i_final, Kx=Kx)
    c_pred_final = C_roll[-1]

    u_true = U[i_final, :]
    u_pred = decode_output_scheme2_slice_fourier(c_pred_final, meta_final)

    print("final time actual:", float(t_grid[i_final]))
    print("final coeff rel:", float(np.linalg.norm(c_pred_final - c_true_final) / (np.linalg.norm(c_true_final) + 1e-12)))
    print("final func rel L2:", float(rel_l2(u_pred, u_true)))

    plt.plot(x_grid, u_true, label="true")
    plt.plot(x_grid, u_pred, label="rollout", alpha=0.8)
    plt.legend()
    plt.title(f"Rollout to t≈{t_grid[i_final]:.3f} (dt={dt}, steps={n_steps})")
    plt.show()

    # optional: error over time
    errs = []
    times = []
    for m in range(n_steps + 1):
        t_m = t_start + m * dt
        i_m = nearest_time_index(t_grid, t_m)
        c_true_m, meta_m = encode_output_scheme2_slice_fourier(t_grid, x_grid, U, t_index=i_m, Kx=Kx)
        u_true_m = U[i_m, :]
        u_pred_m = decode_output_scheme2_slice_fourier(C_roll[m], meta_m)
        errs.append(rel_l2(u_pred_m, u_true_m))
        times.append(float(t_grid[i_m]))

    plt.semilogy(times, errs, marker="o")
    plt.xlabel("t")
    plt.ylabel("rel L2 error")
    plt.title("Rollout error vs time")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()