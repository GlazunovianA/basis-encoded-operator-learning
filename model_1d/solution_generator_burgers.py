import time
import numpy as np
import matplotlib.pyplot as plt

from basis_1d import uniform_quad_on_01
from sampling_u0 import sample_u0_from_uniform_params
from solver_burgers import solve_viscous_burgers_periodic
from io_pipeline import (
    ensure_dirs, save_parameter, save_parameter_coeffs, save_solution, load_solution, load_parameter
)

def main():
    equation = "viscous_burgers_utx_1d"
    save_dir = f"data/{equation}/"
    plot_dir = f"data/{equation}_plot/"
    ensure_dirs(save_dir, plot_dir)

    # dataset config
    n_samples = 5000
    sample_offset = 5000 # write as automatic. 
    # get existing files in save_dir and set sample_offset accordingly.
    import os
    existing_files = os.listdir(save_dir)
    existing_indices = []
    for f in existing_files:
        if f.startswith("fem_solution_") and f.endswith(".npy"):
            idx = int(f.split("_")[2].split(".")[0])
            existing_indices.append(idx)
    if existing_indices:
        sample_offset = max(existing_indices) + 1
    print(f"Existing samples found with indices: {existing_indices}")   
    seed = 42
    rng = np.random.default_rng(seed)

    # grids
    N_x = 256
    x, _ = uniform_quad_on_01(N_x)   # uniform grid on [0,1)

    N_t = 128
    t_final = 1.0

    # basis + sampling
    basis = "fourier_periodic"
    N_basis = 17  # include constant; odd/even handled by indexing
    decay_type = "polynomial"
    decay_rate = 0.8

    # PDE
    nu = 1e-3

    start = time.time()
    for i in range(n_samples):
        idx = sample_offset + i

        theta, coeffs, u0 = sample_u0_from_uniform_params(
            rng=rng,
            basis=basis,
            N_basis=N_basis,
            x=x,
            decay_type=decay_type,
            decay_rate=decay_rate,
            mean_zero=True,
        )

        t, U = solve_viscous_burgers_periodic(
            u0=u0, x=x,
            t_final=t_final, N_t=N_t,
            nu=nu
        )

        save_parameter(save_dir, idx, x, u0) # x is spatial grid and u0 is initial condition on point range x. 
        save_parameter_coeffs(save_dir, idx, theta, coeffs) # theta is initial uniform sampling; coeffs are the actual coefficient with decay imposed. 
        save_solution(save_dir, idx, t, x, U) # t is time grid, x is spatial grid, U is solution on (t,x) grid.

        # load_solution_path = f"{save_dir}/solution_{idx}.npy"   
        # t, x_loaded, U_loaded = load_solution(load_solution_path)
        # x_param, u0, theta, coeffs = load_parameter(save_dir, idx)
        
        # Plots: u(t,x) heatmap + snapshots
        fig, ax = plt.subplots()
        im = ax.imshow(
            U,
            aspect="auto",
            origin="lower",
            extent=[x.min(), x.max(), t.min(), t.max()],
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"u(t,x), sample {idx}, nu={nu}")
        fig.colorbar(im, ax=ax)
        fig.savefig(f"{plot_dir}/u_tx_{idx}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(x, U[0], label="t=0")
        ax.plot(x, U[len(t)//2], label=f"t≈{t[len(t)//2]:.3f}")
        ax.plot(x, U[-1], label=f"t={t_final}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Snapshots, sample {idx}")
        ax.legend()
        fig.savefig(f"{plot_dir}/snapshots_{idx}.png", dpi=300)
        plt.close(fig)

    print(f"Generated {n_samples} samples in {time.time()-start:.2f}s")
    print(f"Saved to {save_dir} and plots to {plot_dir}")

if __name__ == "__main__":
    main()
