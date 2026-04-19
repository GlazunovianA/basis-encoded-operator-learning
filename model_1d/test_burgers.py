import numpy as np
from basis_1d import uniform_quad_on_01
from solver_burgers import solve_viscous_burgers_periodic
import os
def test_burgers_basic_sanity():
    N_x = 256
    x, _ = uniform_quad_on_01(N_x)
    u0 = np.sin(2*np.pi*x) + 0.25*np.cos(4*np.pi*x)
    nu = 1e-3

    t, U = solve_viscous_burgers_periodic(u0, x, t_final=0.5, N_t=64, nu=nu)
    print("Shape of solution:", U.shape)
    # plot solution
    assert np.isfinite(U).all(), "Non-finite values in solution"
    assert U.shape == (64, 256)

def test_burgers_mean_conservation():
    N_x = 256
    x, _ = uniform_quad_on_01(N_x)
    u0 = np.sin(2*np.pi*x) + 0.1
    nu = 1e-3

    t, U = solve_viscous_burgers_periodic(u0, x, t_final=1.0, N_t=128, nu=nu)
    print("Mean at initial time:", np.mean(U[0]))
    m0 = float(np.mean(U[0]))
    mT = float(np.mean(U[-1]))
    assert abs(mT - m0) < 5e-3, f"Mean not (approximately) conserved: {m0} vs {mT}"

def test_burgers_energy_not_exploding():
    N_x = 256
    x, _ = uniform_quad_on_01(N_x)
    u0 = 0.8*np.sin(2*np.pi*x) + 0.4*np.sin(6*np.pi*x)
    nu = 5e-4

    t, U = solve_viscous_burgers_periodic(u0, x, t_final=1.0, N_t=128, nu=nu)
    print("Energy at final time:", np.mean(U[-1]*U[-1]))
    E = np.mean(U*U, axis=1)
    # allow tiny numerical wiggles but forbid blow-up
    assert np.max(E) < 5.0*np.max(E[:5]), "Energy blew up suspiciously"

def test_burgers_basic_sanity():
    os.environ["BURGERS_PLOT"] = "1"  
    N_x = 256
    x, _ = uniform_quad_on_01(N_x)
    u0 = np.sin(2*np.pi*x) + 0.25*np.cos(4*np.pi*x)
    nu = 1e-3

    t, U = solve_viscous_burgers_periodic(u0, x, t_final=0.5, N_t=64, nu=nu)
    print("Shape of solution:", U.shape)

    # Assertions first: keep the test deterministic
    assert np.isfinite(U).all(), "Non-finite values in solution"
    assert U.shape == (64, 256)

    # Optional plotting: enable locally with BURGERS_PLOT=1
    if os.environ.get("BURGERS_PLOT", "0") == "1":
        import matplotlib
        matplotlib.use("Agg")  # non-interactive; safe for CI too
        import matplotlib.pyplot as plt

        outdir = os.environ.get("BURGERS_PLOT_DIR", "plots")
        os.makedirs(outdir, exist_ok=True)

        # (1) Final-time line plot
        plt.figure()
        plt.plot(x, u0, label="u0")
        plt.plot(x, U[-1], label=f"u(t={t[-1]:.3f})")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Viscous Burgers (nu={nu:g})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "burgers_final.png"), dpi=200)
        plt.close()

        # (2) Space–time heatmap (t on y-axis, x on x-axis)
        plt.figure()
        plt.imshow(
            U,
            aspect="auto",
            origin="lower",
            extent=[x[0], x[-1], t[0], t[-1]],
        )
        plt.xlabel("x")
        plt.ylabel("t")
        plt.title("u(x,t) heatmap")
        plt.colorbar(label="u")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "burgers_heatmap.png"), dpi=200)

test_burgers_basic_sanity()
test_burgers_mean_conservation()
test_burgers_energy_not_exploding()
test_burgers_basic_sanity()
print("All Burgers tests passed!")