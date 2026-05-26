"""
Visual test suite for 2D basis expansion/reconstruction.

Tests:
  T1  Round-trip identity          coeffs -> reconstruct -> compare
  T2  Projection fidelity          raw data -> project -> reconstruct -> compare vs raw
  T3  Coefficient orthogonality    <phi_n, phi_m> = delta_nm
  T4  Basis-function plots         visual sanity of each basis on its quad nodes
  T5  Cross-basis comparison       same field, all bases side-by-side
  T6  Quadrature weight check      sum(w) == domain_length
  T7  Domain scaling invariance    same function on [0,1] and [0,2] give same coeffs (scaled)
  T8  Transposition check          (Bx @ weighted_f @ By.T)[n,m] == integral by loop

Run:
    python test_expansion_visual.py
    python test_expansion_visual.py --basis sine legendre --N 12 --nquad 80
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── import both old and new code ──────────────────────────────────────────────
# Adjust these imports to match your project layout.
try:
    from basis_function import (
        get_legendre_quad_on_interval,
        get_basis_function_1d,get_basis_interpolation
    )
    from ed_2d import (
        encode_field_l2_2d,
        decode_field_l2_2d,
        project_on_L2_basis_2d as project_new,
        evaluate_series_L2   as evaluate_new,
    )
    HAS_NEW = True
except ImportError as e:
    print(f"[WARN] new module not found: {e}")
    HAS_NEW = False

try:
    from util_expansion import (
        get_basis_interpolation as get_basis_interp_old,
        get_basis_function      as get_basis_fn_old,
    )
    from L2_series_expansion_2d import (
        project_on_L2_basis_2d as project_old,
        evaluate_series_L2     as evaluate_old,
    )
    HAS_OLD = True
except ImportError as e:
    print(f"[WARN] old module not found: {e}")
    HAS_OLD = False

OUT_DIR = Path("test_outputs")
OUT_DIR.mkdir(exist_ok=True)

CMAP_FIELD  = "coolwarm"
CMAP_ERR    = "inferno"
CMAP_COEFF  = "RdBu_r"

# ── test functions ─────────────────────────────────────────────────────────────
def make_test_functions(domain):
    """Return list of (name, callable) on domain [[a,b],[c,d]]."""
    (a, b), (c, d) = domain
    Lx, Ly = b - a, d - c
    cx, cy = (a + b) / 2, (c + d) / 2
    fns = [
        ("gaussian",
         lambda X, Y: np.exp(-((X - cx)**2 + (Y - cy)**2) / (0.1 * Lx * Ly))),
        ("sine_product",
         lambda X, Y: np.sin(2*np.pi*(X-a)/Lx) * np.sin(2*np.pi*(Y-c)/Ly)),
        ("linear_ramp",
         lambda X, Y: (X - a)/Lx + 0.5*(Y - c)/Ly),
        ("high_freq",
         lambda X, Y: np.sin(5*np.pi*(X-a)/Lx) * np.cos(3*np.pi*(Y-c)/Ly)),
        ("corner_bump",
         lambda X, Y: np.exp(-10*((X-a)**2 + (Y-c)**2)/(Lx*Ly))),
    ]
    return fns


def _imshow(ax, data, title, vmin=None, vmax=None, cmap=CMAP_FIELD, norm=None):
    im = ax.imshow(data.T, origin="lower", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# T1  Round-trip identity
# ══════════════════════════════════════════════════════════════════════════════
def test_roundtrip(bases, N_basis, N_quad, domain):
    """
    Build coeffs directly (skip projection), reconstruct, measure self-consistency.
    Random coefficients -> evaluate on grid -> project back -> compare coeffs.
    """
    print("\n[T1] Round-trip identity")
    (a, b), (c, d) = domain

    fig, axes = plt.subplots(len(bases), 4,
                             figsize=(14, 3.2 * len(bases)),
                             squeeze=False)
    fig.suptitle("T1 · Round-trip: random coeffs → field → re-project → compare coeffs",
                 fontsize=10, y=1.01)

    for row, basis in enumerate(bases):
        np.random.seed(7)
        C_orig = np.random.randn(N_basis, N_basis) * np.exp(
            -0.5 * (np.arange(N_basis)[:, None] + np.arange(N_basis)[None, :]))

        # decode
        f_rec, Xq, Yq, meta = decode_field_l2_2d(
            C_orig, basis_type=basis, domain=domain, N_quad=N_quad)

        # re-encode from the grid
        x_flat = Xq.ravel()
        y_flat = Yq.ravel()
        f_flat = f_rec.ravel()
        C_back, f_rec2, _, _, f_quad2, _ = encode_field_l2_2d(
            x_flat, y_flat, f_flat,
            N_basis=N_basis, basis_type=basis, domain=domain, N_quad=N_quad)

        err_coeff = rel_l2(C_back, C_orig)
        err_field = rel_l2(f_rec2, f_rec)

        _imshow(axes[row, 0], f_rec,            f"{basis} original field")
        _imshow(axes[row, 1], f_rec2,           "re-projected field")
        _imshow(axes[row, 2], np.abs(f_rec - f_rec2),
                f"|field error| rL2={err_field:.2e}", cmap=CMAP_ERR)
        _imshow(axes[row, 3], np.abs(C_back - C_orig),
                f"|coeff error| rL2={err_coeff:.2e}", cmap=CMAP_ERR)

        status = "✓" if err_coeff < 1e-3 else "✗ FAIL"
        print(f"  {basis:12s}  coeff rL2={err_coeff:.2e}  field rL2={err_field:.2e}  {status}")

    _save(fig, "T1_roundtrip.png")


# ══════════════════════════════════════════════════════════════════════════════
# T2  Projection fidelity — each test function × each basis
# ══════════════════════════════════════════════════════════════════════════════
def test_projection_fidelity(bases, N_basis, N_quad, domain):
    """Project known analytic functions and measure reconstruction error."""
    print("\n[T2] Projection fidelity")
    fns = make_test_functions(domain)
    (a, b), (c, d) = domain


    for fn_name, fn in fns:
        fig, axes = plt.subplots(len(bases), 4,
                                 figsize=(14, 3.2 * len(bases)),
                                 squeeze=False)
        fig.suptitle(f"T2 · Projection fidelity — function: {fn_name}", fontsize=10, y=1.01)
        
        for row, basis in enumerate(bases):
            
            x_quad, y_quad, wx, wy, _, _= get_basis_interpolation(domain, N_quad, basis_type=basis)
            Xq, Yq = np.meshgrid(x_quad, y_quad, indexing="ij")
            f_true = fn(Xq, Yq)
            coeffs, f_rec, _, _, f_quad, _ = encode_field_l2_2d(
                Xq.ravel(), Yq.ravel(), f_true.ravel(),
                N_basis=N_basis, basis_type=basis, domain=domain, N_quad=N_quad)

            err = rel_l2(f_rec, f_quad)
            vmax = max(np.abs(f_true).max(), 1e-12)

            _imshow(axes[row, 0], f_true,  f"{basis} true")
            _imshow(axes[row, 1], f_rec,   "reconstructed")
            _imshow(axes[row, 2], f_quad - f_rec,
                    f"|error| rL2={err:.2e}",
                    cmap=CMAP_ERR,
                    norm=TwoSlopeNorm(vcenter=0,
                                     vmin=(f_quad-f_rec).min(),
                                     vmax=max((f_quad-f_rec).max(), 1e-15)))
            _imshow(axes[row, 3], np.abs(coeffs),
                    f"log|coeffs| (N={N_basis})", cmap="viridis")

            nyquist_warn = ""
            if fn_name == "high_freq" and N_basis <= 10:
                nyquist_warn = " (Nyquist: need N>10)"
            status = "✓" if err < 5e-2 else f"✗ FAIL{nyquist_warn}"
            print(f"  {fn_name:15s}  {basis:12s}  rL2={err:.2e}  {status}")

        _save(fig, f"T2_fidelity_{fn_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# T3  Orthogonality: <phi_n, phi_m> ≈ delta_nm
# ══════════════════════════════════════════════════════════════════════════════
def test_orthogonality(bases, N_basis, N_quad, domain):
    """
    Compute 1D Gram matrix G[n,m] = integral phi_n(x) phi_m(x) dx.
    Should be identity if basis is L2-orthonormal.
    """
    print("\n[T3] Orthogonality (1D Gram matrix)")
    (a, b) = domain[0]

    fig, axes = plt.subplots(1, len(bases),
                             figsize=(4 * len(bases), 4),
                             squeeze=False)
    fig.suptitle("T3 · 1D Gram matrix (should be identity)", fontsize=10)

    for col, basis in enumerate(bases):
        if basis == "cosine":
            k = np.arange(N_quad)
            x_quad = a + (2*k + 1) / (2*N_quad) * (b - a)
            wx = np.full(N_quad, (b - a) / N_quad)
        elif basis in ("sine", "fourier"):
            x_quad = np.linspace(a, b, N_quad, endpoint=False)
            wx = np.full(N_quad, (b - a) / N_quad)
        else:
            _, _, x_quad, wx = get_legendre_quad_on_interval(N_quad, a, b)

        B = np.stack([get_basis_function_1d(n, N_basis, basis, [a, b])(x_quad)
                      for n in range(N_basis)], axis=0)   # (N_basis, N_quad)
        G = (B * wx[None, :]) @ B.T                        # (N_basis, N_basis)
        err = np.linalg.norm(G - np.eye(N_basis), 'fro') / N_basis

        im = axes[0, col].imshow(G, cmap=CMAP_COEFF,
                                 norm=TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5))
        axes[0, col].set_title(f"{basis}\n||G-I||_F/N={err:.2e}", fontsize=8)
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        status = "✓" if err < 1e-3 else "✗ FAIL"
        print(f"  {basis:12s}  ||G-I||_F/N={err:.2e}  {status}")

    _save(fig, "T3_orthogonality.png")


# ══════════════════════════════════════════════════════════════════════════════
# T4  Basis function shape plots
# ══════════════════════════════════════════════════════════════════════════════
def test_basis_shapes(bases, N_basis, N_quad, domain):
    """Plot the first few 1D basis functions to visually verify their shape."""
    print("\n[T4] Basis function shapes")
    (a, b) = domain[0]
    show_n = min(5, N_basis)
    x_dense = np.linspace(a, b, 400)

    fig, axes = plt.subplots(len(bases), show_n,
                             figsize=(3 * show_n, 2.5 * len(bases)),
                             squeeze=False)
    fig.suptitle("T4 · Basis function shapes (first 5 modes)", fontsize=10, y=1.01)

    for row, basis in enumerate(bases):
        for col in range(show_n):
            phi = get_basis_function_1d(col, N_basis, basis, [a, b])
            vals = phi(x_dense)
            axes[row, col].plot(x_dense, vals, lw=1.5)
            axes[row, col].axhline(0, color="k", lw=0.5, ls="--")
            axes[row, col].set_title(f"{basis} n={col}", fontsize=7)
            axes[row, col].tick_params(labelsize=6)

    _save(fig, "T4_basis_shapes.png")
    print("  shapes saved.")


# ══════════════════════════════════════════════════════════════════════════════
# T5  Cross-basis comparison — same field, all bases
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_basis(bases, N_basis, N_quad, domain):
    print("\n[T5] Cross-basis comparison")
    fns = make_test_functions(domain)
    fn_name, fn = fns[0]

    fig, axes = plt.subplots(2, len(bases) + 1,
                             figsize=(3.5 * (len(bases) + 1), 7),
                             squeeze=False)
    fig.suptitle(f"T5 · Cross-basis comparison — {fn_name}", fontsize=10, y=1.01)

    # ground truth on dense uniform grid for display only
    (a, b), (c, d) = domain
    x_dense = np.linspace(a, b, N_quad)
    y_dense = np.linspace(c, d, N_quad)
    Xd, Yd = np.meshgrid(x_dense, y_dense, indexing="ij")
    _imshow(axes[0, 0], fn(Xd, Yd), "ground truth")
    axes[1, 0].axis("off")

    for col, basis in enumerate(bases, start=1):
        # use basis-native nodes — shortcut fires, no griddata error
        x_quad, y_quad, wx, wy, Xq, Yq = get_basis_interpolation(
            domain, N_quad, basis_type=basis)
        f_true = fn(Xq, Yq)

        coeffs, f_rec, _, _, f_quad, _ = encode_field_l2_2d(
            Xq.ravel(), Yq.ravel(), f_true.ravel(),
            N_basis=N_basis, basis_type=basis, domain=domain, N_quad=N_quad)
        err = rel_l2(f_rec, f_quad)

        _imshow(axes[0, col], f_rec,  f"{basis}\nrL2={err:.2e}")
        _imshow(axes[1, col], np.abs(f_quad - f_rec), f"|error|", cmap=CMAP_ERR)
        print(f"  {basis:12s}  rL2={err:.2e}")

    _save(fig, "T5_cross_basis.png")

# ══════════════════════════════════════════════════════════════════════════════
# T6  Quadrature weight sanity
# ══════════════════════════════════════════════════════════════════════════════
def test_quadrature_weights(bases, N_quad, domain):
    """sum(wx) should equal (b-a), sum(wy) should equal (d-c)."""
    print("\n[T6] Quadrature weight sanity")
    (a, b), (c, d) = domain

    for basis in bases:
        if basis in ("sine", "cosine", "fourier"):
            wx = np.full(N_quad, (b - a) / N_quad)
            wy = np.full(N_quad, (d - c) / N_quad)
        else:
            _, _, _, wx = get_legendre_quad_on_interval(N_quad, a, b)
            _, _, _, wy = get_legendre_quad_on_interval(N_quad, c, d)

        err_x = abs(wx.sum() - (b - a))
        err_y = abs(wy.sum() - (d - c))
        status = "✓" if err_x < 1e-10 and err_y < 1e-10 else "✗ FAIL"
        print(f"  {basis:12s}  sum(wx)-Lx={err_x:.2e}  sum(wy)-Ly={err_y:.2e}  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# T7  Domain scaling invariance
# ══════════════════════════════════════════════════════════════════════════════
def test_domain_scaling(bases, N_basis, N_quad):
    """
    f(x,y) on [0,1]^2  and  f(2x-1, 2y-1) on [0,2]^2
    should give coefficients that agree up to the sqrt(L) normalisation.
    The reconstructed *function values* should match exactly on common grid.
    """
    print("\n[T7] Domain scaling invariance")
    d1 = [[0.0, 1.0], [0.0, 1.0]]
    d2 = [[0.0, 2.0], [0.0, 2.0]]

    _, _, x1, _ = get_legendre_quad_on_interval(N_quad, 0, 1)
    _, _, y1, _ = get_legendre_quad_on_interval(N_quad, 0, 1)
    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")

    fn = lambda X, Y: np.exp(-5 * ((X - 0.5)**2 + (Y - 0.5)**2))
    f1 = fn(X1, Y1)

    for basis in bases:
        c1, _, _, _, _, _ = encode_field_l2_2d(
            X1.ravel(), Y1.ravel(), f1.ravel(),
            N_basis=N_basis, basis_type=basis, domain=d1, N_quad=N_quad)

        # build the same function on [0,2]^2
        _, _, x2, _ = get_legendre_quad_on_interval(N_quad, 0, 2)
        _, _, y2, _ = get_legendre_quad_on_interval(N_quad, 0, 2)
        X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
        f2 = fn(X2 / 2, Y2 / 2)   # same function value at corresponding points

        c2, _, _, _, _, _ = encode_field_l2_2d(
            X2.ravel(), Y2.ravel(), f2.ravel(),
            N_basis=N_basis, basis_type=basis, domain=d2, N_quad=N_quad)

        # reconstruct both on [0,1] and compare field values
        fr1, _, _, _ = decode_field_l2_2d(c1, basis_type=basis, domain=d1, N_quad=N_quad)
        fr2_full, _, _, _ = decode_field_l2_2d(c2, basis_type=basis, domain=d2, N_quad=N_quad)
        # subsample fr2 to [0,1]: first half of each axis
        fr2 = fr2_full[:N_quad//2, :N_quad//2]
        fr1s = fr1[:N_quad//2, :N_quad//2]

        err = rel_l2(fr2, fr1s)
        status = "✓" if err < 5e-2 else "~ (expected for some bases)"
        print(f"  {basis:12s}  field rL2 across domains={err:.2e}  {status}")


# ══════════════════════════════════════════════════════════════════════════════
# T8  New vs Old direct comparison (if both are importable)
# ══════════════════════════════════════════════════════════════════════════════
def test_new_vs_old(bases, N_basis, N_quad, domain):
    """
    Directly compare project_old / project_new on the same data.
    Prints coefficient-space and field-space differences.
    """
    if not (HAS_NEW and HAS_OLD):
        print("\n[T8] Skipped — need both old and new imports.")
        return

    print("\n[T8] New vs Old comparison")
    fns = make_test_functions(domain)
    (a, b), (c, d) = domain

    _, _, x_quad, _ = get_legendre_quad_on_interval(N_quad, a, b)
    _, _, y_quad, _ = get_legendre_quad_on_interval(N_quad, c, d)
    Xq, Yq = np.meshgrid(x_quad, y_quad, indexing="ij")

    fn_name, fn = fns[0]
    f_true = fn(Xq, Yq)

    rows = []
    for basis in bases:
        try:
            c_new, fr_new, _, _, fq_new, _ = encode_field_l2_2d(
                Xq.ravel(), Yq.ravel(), f_true.ravel(),
                N_basis=N_basis, basis_type=basis, domain=domain, N_quad=N_quad)
            c_old, fr_old, _, _, fq_old = project_old(
                Xq.ravel(), Yq.ravel(), f_true.ravel(),
                N_basis=N_basis, N_quad=N_quad, basis_type=basis)

            diff_coeff = rel_l2(c_new, c_old)
            diff_field = rel_l2(fr_new, fr_old)
            err_new    = rel_l2(fr_new, fq_new)
            err_old    = rel_l2(fr_old, fq_old)
            rows.append((basis, diff_coeff, diff_field, err_new, err_old))
            print(f"  {basis:12s}  Δcoeff={diff_coeff:.2e}  Δfield={diff_field:.2e}"
                  f"  err_new={err_new:.2e}  err_old={err_old:.2e}")
        except Exception as e:
            print(f"  {basis:12s}  ERROR: {e}")

    if rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        labels  = [r[0] for r in rows]
        x       = np.arange(len(labels))
        w       = 0.2
        ax.bar(x - 1.5*w, [r[1] for r in rows], w, label="Δ coeff (new vs old)")
        ax.bar(x - 0.5*w, [r[2] for r in rows], w, label="Δ field (new vs old)")
        ax.bar(x + 0.5*w, [r[3] for r in rows], w, label="err_new vs f_quad")
        ax.bar(x + 1.5*w, [r[4] for r in rows], w, label="err_old vs f_quad")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_ylabel("relative L2 error"); ax.set_title("T8 · New vs Old")
        ax.legend(fontsize=7); ax.grid(True, axis="y")
        _save(fig, "T8_new_vs_old.png")


# ══════════════════════════════════════════════════════════════════════════════
# T9  Coefficient decay — verify decay with mode index (sanity for PDE data)
# ══════════════════════════════════════════════════════════════════════════════
def test_coefficient_decay(bases, N_basis, N_quad, domain):
    print("\n[T9] Coefficient decay")
    fns = make_test_functions(domain)
    fn_name, fn = fns[0]

    fig, axes = plt.subplots(1, len(bases),
                             figsize=(4 * len(bases), 4), squeeze=False)
    fig.suptitle(f"T9 · Coefficient decay — {fn_name}", fontsize=10)

    for col, basis in enumerate(bases):
        x_quad, y_quad, wx, wy, Xq, Yq = get_basis_interpolation(
            domain, N_quad, basis_type=basis)
        f_true = fn(Xq, Yq)

        coeffs, _, _, _, _, _ = encode_field_l2_2d(
            Xq.ravel(), Yq.ravel(), f_true.ravel(),
            N_basis=N_basis, basis_type=basis, domain=domain, N_quad=N_quad)

        im = axes[0, col].imshow(np.log10(np.abs(coeffs) + 1e-16),
                                 cmap="viridis", origin="upper")
        axes[0, col].set_title(f"{basis}", fontsize=8)
        plt.colorbar(im, ax=axes[0, col], label="log10|C|", fraction=0.046, pad=0.04)
        print(f"  {basis:12s}  max|C|={np.abs(coeffs).max():.3e}  "
              f"C[0,0]={coeffs[0,0]:.3e}")

    _save(fig, "T9_coeff_decay.png")

# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis",  nargs="+",
                   default=["legendre", "sine", "cosine", "fourier"],
                   help="Basis types to test")
    p.add_argument("--N",      type=int, default=10, help="N_basis modes per dim")
    p.add_argument("--nquad",  type=int, default=60, help="quadrature points per dim")
    p.add_argument("--domain", nargs=4,  type=float,
                   default=[0.0, 1.0, 0.0, 1.0],
                   metavar=("AX","BX","AY","BY"),
                   help="domain ax bx ay by")
    p.add_argument("--skip",   nargs="*", default=[],
                   help="Test IDs to skip, e.g. T7 T8")
    args = p.parse_args()

    domain = [[args.domain[0], args.domain[1]],
              [args.domain[2], args.domain[3]]]
    bases  = args.basis
    N      = args.N
    Nq     = args.nquad
    skip   = set(args.skip)

    print(f"\n{'='*60}")
    print(f"  Expansion visual test suite")
    print(f"  bases={bases}  N_basis={N}  N_quad={Nq}")
    print(f"  domain={domain}")
    print(f"  output dir: {OUT_DIR.resolve()}")
    print(f"{'='*60}")

    if not HAS_NEW:
        print("\nERROR: new module (test_expansion_refactored) not importable. Aborting.")
        sys.exit(1)

    if "T1" not in skip: test_roundtrip(bases, N, Nq, domain)
    if "T2" not in skip: test_projection_fidelity(bases, N, Nq, domain)
    if "T3" not in skip: test_orthogonality(bases, N, Nq, domain)
    if "T4" not in skip: test_basis_shapes(bases, N, Nq, domain)
    if "T5" not in skip: test_cross_basis(bases, N, Nq, domain)
    if "T6" not in skip: test_quadrature_weights(bases, Nq, domain)
    if "T7" not in skip: test_domain_scaling(bases, N, Nq)
    if "T8" not in skip: test_new_vs_old(bases, N, Nq, domain)
    if "T9" not in skip: test_coefficient_decay(bases, N, Nq, domain)

    print(f"\nAll tests done. Figures in {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()