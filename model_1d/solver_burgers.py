import numpy as np

def _flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * u

def solve_viscous_burgers_periodic(
    u0: np.ndarray,
    x: np.ndarray,
    t_final: float,
    N_t: int,
    nu: float,
    cfl_adv: float = 0.4,
    cfl_dif: float = 0.4,
):
    """
    u_t + (1/2 u^2)_x = nu u_xx on [0,1) with periodic BC.
    Returns t_grid (N_t,), U (N_t, N_x).
    """
    if nu < 0:
        raise ValueError("nu must be >= 0")

    N_x = len(x)
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx):
        raise ValueError("x must be uniform grid")

    t_grid = np.linspace(0.0, t_final, N_t)
    U = np.zeros((N_t, N_x), dtype=float)
    U[0] = u0.copy()

    u = u0.copy()
    t = 0.0
    out_k = 1

    def L(a): return np.roll(a, -1)
    def R(a): return np.roll(a,  1)

    while out_k < N_t:
        umax = float(np.max(np.abs(u)))
        dt_adv = np.inf if umax < 1e-12 else cfl_adv * dx / umax
        dt_dif = np.inf if nu == 0 else cfl_dif * dx * dx / (2.0 * nu)
        dt = min(dt_adv, dt_dif)

        t_target = float(t_grid[out_k])
        if t + dt > t_target:
            dt = t_target - t

        # Rusanov flux at i+1/2: left=u_i, right=u_{i+1}
        uL = u
        uR = L(u)
        a = np.maximum(np.abs(uL), np.abs(uR))
        F_half = 0.5 * (_flux(uL) + _flux(uR)) - 0.5 * a * (uR - uL)

        adv = (F_half - R(F_half)) / dx

        dif = nu * (L(u) - 2.0 * u + R(u)) / (dx * dx)

        u = u - dt * adv + dt * dif
        t += dt

        if abs(t - t_target) < 1e-13:
            U[out_k] = u
            out_k += 1

    return t_grid, U

