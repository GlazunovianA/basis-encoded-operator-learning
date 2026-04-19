import os
import numpy as np

def ensure_dirs(save_dir: str, plot_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

def save_parameter(save_dir: str, idx: int, x: np.ndarray, u0: np.ndarray):
    # analog of (x,y,a) : here (x,u0)
    arr = np.column_stack([x, u0])
    np.save(os.path.join(save_dir, f"fem_parameter_{idx}.npy"), arr)

def save_parameter_coeffs(save_dir: str, idx: int, theta: np.ndarray, coeffs: np.ndarray):
    np.save(os.path.join(save_dir, f"fem_parameter_theta_{idx}.npy"), theta)
    np.save(os.path.join(save_dir, f"fem_parameter_coeffs_{idx}.npy"), coeffs)

def save_solution(save_dir: str, idx: int, t: np.ndarray, x: np.ndarray, U: np.ndarray):
    # analog of (x,y,u) : here (t,x,u) flattened
    T, X = np.meshgrid(t, x, indexing="ij")
    out = np.column_stack([T.ravel(), X.ravel(), U.ravel()])
    np.save(os.path.join(save_dir, f"fem_solution_{idx}.npy"), out)

def load_parameter(save_dir: str, idx: int):
    arr = np.load(os.path.join(save_dir, f"fem_parameter_{idx}.npy"))
    x = arr[:, 0]
    u0 = arr[:, 1]
    theta = np.load(os.path.join(save_dir, f"fem_parameter_theta_{idx}.npy"))
    coeffs = np.load(os.path.join(save_dir, f"fem_parameter_coeffs_{idx}.npy"))
    return x, u0, theta, coeffs

def load_solution(save_dir: str, idx: int):
    arr = np.load(os.path.join(save_dir, f"fem_solution_{idx}.npy"))
    # recover unique grids (assumes full tensor grid was saved)
    t = np.unique(arr[:, 0])
    x = np.unique(arr[:, 1])
    U = arr[:, 2].reshape(len(t), len(x))
    return t, x, U
