# A simple operator learning framework

This repository contains research code for learning parameter-to-solution maps for PDE-type problems after encoding inputs and outputs into finite-dimensional coefficient representations. The codebase currently supports both 1D and 2D settings.

At a high level, the workflow is:

1. load raw simulation/FEM data,
2. encode inputs and outputs into basis coefficients,
3. optionally compress outputs further with PCA,
4. train a surrogate model in coefficient space,
5. decode predictions back to function space,
6. evaluate approximation and prediction errors,
7. visualize reconstructions and benchmark inference time.

The package is designed for controlled numerical experiments rather than as a general-purpose library. Many utilities are therefore exposed directly inside experiment scripts.

---

## Main functionalities

The codebase contains two main experiment pipelines.

### 1. 1D operator learning pipeline

The 1D pipeline is currently set up for the viscous Burgers problem and supports two learning tasks:

- **Step task**: learn a map from encoded input parameters to a solution snapshot \(u(t_1, x)\)
- **Continuous task**: learn a map from encoded input parameters to the full trajectory \(U(t, x)\)

Main features:
- Fourier encoding for step outputs
- Legendre-in-time × Fourier-in-space encoding for continuous outputs
- optional PCA compression on encoded outputs
- random feature regression and MLP baselines
- automatic dimension/rank selection for input truncation, output basis size, and PCA rank
- function-space and coefficient-space error evaluation
- trajectory visualization and reconstruction diagnostics

### 2. 2D operator learning pipeline

The 2D pipeline is designed for FEM-generated PDE data and learns maps of the form

\[
a(x,y) \mapsto u(x,y).
\]

Main features:
- loading scattered FEM parameter and solution fields
- projection of output fields onto 2D \(L^2\)-type basis expansions
- loading precomputed input coefficient tensors
- optional PCA on encoded outputs
- random feature regression and MLP baselines
- automatic output-rank and PCA-rank selection
- visualization of raw fields, projected targets, predictions, and error fields
- encoded-only and end-to-end timing benchmarks

---