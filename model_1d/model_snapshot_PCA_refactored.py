import os
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from io_pipeline import load_solution, load_parameter


# ============================================================
# 0) Small helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def weighted_spatial_rel_l2(
    predicted_snapshot_values: np.ndarray,
    true_snapshot_values: np.ndarray,
    space_grid: np.ndarray,
) -> float:
    """
    Relative L2 error on a single spatial snapshot u(x) at fixed time.
    """
    predicted_snapshot_values = np.asarray(predicted_snapshot_values, dtype=float).reshape(-1)
    true_snapshot_values = np.asarray(true_snapshot_values, dtype=float).reshape(-1)
    space_grid = np.asarray(space_grid, dtype=float).reshape(-1)

    if predicted_snapshot_values.shape != true_snapshot_values.shape:
        raise ValueError(
            f"predicted_snapshot_values shape {predicted_snapshot_values.shape} "
            f"!= true_snapshot_values shape {true_snapshot_values.shape}"
        )
    if predicted_snapshot_values.shape[0] != space_grid.shape[0]:
        raise ValueError(
            f"snapshot length {predicted_snapshot_values.shape[0]} "
            f"!= space_grid length {space_grid.shape[0]}"
        )

    spatial_weights = np.full(space_grid.shape[0], 1.0 / space_grid.shape[0], dtype=float)
    error_energy = np.sum(spatial_weights * (predicted_snapshot_values - true_snapshot_values) ** 2)
    reference_energy = np.sum(spatial_weights * true_snapshot_values ** 2) + 1e-12
    return float(np.sqrt(error_energy / reference_energy))


def default_encode_snapshot_output(
    space_grid: np.ndarray,
    snapshot_values: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Default high-fidelity output representation:
    use raw spatial snapshot values directly.

    This is intentionally simple and avoids introducing another
    output representation layer in this new route.
    """
    space_grid = np.asarray(space_grid, dtype=float).reshape(-1)
    snapshot_values = np.asarray(snapshot_values, dtype=float).reshape(-1)

    if snapshot_values.shape[0] != space_grid.shape[0]:
        raise ValueError(
            f"snapshot_values length {snapshot_values.shape[0]} != space_grid length {space_grid.shape[0]}"
        )

    return snapshot_values.copy(), {
        "representation_type": "raw_spatial_snapshot_values",
        "output_dimension": int(snapshot_values.shape[0]),
    }


# ============================================================
# 1) Dataset building: (coeffs, t) -> snapshot
# ============================================================

def encode_input_function(
    data_dir: str,
    sample_index: int,
) -> np.ndarray:
    """
    Input source of truth reused from the old code:
    load_parameter(...)[3] = coeffs
    """
    _, _, _, input_coefficient_vector = load_parameter(data_dir, int(sample_index))
    return np.asarray(input_coefficient_vector, dtype=float).reshape(-1)


def build_snapshot_dataset(
    data_dir: str,
    sample_indices: Sequence[int],
    snapshot_time_indices: Sequence[int] | None = None,
    max_snapshots_per_trajectory: int | None = None,
    encode_snapshot_output=default_encode_snapshot_output,
) -> dict:
    """
    Build the snapshot-route dataset:

        (input_coefficients, query_time) -> encoded_snapshot_output

    Returns a single connected bundle used by PCA and regression.
    """
    collected_input_feature_vectors: list[np.ndarray] = []
    collected_query_time_values: list[float] = []
    collected_snapshot_output_vectors: list[np.ndarray] = []
    collected_sample_indices: list[int] = []
    collected_time_indices: list[int] = []
    collected_time_values: list[float] = []

    reference_space_grid = None
    reference_time_grid = None
    output_representation_metadata = None

    for sample_index in sample_indices:
        input_feature_vector = encode_input_function(data_dir, int(sample_index))
        time_grid, space_grid, solution_values = load_solution(data_dir, int(sample_index))

        time_grid = np.asarray(time_grid, dtype=float)
        space_grid = np.asarray(space_grid, dtype=float)
        solution_values = np.asarray(solution_values, dtype=float)

        if solution_values.shape != (time_grid.shape[0], space_grid.shape[0]):
            raise ValueError(
                f"solution_values shape {solution_values.shape} does not match "
                f"(len(time_grid), len(space_grid)) = ({time_grid.shape[0]}, {space_grid.shape[0]})"
            )

        if reference_space_grid is None:
            reference_space_grid = space_grid.copy()
            reference_time_grid = time_grid.copy()
        else:
            if reference_space_grid.shape != space_grid.shape or not np.allclose(reference_space_grid, space_grid):
                raise ValueError("All trajectories must share the same space_grid.")
            if reference_time_grid.shape != time_grid.shape or not np.allclose(reference_time_grid, time_grid):
                raise ValueError("All trajectories must share the same time_grid for this route.")

        if snapshot_time_indices is None:
            selected_time_index_list = list(range(time_grid.shape[0]))
        else:
            selected_time_index_list = [int(time_index) for time_index in snapshot_time_indices]

        if max_snapshots_per_trajectory is not None:
            selected_time_index_list = selected_time_index_list[: int(max_snapshots_per_trajectory)]

        for time_index in selected_time_index_list:
            if not (0 <= time_index < time_grid.shape[0]):
                raise ValueError(
                    f"time_index={time_index} is out of bounds for time_grid length {time_grid.shape[0]}"
                )

            snapshot_values = solution_values[time_index, :]
            snapshot_output_vector, current_output_representation_metadata = encode_snapshot_output(
                space_grid,
                snapshot_values,
            )

            if output_representation_metadata is None:
                output_representation_metadata = dict(current_output_representation_metadata)
            else:
                if output_representation_metadata != current_output_representation_metadata:
                    raise ValueError("Output representation metadata changed across samples.")

            collected_input_feature_vectors.append(input_feature_vector.copy())
            collected_query_time_values.append(float(time_grid[time_index]))
            collected_snapshot_output_vectors.append(np.asarray(snapshot_output_vector, dtype=float).reshape(-1))
            collected_sample_indices.append(int(sample_index))
            collected_time_indices.append(int(time_index))
            collected_time_values.append(float(time_grid[time_index]))

    if len(collected_snapshot_output_vectors) == 0:
        raise ValueError("No snapshots were collected.")

    input_function_feature_matrix = np.stack(collected_input_feature_vectors, axis=0)
    query_time_vector = np.asarray(collected_query_time_values, dtype=float).reshape(-1, 1)
    snapshot_output_matrix = np.stack(collected_snapshot_output_vectors, axis=0)

    return {
        "input_function_feature_matrix": input_function_feature_matrix,
        "query_time_vector": query_time_vector,
        "snapshot_output_matrix": snapshot_output_matrix,
        "snapshot_sample_index_vector": np.asarray(collected_sample_indices, dtype=int),
        "snapshot_time_index_vector": np.asarray(collected_time_indices, dtype=int),
        "snapshot_time_value_vector": np.asarray(collected_time_values, dtype=float),
        "space_grid": np.asarray(reference_space_grid, dtype=float),
        "time_grid": np.asarray(reference_time_grid, dtype=float),
        "output_representation_metadata": dict(output_representation_metadata),
    }


# ============================================================
# 2) Output PCA only
# ============================================================

def fit_spatial_pca_basis(
    training_snapshot_dataset_bundle: dict,
    max_output_pca_rank: int | None = None,
) -> dict:
    """
    Fit PCA on training snapshot outputs only.
    """
    snapshot_output_matrix = np.asarray(training_snapshot_dataset_bundle["snapshot_output_matrix"], dtype=float)
    num_snapshots, output_dimension = snapshot_output_matrix.shape

    if max_output_pca_rank is None:
        max_output_pca_rank = min(num_snapshots, output_dimension)
    else:
        max_output_pca_rank = int(min(max_output_pca_rank, num_snapshots, output_dimension))

    output_pca_model = PCA(n_components=max_output_pca_rank)
    output_pca_model.fit(snapshot_output_matrix)

    explained_variance_ratio_vector = np.asarray(output_pca_model.explained_variance_ratio_, dtype=float)
    cumulative_explained_variance_ratio_vector = np.cumsum(explained_variance_ratio_vector)

    return {
        "output_pca_model": output_pca_model,
        "output_pca_mean_vector": np.asarray(output_pca_model.mean_, dtype=float),
        "output_pca_component_matrix": np.asarray(output_pca_model.components_, dtype=float),
        "explained_variance_ratio_vector": explained_variance_ratio_vector,
        "cumulative_explained_variance_ratio_vector": cumulative_explained_variance_ratio_vector,
    }


def project_snapshots_to_pca_coefficients(
    snapshot_dataset_bundle: dict,
    spatial_pca_bundle: dict,
    retained_output_pca_rank: int,
) -> dict:
    """
    Build regression inputs and regression targets.

    Regression input:
        [input_function_features | query_time]

    Regression target:
        first retained_output_pca_rank output PCA coefficients
    """
    retained_output_pca_rank = int(retained_output_pca_rank)
    output_pca_model = spatial_pca_bundle["output_pca_model"]

    if not (1 <= retained_output_pca_rank <= output_pca_model.components_.shape[0]):
        raise ValueError(
            f"retained_output_pca_rank={retained_output_pca_rank} is invalid for "
            f"{output_pca_model.components_.shape[0]} fitted PCA components"
        )

    full_output_pca_coefficient_matrix = output_pca_model.transform(
        np.asarray(snapshot_dataset_bundle["snapshot_output_matrix"], dtype=float)
    )
    retained_output_pca_coefficient_matrix = full_output_pca_coefficient_matrix[:, :retained_output_pca_rank]

    regressor_input_matrix = np.concatenate(
        [
            np.asarray(snapshot_dataset_bundle["input_function_feature_matrix"], dtype=float),
            np.asarray(snapshot_dataset_bundle["query_time_vector"], dtype=float),
        ],
        axis=1,
    )

    return {
        "regressor_input_matrix": regressor_input_matrix,
        "regressor_target_matrix": np.asarray(retained_output_pca_coefficient_matrix, dtype=float),
        "snapshot_dataset_bundle": snapshot_dataset_bundle,
        "spatial_pca_bundle": spatial_pca_bundle,
        "retained_output_pca_rank": retained_output_pca_rank,
    }


# ============================================================
# 3) RFF ridge regressor
# ============================================================

def standardize_train_test_inputs(
    training_input_matrix: np.ndarray,
    test_input_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Standardize regression inputs using training-set statistics only.
    """
    training_input_matrix = np.asarray(training_input_matrix, dtype=float)
    test_input_matrix = np.asarray(test_input_matrix, dtype=float)

    input_mean_vector = np.mean(training_input_matrix, axis=0)
    input_std_vector = np.std(training_input_matrix, axis=0)
    input_std_vector = np.where(input_std_vector < 1e-12, 1.0, input_std_vector)

    standardized_training_input_matrix = (training_input_matrix - input_mean_vector[None, :]) / input_std_vector[None, :]
    standardized_test_input_matrix = (test_input_matrix - input_mean_vector[None, :]) / input_std_vector[None, :]

    return standardized_training_input_matrix, standardized_test_input_matrix, {
        "input_mean_vector": input_mean_vector,
        "input_std_vector": input_std_vector,
    }


def build_rff_feature_matrix(
    input_matrix: np.ndarray,
    num_random_features: int,
    lengthscale: float,
    random_seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Real random Fourier features for a Gaussian kernel.
    """
    input_matrix = np.asarray(input_matrix, dtype=float)
    if input_matrix.ndim != 2:
        raise ValueError("input_matrix must be 2D")

    num_samples, input_dimension = input_matrix.shape
    rng = np.random.default_rng(random_seed)

    random_frequency_matrix = rng.normal(
        loc=0.0,
        scale=1.0 / max(float(lengthscale), 1e-12),
        size=(input_dimension, num_random_features),
    )
    random_phase_vector = rng.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=(num_random_features,),
    )

    projected_matrix = input_matrix @ random_frequency_matrix + random_phase_vector[None, :]
    feature_matrix = np.sqrt(2.0 / num_random_features) * np.cos(projected_matrix)

    return feature_matrix, {
        "random_frequency_matrix": random_frequency_matrix,
        "random_phase_vector": random_phase_vector,
        "num_random_features": int(num_random_features),
        "lengthscale": float(lengthscale),
        "random_seed": int(random_seed),
    }


def apply_rff_feature_matrix(
    input_matrix: np.ndarray,
    feature_metadata: dict,
) -> np.ndarray:
    """
    Apply stored RFF parameters to new inputs.
    """
    input_matrix = np.asarray(input_matrix, dtype=float)
    random_frequency_matrix = np.asarray(feature_metadata["random_frequency_matrix"], dtype=float)
    random_phase_vector = np.asarray(feature_metadata["random_phase_vector"], dtype=float)

    projected_matrix = input_matrix @ random_frequency_matrix + random_phase_vector[None, :]
    feature_matrix = np.sqrt(2.0 / random_frequency_matrix.shape[1]) * np.cos(projected_matrix)
    return feature_matrix


def ridge_fit_multioutput(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    """
    Solve W = argmin ||Phi W - Y||_F^2 + lambda ||W||_F^2
    """
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    target_matrix = np.asarray(target_matrix, dtype=float)

    if feature_matrix.ndim != 2 or target_matrix.ndim != 2:
        raise ValueError("feature_matrix and target_matrix must both be 2D arrays")
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise ValueError(
            f"feature_matrix rows {feature_matrix.shape[0]} != target_matrix rows {target_matrix.shape[0]}"
        )

    num_features = feature_matrix.shape[1]
    identity_matrix = np.eye(num_features, dtype=float)

    gram_matrix = feature_matrix.T @ feature_matrix
    rhs_matrix = feature_matrix.T @ target_matrix

    weight_matrix = np.linalg.solve(gram_matrix + float(ridge_lambda) * identity_matrix, rhs_matrix)
    return weight_matrix


def fit_snapshot_rff_regressor(
    projected_training_snapshot_dataset_bundle: dict,
    num_random_features: int = 4096,
    lengthscale: float = 1.0,
    ridge_lambda: float = 1e-6,
    random_seed: int = 0,
) -> dict:
    """
    Fit:
        (input_function_coeffs, query_time) -> output PCA coefficients
    """
    training_input_matrix = np.asarray(
        projected_training_snapshot_dataset_bundle["regressor_input_matrix"],
        dtype=float,
    )
    training_target_matrix = np.asarray(
        projected_training_snapshot_dataset_bundle["regressor_target_matrix"],
        dtype=float,
    )

    training_feature_matrix, feature_metadata = build_rff_feature_matrix(
        input_matrix=training_input_matrix,
        num_random_features=num_random_features,
        lengthscale=lengthscale,
        random_seed=random_seed,
    )

    weight_matrix = ridge_fit_multioutput(
        feature_matrix=training_feature_matrix,
        target_matrix=training_target_matrix,
        ridge_lambda=ridge_lambda,
    )

    return {
        "weight_matrix": weight_matrix,
        "feature_metadata": feature_metadata,
        "ridge_lambda": float(ridge_lambda),
        "num_random_features": int(num_random_features),
        "lengthscale": float(lengthscale),
    }


# ============================================================
# 4) Evaluation
# ============================================================

def evaluate_snapshot_regressor(
    projected_test_snapshot_dataset_bundle: dict,
    trained_snapshot_rff_regressor_bundle: dict,
) -> dict:
    """
    Evaluate two errors for each test snapshot:

    1) PCA truncation error:
       true snapshot vs reconstruction from true retained PCA coefficients

    2) Prediction error:
       true snapshot vs reconstruction from predicted retained PCA coefficients
    """
    snapshot_dataset_bundle = projected_test_snapshot_dataset_bundle["snapshot_dataset_bundle"]
    spatial_pca_bundle = projected_test_snapshot_dataset_bundle["spatial_pca_bundle"]
    retained_output_pca_rank = int(projected_test_snapshot_dataset_bundle["retained_output_pca_rank"])

    output_pca_model = spatial_pca_bundle["output_pca_model"]
    true_snapshot_output_matrix = np.asarray(snapshot_dataset_bundle["snapshot_output_matrix"], dtype=float)
    regressor_input_matrix = np.asarray(projected_test_snapshot_dataset_bundle["regressor_input_matrix"], dtype=float)
    true_retained_output_pca_coefficient_matrix = np.asarray(
        projected_test_snapshot_dataset_bundle["regressor_target_matrix"],
        dtype=float,
    )

    test_feature_matrix = apply_rff_feature_matrix(
        input_matrix=regressor_input_matrix,
        feature_metadata=trained_snapshot_rff_regressor_bundle["feature_metadata"],
    )
    predicted_retained_output_pca_coefficient_matrix = test_feature_matrix @ np.asarray(
        trained_snapshot_rff_regressor_bundle["weight_matrix"],
        dtype=float,
    )

    output_pca_mean_vector = np.asarray(output_pca_model.mean_, dtype=float)
    retained_output_pca_component_matrix = np.asarray(
        output_pca_model.components_[:retained_output_pca_rank, :],
        dtype=float,
    )

    space_grid = np.asarray(snapshot_dataset_bundle["space_grid"], dtype=float)

    pca_truncation_error_vector = []
    prediction_error_vector = []

    for snapshot_row_index in range(true_snapshot_output_matrix.shape[0]):
        true_snapshot_output_vector = true_snapshot_output_matrix[snapshot_row_index]

        true_retained_output_pca_coefficient_vector = true_retained_output_pca_coefficient_matrix[snapshot_row_index]
        predicted_retained_output_pca_coefficient_vector = predicted_retained_output_pca_coefficient_matrix[snapshot_row_index]

        pca_reconstructed_output_vector = (
            output_pca_mean_vector
            + true_retained_output_pca_coefficient_vector @ retained_output_pca_component_matrix
        )
        predicted_output_vector = (
            output_pca_mean_vector
            + predicted_retained_output_pca_coefficient_vector @ retained_output_pca_component_matrix
        )

        pca_truncation_error = weighted_spatial_rel_l2(
            predicted_snapshot_values=pca_reconstructed_output_vector,
            true_snapshot_values=true_snapshot_output_vector,
            space_grid=space_grid,
        )
        prediction_error = weighted_spatial_rel_l2(
            predicted_snapshot_values=predicted_output_vector,
            true_snapshot_values=true_snapshot_output_vector,
            space_grid=space_grid,
        )

        pca_truncation_error_vector.append(pca_truncation_error)
        prediction_error_vector.append(prediction_error)

    pca_truncation_error_vector = np.asarray(pca_truncation_error_vector, dtype=float)
    prediction_error_vector = np.asarray(prediction_error_vector, dtype=float)

    return {
        "num_test_snapshots": int(true_snapshot_output_matrix.shape[0]),
        "retained_output_pca_rank": retained_output_pca_rank,
        "pca_truncation_error_vector": pca_truncation_error_vector,
        "prediction_error_vector": prediction_error_vector,
        "mean_pca_truncation_error": float(np.mean(pca_truncation_error_vector)),
        "mean_prediction_error": float(np.mean(prediction_error_vector)),
        "median_pca_truncation_error": float(np.median(pca_truncation_error_vector)),
        "median_prediction_error": float(np.median(prediction_error_vector)),
        "max_pca_truncation_error": float(np.max(pca_truncation_error_vector)),
        "max_prediction_error": float(np.max(prediction_error_vector)),
    }


# ============================================================
# 5) PCA rank sweep and plots
# ============================================================

def plot_output_pca_residual_energy(
    spatial_pca_bundle: dict,
) -> None:
    explained_variance_ratio_vector = np.asarray(
        spatial_pca_bundle["explained_variance_ratio_vector"],
        dtype=float,
    )
    cumulative_explained_variance_ratio_vector = np.cumsum(explained_variance_ratio_vector)
    residual_energy_vector = 1.0 - cumulative_explained_variance_ratio_vector

    plt.figure()
    plt.semilogy(
        np.arange(1, residual_energy_vector.shape[0] + 1),
        residual_energy_vector,
        marker="o",
    )
    plt.xlabel("retained output PCA rank")
    plt.ylabel("1 - cumulative explained variance")
    plt.title("Output PCA residual energy")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()


def run_snapshot_pca_rank_sweep(
    training_snapshot_dataset_bundle: dict,
    test_snapshot_dataset_bundle: dict,
    output_pca_rank_list: Sequence[int],
    num_random_features: int = 4096,
    lengthscale: float = 1.0,
    ridge_lambda: float = 1e-6,
    random_seed: int = 0,
) -> dict:
    """
    Full connected experiment over multiple output PCA ranks.
    """
    max_output_pca_rank = int(max(output_pca_rank_list))
    spatial_pca_bundle = fit_spatial_pca_basis(
        training_snapshot_dataset_bundle=training_snapshot_dataset_bundle,
        max_output_pca_rank=max_output_pca_rank,
    )

    evaluation_rows = []

    for retained_output_pca_rank in output_pca_rank_list:
        projected_training_snapshot_dataset_bundle = project_snapshots_to_pca_coefficients(
            snapshot_dataset_bundle=training_snapshot_dataset_bundle,
            spatial_pca_bundle=spatial_pca_bundle,
            retained_output_pca_rank=retained_output_pca_rank,
        )
        projected_test_snapshot_dataset_bundle = project_snapshots_to_pca_coefficients(
            snapshot_dataset_bundle=test_snapshot_dataset_bundle,
            spatial_pca_bundle=spatial_pca_bundle,
            retained_output_pca_rank=retained_output_pca_rank,
        )

        trained_snapshot_rff_regressor_bundle = fit_snapshot_rff_regressor(
            projected_training_snapshot_dataset_bundle=projected_training_snapshot_dataset_bundle,
            num_random_features=num_random_features,
            lengthscale=lengthscale,
            ridge_lambda=ridge_lambda,
            random_seed=random_seed,
        )

        evaluation_bundle = evaluate_snapshot_regressor(
            projected_test_snapshot_dataset_bundle=projected_test_snapshot_dataset_bundle,
            trained_snapshot_rff_regressor_bundle=trained_snapshot_rff_regressor_bundle,
        )

        evaluation_rows.append(
            {
                "retained_output_pca_rank": int(retained_output_pca_rank),
                "mean_pca_truncation_error": evaluation_bundle["mean_pca_truncation_error"],
                "mean_prediction_error": evaluation_bundle["mean_prediction_error"],
                "median_pca_truncation_error": evaluation_bundle["median_pca_truncation_error"],
                "median_prediction_error": evaluation_bundle["median_prediction_error"],
                "max_pca_truncation_error": evaluation_bundle["max_pca_truncation_error"],
                "max_prediction_error": evaluation_bundle["max_prediction_error"],
            }
        )

        print(
            f"rank={retained_output_pca_rank:4d} | "
            f"mean_pca_trunc={evaluation_bundle['mean_pca_truncation_error']:.4e} | "
            f"mean_pred={evaluation_bundle['mean_prediction_error']:.4e}"
        )

    return {
        "spatial_pca_bundle": spatial_pca_bundle,
        "evaluation_rows": evaluation_rows,
    }


def plot_snapshot_pca_rank_sweep(
    rank_sweep_result_bundle: dict,
) -> None:
    evaluation_rows = rank_sweep_result_bundle["evaluation_rows"]

    retained_output_pca_rank_vector = np.asarray(
        [row["retained_output_pca_rank"] for row in evaluation_rows],
        dtype=int,
    )
    mean_pca_truncation_error_vector = np.asarray(
        [row["mean_pca_truncation_error"] for row in evaluation_rows],
        dtype=float,
    )
    mean_prediction_error_vector = np.asarray(
        [row["mean_prediction_error"] for row in evaluation_rows],
        dtype=float,
    )

    plt.figure()
    plt.plot(
        retained_output_pca_rank_vector,
        mean_pca_truncation_error_vector,
        marker="o",
        label="mean PCA truncation error",
    )
    plt.plot(
        retained_output_pca_rank_vector,
        mean_prediction_error_vector,
        marker="o",
        label="mean prediction error",
    )
    plt.xlabel("retained output PCA rank")
    plt.ylabel("relative L2 error on snapshot")
    plt.title("Snapshot PCA rank vs reconstruction / prediction error")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.show()


# ============================================================
# 6) Example main
# ============================================================

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data_root = here.parent / "data"
    data_dir = str(data_root / "viscous_burgers_utx_1d")

    train_size = 9000
    test_size = 1000
    train_indices = np.arange(0, train_size)
    test_indices = np.arange(train_size, train_size + test_size)

    # You can change this to subsample time snapshots if needed.
    snapshot_time_indices = None
    max_snapshots_per_trajectory = None

    print("Building training snapshot dataset...")
    training_snapshot_dataset_bundle = build_snapshot_dataset(
        data_dir=data_dir,
        sample_indices=train_indices,
        snapshot_time_indices=snapshot_time_indices,
        max_snapshots_per_trajectory=max_snapshots_per_trajectory,
    )

    print("Building test snapshot dataset...")
    test_snapshot_dataset_bundle = build_snapshot_dataset(
        data_dir=data_dir,
        sample_indices=test_indices,
        snapshot_time_indices=snapshot_time_indices,
        max_snapshots_per_trajectory=max_snapshots_per_trajectory,
    )

    print("Training snapshot dataset shapes:")
    print("input_function_feature_matrix:", training_snapshot_dataset_bundle["input_function_feature_matrix"].shape)
    print("query_time_vector:", training_snapshot_dataset_bundle["query_time_vector"].shape)
    print("snapshot_output_matrix:", training_snapshot_dataset_bundle["snapshot_output_matrix"].shape)

    output_pca_rank_list = [8, 16, 32, 64, 96, 128]

    rank_sweep_result_bundle = run_snapshot_pca_rank_sweep(
        training_snapshot_dataset_bundle=training_snapshot_dataset_bundle,
        test_snapshot_dataset_bundle=test_snapshot_dataset_bundle,
        output_pca_rank_list=output_pca_rank_list,
        num_random_features=4096,
        lengthscale=1.0,
        ridge_lambda=1e-6,
        random_seed=0,
    )

    plot_output_pca_residual_energy(rank_sweep_result_bundle["spatial_pca_bundle"])
    plot_snapshot_pca_rank_sweep(rank_sweep_result_bundle)