from sklearn.decomposition import PCA
import joblib
import numpy as np
import matplotlib.pyplot as plt

def fit_and_save_pca(data_matrix, n_components, save_path):
    """
    Fit PCA on data_matrix (N, D), reduce to n_components dimensions,
    and save the PCA object to disk.
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_matrix)
    joblib.dump(pca, save_path)
    return transformed, pca

def load_pca(pca_path):
    return joblib.load(pca_path)

def recover_from_pca(transformed_data, pca):
    """ Project low-dimensional data back to original basis """
    return pca.inverse_transform(transformed_data)



def test_fit_and_save_pca():
    np.random.seed(42)

    # Create synthetic data with known structure
    N, D = 100, 20
    true_rank = 5
    U = np.random.randn(N, true_rank)
    V = np.random.randn(true_rank, D)
    data = U @ V + 0.1 * np.random.randn(N, D)  # Low-rank + small noise

    # Fit PCA
    n_components = 5
    save_path = "test_pca_model.pkl"
    transformed, pca = fit_and_save_pca(data, n_components, save_path)

    # Reconstruct
    recovered = pca.inverse_transform(transformed)

    # Compute reconstruction error
    mse = np.mean((data - recovered) ** 2)
    print("Reconstruction MSE:", mse)

    # Plot a few example reconstructions
    for i in range(3):
        plt.figure(figsize=(10, 2))
        plt.plot(data[i], label="Original")
        plt.plot(recovered[i], label="Reconstructed", linestyle='--')
        plt.legend()
        plt.title(f"Sample {i} Reconstruction")
        plt.show()

# test_fit_and_save_pca()