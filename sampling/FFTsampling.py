import numpy as np
import matplotlib.pyplot as plt

'''works! '''

def fft_sample_grf(domain, n_samples, grid_size, length_scale=1, sigma_f = 100):
    """
    Sample functions from a 2D Gaussian Random Field using the FFT method.
    
    Parameters:
        domain (array): 2*2 array representing the domain. [[x_inf, x_sup],[y_inf, y_sup]]
        n_samples (int): Number of samples to generate.
        grid_size (int): Number of points per dimension.
        length_scale (float): Correlation length of the RBF kernel.
        sigma_f (float): Standard deviation of the field.
    
    Returns:
        X (ndarray): Meshgrid X coordinates.
        Y (ndarray): Meshgrid Y coordinates.
        samples (ndarray): (n_samples, grid_size, grid_size) sampled GRFs.
    """
    
    # Define 2D grid
    x = np.linspace(domain[0][0], domain[0][1], grid_size)
    y = np.linspace(domain[1][0], domain[1][1], grid_size)

    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # Generate Fourier frequency coordinates
    k_x = np.fft.fftfreq(grid_size, d=1/grid_size)[:, None]  # Shape (grid_size, 1)
    k_y = np.fft.fftfreq(grid_size, d=1/grid_size)[None, :]  # Shape (1, grid_size)
    k_sq = k_x**2 + k_y**2  # Squared frequency components
    
    # Compute power spectrum S(k) based on RBF kernel
    S_k = np.exp(-0.5 * k_sq / (length_scale**2))
    
    # # Normalize power spectrum
    # # NO need
    # print(np.max(S_k) - np.min(S_k))
    # S_k /= np.max(S_k)
    
    
    # Generate random samples in Fourier space
    samples = np.zeros((n_samples, grid_size, grid_size))
    
    for i in range(n_samples):
        # Generate white noise in Fourier space (real and imaginary parts)
        noise_real = np.random.randn(grid_size, grid_size)
        noise_imag = np.random.randn(grid_size, grid_size)
        noise = noise_real + 1j * noise_imag
        
        # Apply spectral filter (square root of power spectrum)
        field_k = noise * np.sqrt(S_k)
        
        # Transform back to real space using inverse FFT
        field_real = np.real(np.fft.ifft2(field_k)) * grid_size  * sigma_f  # Scale by sigma_f
        # print(np.max(field_real) - np.min(field_real))
        # Store the sample
        samples[i] = field_real
    
    return X, Y, samples

# # Example usage
# np.random.seed(42)  # For reproducibility
# grid_size = 100  # Number of points per dimension
# n_samples = 3    # Number of sampleds fields

# X, Y, samples = fft_sample_grf(np.array([[-1,1],[-1,1]]), n_samples, grid_size) # length_scale=5, sigma_f=10)

# # Plot the sampled 2D fields
# fig, axes = plt.subplots(1, n_samples, figsize=(12, 4))
# for i in range(n_samples):
#     ax = axes[i]
#     c = ax.contourf(X, Y, samples[i], levels=100, cmap="coolwarm")
#     fig.colorbar(c, ax=ax)
#     ax.set_title(f"Sample {i+1}")

# plt.show()
