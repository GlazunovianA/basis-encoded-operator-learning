import numpy as np
from pca_for_training_data import fit_and_save_pca, load_pca, recover_from_pca

def descending_matrix(N, scale=1.0, decay='exponential'):
    """
    Generate an (N, N) matrix where values descend smoothly from (0, 0).
    Uses exponential decay based on distance from the top-left corner.
    """
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    distance = np.sqrt(X**2 + Y**2)
    if decay == 'exponential':
        matrix = 1*np.exp(-distance / (N / scale))  # scale controls steepness of decay
    elif decay == 'polynomial':
        deg = 2
        matrix = np.power((np.sqrt(2)*N) - distance, deg) / ((np.sqrt(2)*N)**deg)  # polynomial decay
    elif decay == 'linear':
        matrix = ((np.sqrt(2)*N) - distance) / ((np.sqrt(2)*N))
    return matrix



def zigzag_rowcol(n):
    '''Returns two 1D arrays of row and col indices in zigzag order.'''
    coords = sorted(((i, j) for i in range(n) for j in range(n)),
                    key=lambda x: (x[0] + x[1], -x[0] if (x[0] + x[1]) % 2 else x[0]))
    rows, cols = zip(*coords)
    return np.array(rows), np.array(cols)

def z_flatten(M):
    """
    Flatten a batch of 2D matrices in zigzag order.
    
    Parameters:
        M (ndarray): 3D matrix to flatten.
        
    Returns:
        ndarray: Flattened array in zigzag order.
    """
    # print("M.shape:", M.shape)
    n = M.shape[1]
    assert M.shape[1] == M.shape[2], "Matrices must be square."
    assert M.ndim == 3, "Input must be a 3D matrix."
    rows, cols = zigzag_rowcol(n)
    return np.array([m[rows, cols] for m in M])


def z_reconstruction(M):
    '''
    Reconstruct a 2D matrix from zigzag flattened data.
    Parameters:
        M (ndarray): 2D array of zigzag flattened data. 1st dim: number of matrices, 2nd dim: flattened data.
    Returns:    
        ndarray: Reconstructed 2D matrices.
    '''
    if M.ndim == 2:
        B, flat_len = M.shape

    elif M.ndim == 1:
        # assume M is a single flattened matrix
        flat_len = M.shape[0]
        B = 1
    n = int(np.sqrt(flat_len))
    assert n * n == flat_len, "Flattened data must have square length."

    rows, cols = zigzag_rowcol(n)
    output = np.zeros((B, n, n), dtype=M.dtype)
    for i in range(B):
        output[i][rows, cols] = M[i]
    return output.squeeze() if B == 1 else output


# for i in range(10):
#     M = np.random.rand(100, 5, 5)
#     M_flatten = z_flatten(M)
#     M_transform, pca = fit_and_save_pca(M_flatten, n_components=24, save_path='pca_model_test')
#     M_recover = recover_from_pca(M_transform, pca)
#     M_reconstruction = z_reconstruction(M_recover)
#     # print(np.allclose(z_reconstruction(z_flatten(M_reconstruction)), M, atol = 1))  # should be True
#     print(f"Iteration {i}: Reconstruction error:", np.linalg.norm(M - M_reconstruction))

def sigma_generate(d_in, PCA = False, decay_type = 'exponential', decay_rate = None, sigma_max = 1.0, sigma_min = 0.01):
    ''' returns appropriately decaying np.array of sigma values.'''
    if PCA == True:
        sigma = np.linspace(sigma_max, sigma_min, d_in)
    else:
        # matrix-ordered
        if decay_rate == None:
            dim = np.sqrt(d_in).astype(int)
            sigma_val = np.linspace(sigma_max, sigma_min, 2*dim-1)
            diag_mat = np.concat((np.linspace(1, dim, dim), np.linspace(dim-1, 1, dim-1)), axis = 0).astype(int)
            for i in range(2*dim-1):
                if i > 0:
                    sigma = np.concat((sigma, [sigma_val[i]]*diag_mat[i]), axis = 0)
                else:
                    sigma = np.array([sigma_val[i]]*diag_mat[i])
        else:
            dim = np.sqrt(d_in).astype(int)
            if decay_type == 'exponential':
                sigma_val = np.array([np.exp(-decay_rate * ((1 + i)**2)) for i in range(2*dim-1)])
            elif decay_type == 'polynomial':
                sigma_val = np.array([(i+1)**(-decay_rate) for i in range(2*dim-1)])
            diag_mat = np.concat((np.linspace(1, dim, dim), np.linspace(dim-1, 1, dim-1)), axis = 0).astype(int)
            for i in range(2*dim-1):
                if i > 0:
                    sigma = np.concat((sigma, [sigma_val[i]]*diag_mat[i]), axis = 0)
                else:
                    sigma = np.array([sigma_val[i]]*diag_mat[i])
            
    return sigma
# print(sigma_generate(9, PCA=False))