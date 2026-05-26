import torch

class Settings:
    USE_PCA = False
    PCA_COMPONENTS = 9
    PCA_PATH_IN = "models/pca_model_in.pkl"
    PCA_PATH_OUT = "models/pca_model_out.pkl"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_INTERVAL = 100
    LAMBDA_REG = 1e-4
    EPOCHS = 3000
    BATCH_SIZE = 64
    train = 'rf'  # 'rf' or 'nn'
    FORCE_RECOMPUTE = False # set to True when dataset or expansion dimension changes
    EQ = 'semilinear_poisson_3_basis_input' # options: 'darcy', 'poisson', 'semilinear_poisson_2', 'semilinear_poisson_3', 'semilinear_poisson_4', 'semilinear_poisson_5', 'sine_gordon'
    compare_original = False
    lengthscale = 4 # larger lengthscale -> smoother kernel in general. more detailed adjustment see scaling
    # NOTE is approximation fails try tuning the lengthscale value.
    scaling = 0 # scaling for sigma decay, larger value -> more decay
    sample_num = 10
    basis_a = 'fourier'
    # sol_space = 'H1'  # options: 'L2', 'H1'
    # param_space = 'L2'  # options: 'L2', 'H1'
    d_features = 1024 # ? why this has huge error when testing?
    basis_u = 'sine' #  'sine'
    kernel =  'Gaussian' # 'Gaussian' 'Cauchy'
    # --- patch ---
    N_ex_input = 8 # full sampled vector input 
    N_ex_output = 32   # initial fallback/default only

    train_sample_size = 500
    test_sample_size = 100
    USE_PCA = True
    PCA_COMPONENTS = 64   # fallback/default only

    AUTO_DIMENSION_SELECTION = True

    ANALYSIS_SAMPLE_SIZE_TRAIN = 300
    ANALYSIS_SAMPLE_SIZE_TEST = 100

    OUTPUT_RANK_CANDIDATES = [4, 8, 12, 16, 20, 24, 32] 
    # NOTE this is the single-axis dimension candidate. the overall dimension is c
    REFERENCE_OUTPUT_RANK = 32
    OUTPUT_BASIS_TOL = 1e-5
    OUTPUT_BASIS_ERROR_METRIC = "function_rel_l2_mean_vs_ref"

    PCA_RANK_CANDIDATES = [4, 8, 16, 24, 32, 48, 64, 96]
    OUTPUT_PCA_TOL = 1e-4 # TODO check these values
    OUTPUT_PCA_ERROR_METRIC = "function_rel_l2_mean"

cfg = Settings()

