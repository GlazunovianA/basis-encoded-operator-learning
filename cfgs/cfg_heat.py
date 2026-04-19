import torch

class Settings:
    USE_PCA = False
    PCA_PATH_IN = "models/heat/pca_model_in.pkl"
    PCA_PATH_OUT = "models/heat/pca_model_out.pkl"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_INTERVAL = 100
    LAMBDA_REG = 0.5
    LR = 1e-2
    EPOCHS = 3000
    BATCH_SIZE = 64
    train = 'rf'  # 'rf' or 'nn'
    FORCE_RECOMPUTE = False # set to True when dataset or expansion dimension changes
    EQ = 'heat' # options: 'darcy', 'poisson', 'semilinear_poisson_2', 'semilinear_poisson_3', 'semilinear_poisson_4', 'semilinear_poisson_5', 'sine_gordon'
    N_ex = 5 # expansion dimension on each input channel
    compare_original = False
    lengthscale = 1 # larger lengthscale -> smoother kernel in general. more detailed adjustment see scaling
    # NOTE if approximation fails try tuning the lengthscale value.
    scaling = 1 # scaling for sigma decay, larger value -> more decay. this should match sampling scaling. 
    # scaling scales with input smoothness factor 
    sample_num = 10
    basis_a = 'legendre'
    # sol_space = 'H1'  # options: 'L2', 'H1'
    # param_space = 'L2'  # options: 'L2', 'H1'

    basis_u = 'sine' #  'sine'
    kernel =  'Gaussian' # 'Gaussian' 'Cauchy'
    d_features = 2048

        # --- patch ---
    N_ex_input = 16
    N_ex_output = 32   # initial fallback/default only

    USE_PCA = True
    PCA_COMPONENTS = 32   # fallback/default only

    AUTO_DIMENSION_SELECTION = True

    ANALYSIS_SAMPLE_SIZE_TRAIN = 300
    ANALYSIS_SAMPLE_SIZE_TEST = 100

    OUTPUT_RANK_CANDIDATES = [4, 8, 12, 16, 20, 24, 32]
    REFERENCE_OUTPUT_RANK = 32
    OUTPUT_BASIS_TOL = 1e-4
    OUTPUT_BASIS_ERROR_METRIC = "function_rel_l2_mean_vs_ref"

    PCA_RANK_CANDIDATES = [4, 8, 16, 24, 32, 48, 64]
    OUTPUT_PCA_TOL = 1e-4
    OUTPUT_PCA_ERROR_METRIC = "function_rel_l2_mean"
    VISUALIZE_PREDICTION = True
cfg = Settings()
