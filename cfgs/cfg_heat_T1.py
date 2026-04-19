import torch

class Settings:
    USE_PCA = False
    PCA_COMPONENTS = 9
    PCA_PATH_IN = "models/pca_model_in.pkl"
    PCA_PATH_OUT = "models/pca_model_out.pkl"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_INTERVAL = 100
    LAMBDA_REG = 0.0
    LR = 1e-3
    EPOCHS = 3000
    BATCH_SIZE = 64
    train = 'rf'  # 'rf' or 'nn'
    FORCE_RECOMPUTE = True # set to True when dataset or expansion dimension changes
    EQ = 'heat_T1' # options: 'darcy', 'poisson', 'semilinear_poisson_2', 'semilinear_poisson_3', 'semilinear_poisson_4', 'semilinear_poisson_5', 'sine_gordon'
    N_ex = 4
    compare_original = False
    lengthscale = 4 # larger lengthscale -> smoother kernel in general. more detailed adjustment see scaling
    # NOTE is approximation fails try tuning the lengthscale value.
    scaling = 0.1 # scaling for sigma decay, larger value -> more decay. this should match sampling scaling. 
    sample_num = 10
    basis_a = 'sine'
    # sol_space = 'H1'  # options: 'L2', 'H1'
    # param_space = 'L2'  # options: 'L2', 'H1'

    basis_u = 'sine' #  'sine'
    kernel =  'Gaussian' # 'Gaussian' 'Cauchy'
    d_features = 512 # err 2048 : 0.0012664258 1024: 0.0011661687 512: 0.0009678079 ! overfitting?
cfg = Settings()
