class Config:
    seed = 42

    EQ = "viscous_burgers_utx_1d"

    train_sample_size = 5000
    test_sample_size = 5000
    max_available_samples = 10000

    tasks = ["continuous"]

    t1 = 0.5
    Kx = 128
    Nt = 32
    Kin = None

    USE_PCA = True
    PCA_COMPONENTS = 2048

    train = "rf"
    device = "cpu"
    FORCE_RECOMPUTE = False

    shared_kernel = True
    d_features = 8192
    lengthscale = 0.5
    LAMBDA_REG = 1e-4

    normalize_const_in = 1.0
    normalize_const_out = 1.0
    eval_count = 50

    ANALYSIS_SAMPLE_SIZE_TRAIN = 2000
    ANALYSIS_SAMPLE_SIZE_TEST = 2000

    INPUT_RANK_CANDIDATES = [4, 6, 8, 10, 12, 16, 20]

    STEP_OUTPUT_RANK_CANDIDATES = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    STEP_PCA_RANK_CANDIDATES = [4, 8, 16, 32, 64, 128]

    Nt_CANDIDATES = [4, 8, 12, 16, 20, 24]
    Kx_CANDIDATES = [8, 16, 24, 32, 48, 64, 96, 128]
    CONTINUOUS_PCA_RANK_CANDIDATES = [128, 256, 512, 1024, 2048]

    OUTPUT_BASIS_TOL = 1e-4
    OUTPUT_PCA_TOL = 1e-4

    STEP_OUTPUT_BASIS_ERROR_METRIC = "function_rel_l2_mean_vs_raw"
    STEP_OUTPUT_PCA_ERROR_METRIC = "function_rel_l2_mean"

    CONTINUOUS_OUTPUT_BASIS_ERROR_METRIC = "function_rel_l2_mean_vs_raw"
    CONTINUOUS_OUTPUT_PCA_ERROR_METRIC = "function_rel_l2_mean"

    PLOT_DIMENSION_SELECTION = True
    USE_AUTO_SELECTED_STEP_DIMENSIONS = True
    USE_AUTO_SELECTED_CONTINUOUS_DIMENSIONS = True

    RUN_PCA_COMPRESSION_STUDY = True
    PCA_COMPRESSION_STUDY_RANKS = (128, 256, 512, 1024)
    PCA_COMPRESSION_STUDY_N_SHOW = 3

    VISUALIZE_CONTINUOUS_RESULT = True
    VIS_SAMPLE_INDEX = 0
    VIS_TITLE_PREFIX = "Task B"


cfg = Config()