import matplotlib.pyplot as plt

feature_dim = [128, 256, 512, 1024, 2048, 4096, 8192]

vec_mse = [
    1.2377881318308753e-12,
    1.3179217899439289e-12,
    1.2521195745154386e-12,
    1.3596430839411816e-12,
    1.3080976506136738e-12,
    1.2742603971758785e-12,
    1.2599128817760805e-12,
]

coeff_rel_l2 = [
    0.005143158497208342,
    0.005307030338193634,
    0.005172847242120265,
    0.005390377862349876,
    0.005287213331421215,
    0.005218381717935789,
    0.005188920408924661,
]

function_rel_l2_mean = [
    0.004133403043073112,
    0.0044039487891189425,
    0.004147981222027595,
    0.004402167653288475,
    0.004254063874922372,
    0.004152561855486933,
    0.0041625366323433536,
]

function_rel_l2_median = [
    0.0027044411465740707,
    0.0028489062087875636,
    0.0026832552343353964,
    0.002863268790730429,
    0.0027744399770714056,
    0.002662051352919119,
    0.002691770669996498,
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(feature_dim, vec_mse, marker='o')
axes[0, 0].set_title("Vector-space test MSE")
axes[0, 0].set_xlabel("Feature dimension")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].set_xscale("log", base=2)
axes[0, 0].set_yscale("log")

axes[0, 1].plot(feature_dim, coeff_rel_l2, marker='o')
axes[0, 1].set_title("Coefficient-space relative L2")
axes[0, 1].set_xlabel("Feature dimension")
axes[0, 1].set_ylabel("Relative L2")
axes[0, 1].set_xscale("log", base=2)

axes[1, 0].plot(feature_dim, function_rel_l2_mean, marker='o', label='mean')
axes[1, 0].plot(feature_dim, function_rel_l2_median, marker='s', label='median')
axes[1, 0].set_title("Function-space relative L2")
axes[1, 0].set_xlabel("Feature dimension")
axes[1, 0].set_ylabel("Relative L2")
axes[1, 0].set_xscale("log", base=2)
axes[1, 0].legend()

axes[1, 1].plot(feature_dim, coeff_rel_l2, marker='o', label='coeff rel L2')
axes[1, 1].plot(feature_dim, function_rel_l2_mean, marker='s', label='function rel L2 mean')
axes[1, 1].plot(feature_dim, function_rel_l2_median, marker='^', label='function rel L2 median')
axes[1, 1].set_title("Error comparison")
axes[1, 1].set_xlabel("Feature dimension")
axes[1, 1].set_ylabel("Error")
axes[1, 1].set_xscale("log", base=2)
axes[1, 1].legend()

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()