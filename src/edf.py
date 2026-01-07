import numpy as np
import matplotlib.pyplot as plt

def edf_fit(x):
    x_sorted = np.sort(x)
    n = len(x_sorted)
    eps = 1.0/(n+1)

    u_grid = np.linspace(eps, 1 - eps, n)

    def edf(z):
        u = np.searchsorted(x_sorted, z, side="right")/n
        return np.clip(u, eps, 1-eps)
    
    def edf_inv(u):
        u = np.clip(u, eps, 1 - eps)
        return np.interp(u, u_grid, x_sorted)

    return edf, edf_inv

def edf_normalize(X_train, X_test):
    edf_models = {}
    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    for col in X_train.columns:
        edf, edf_inv = edf_fit(X_train[col].values)
        edf_models[col] = (edf, edf_inv)

        X_train_norm[col] = edf(X_train[col].values)
        X_test_norm[col]  = edf(X_test[col].values)

    return X_train_norm, X_test_norm, edf_models

def col_denorm(col, edf_models):
    def denormalize(x_norm):
        return edf_models[col][1](x_norm)
    return denormalize

def plot_hists(x, x_norm, name="", bins=50, grid=False):
    _, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    axes[0].hist(x, bins=bins, density=True)
    axes[0].set_title(f"{name} – before EDF")

    axes[1].hist(x_norm, bins=bins, density=True)
    axes[1].set_title(f"{name} – after EDF")

    for ax in axes:
        ax.grid(grid)

    plt.show()

def plot_raw_vs_norm(x, x_norm, name="", grid=False):
    plt.figure(figsize=(5, 4))
    plt.scatter(x, x_norm, s=10, alpha=0.6)
    plt.title(name)
    plt.xlabel("raw")
    plt.ylabel("EDF(raw)")
    plt.grid(grid)
    plt.show()