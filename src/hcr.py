import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from src.legendre import RescaledLegendre

def fit_lasso (V, target, lambda_val=1e-3):
    model = Lasso(
        alpha=lambda_val,
        fit_intercept=False,
        max_iter=100_000
    )

    model.fit(V, target)
    return model

def make_density(models, v):
    coeffs = [model.predict(v)[0] for model in models]
    polys = [RescaledLegendre(i) for i in range(1, len(coeffs) + 1)]

    def density(y):
        p = 1

        for a, L in zip(coeffs, polys):
            p += a*L(y)
        
        return p
    
    return density

def softplus(z):
    return np.log1p(np.exp(z))

def calibrate_density(density, method="softplus", eps=1e-6, n_grid=1_000):
    grid = np.linspace(0, 1, n_grid)
    raw_vals = density(grid)

    if method == "softplus":
        vals = softplus(raw_vals)
        area = np.trapz(vals, grid)

        def calibrate_density(y):
            p = density(y)
            p = softplus(p)
            return p/area
        
        return calibrate_density
    
    elif method == "clip":
        vals = np.maximum(raw_vals, eps)
        area = np.trapz(vals, grid)

        def calibrate_density(y):
            p = density(y)
            p = np.maximum(p, eps)
            return p/area
        
        return calibrate_density
    
    else:
        raise ValueError("method must be 'softplus' or 'clip'")
    
def plot_density(density, y=None, method="softplus", raw=False):
    plt.figure(figsize=(6,4))
    xs = np.linspace(0, 1, 500)

    ys = density(xs)
    if raw==True:
        plt.plot(xs, ys, lw=2, color="orange", label="Raw density")

    calibrated_density = calibrate_density(density, method=method)
    ys = calibrated_density(xs)
    plt.plot(xs, ys, lw=2, color="green", label="Calibrated density")

    plt.axhline(
        y=0, 
        color="black",
        linewidth=1
        )

    if y is not None:
        plt.axvline(
            x=y, 
            color="red", 
            linestyle="--",
            linewidth=1,
            label="True value")

    plt.title(f"Density calibration (method: {method})")
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_example_densities(V_test, y_test, models, name="Example densities", method="softplus", seed=None, raw=False):
    if seed is not None:
        np.random.seed(seed)

    n = len(V_test)
    ids = np.random.choice(n, size=16, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()

    xs = np.linspace(0, 1, 500)

    for ax, idx in zip(axes, ids):
        v = V_test.iloc[[idx]]
        y = y_test.iloc[[idx]].iloc[0, 0]

        density = make_density(models, v)
        ys = density(xs)
        if raw==True:
            ax.plot(xs, ys, lw=2, color="orange", label="Raw density")

        calibrated_density = calibrate_density(density, method=method)
        ys = calibrated_density(xs)
        ax.plot(xs, ys, lw=2, color="green", label="Calibrated density")

        ax.axvline(
            x=y,
            color="red",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            y=0, 
            color="black",
            linewidth=1
            )
        
    fig.suptitle(name, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()