import numpy as np
from src.hcr import fit_lasso, make_density, calibrate_density
from src.features import moment_like_features, target_features

def mean_log_likelihood(V_test, y_test, models,
                        calibration_method = "softplus",
                        eps=1e-6, n_grid=1_000):
    log_vals = []
    for id in range(len(V_test)):
        v = V_test.iloc[[id]]
        y_true = y_test.iloc[id, 0]

        density = make_density(models, v)
        density = calibrate_density(density,
                                    method=calibration_method,
                                    eps=eps, n_grid=n_grid)

        density_val = density(y_true)
        log_vals.append(np.log(max(density_val, eps)))
    
    return np.mean(log_vals)

def evaluate_fold(
    X_train, X_test,
    y_train, y_test,
    N, lambda_val,
    calibration_method
):
    V_train = moment_like_features(X_train, N)
    V_test  = moment_like_features(X_test, N)

    targets_train = []
    targets_test  = []

    for n in range(1, N+1):
        targets_train.append(target_features(y_train, n))
        targets_test.append(target_features(y_test, n))

    models = []
    for n in range(N):
        models.append(fit_lasso(V_train, targets_train[n], lambda_val))

    ll = mean_log_likelihood(
        V_test,
        y_test,
        models,
        calibration_method=calibration_method
    )

    return ll

def expected_value(density, n_grid=1_000):
    grid = np.linspace(0, 1, n_grid)
    p = density(grid)
    return np.trapz(grid * p, grid)

def mse_evaluation(V, y, models, y_denorm,
                   calibration_method="softplus",
                   n_grid=1_000):
    errors = []
    for id in range(len(V)):
        v = V.iloc[[id]]
        y_true = y.iloc[[id]].iloc[0, 0]

        density = calibrate_density(make_density(models, v), 
                                    method=calibration_method)
        y_pred = expected_value(density, n_grid=n_grid)
        y_pred_denorm = y_denorm(y_pred)

        errors.append((y_true-y_pred_denorm)**2)
    return np.mean(errors)