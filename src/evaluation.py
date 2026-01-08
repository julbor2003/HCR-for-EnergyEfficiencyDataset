import numpy as np
from src.hcr import fit_lasso, make_density, calibrate_density
from src.features import moment_like_features, prepare_targets
from src.weights import analyze_weights, group_coeffs

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
    calibration_method,
    return_coeffs=False
):
    V_train = moment_like_features(X_train, N)
    V_test  = moment_like_features(X_test, N)
    targets_train = prepare_targets(y_train, N)

    models = []
    for n in range(N):
        models.append(fit_lasso(V_train, targets_train[n], lambda_val))

    ll = mean_log_likelihood(
        V_test,
        y_test,
        models,
        calibration_method=calibration_method
    )
    if return_coeffs:
        coeffs_dict = {}
        for deg, model in enumerate(models, 1):
            weights = analyze_weights(model, V_train)
            coeffs_dict[deg] = group_coeffs(weights)
        return ll, coeffs_dict
    else:
        return ll

def relevance(
        X_train, X_test,
        y_train, y_test,
        col, N, lambda_val,
        calibration_method
):
    X_train_mod = X_train[[col]]
    X_test_mod  = X_test[[col]]

    return evaluate_fold(
        X_train_mod, X_test_mod,
        y_train, y_test,
        N, lambda_val,
        calibration_method
    )

def novelty(
        X_train, X_test,
        y_train, y_test,
        col, N, lambda_val,
        calibration_method
):
    X_train_mod = X_train.drop(columns=[col])
    X_test_mod  = X_test.drop(columns=[col])

    ll_base = evaluate_fold(
        X_train, X_test,
        y_train, y_test,
        N, lambda_val,
        calibration_method
    )
    ll_mod = evaluate_fold(
        X_train_mod, X_test_mod,
        y_train, y_test,
        N, lambda_val,
        calibration_method
    )

    return ll_mod-ll_base

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