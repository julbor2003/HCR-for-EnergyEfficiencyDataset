import numpy as np
import pandas as pd
from src.hcr import fit_lasso, make_density, calibrate_density
from src.features import moment_like_features, prepare_targets
from src.weights import analyze_weights, group_coeffs

def mean_log_likelihood(V_test, y_test, models,
                        method = "softplus",
                        a=1, b=1, eps=1e-6, 
                        n_grid=1_000):
    log_vals = []
    for id in range(len(V_test)):
        v = V_test.iloc[[id]]
        y_true = y_test.iloc[id, 0]

        density = make_density(models, v)
        density = calibrate_density(density,
                                    method=method,
                                    a=a, b=b, eps=eps, 
                                    n_grid=n_grid)

        density_val = density(y_true)
        log_vals.append(np.log(max(density_val, eps)))
    
    return np.mean(log_vals)

def expected_value(density, n_grid=1_000):
    grid = np.linspace(0, 1, n_grid)
    p = density(grid)
    return np.trapz(grid * p, grid)

def mse_evaluation(V, y, models, y_denorm,
                   method="softplus",
                   a=1, b=1, eps=1e-6,
                   n_grid=1_000):
    errors = []
    for id in range(len(V)):
        v = V.iloc[[id]]
        y_true = y.iloc[[id]].iloc[0, 0]

        density = make_density(models, v)
        density = calibrate_density(density, 
                                    method=method, 
                                    a=a, b=b, eps=eps,
                                    n_grid=n_grid)
        
        y_pred = expected_value(density, n_grid=n_grid)
        y_pred_denorm = y_denorm(y_pred)

        errors.append((y_true-y_pred_denorm)**2)
    return np.mean(errors)

def evaluate_fold(X_train, X_test,
                  y_train, y_test,
                  y_test_original, y_denorm,
                  N_feature, N_target,
                  lambda_val=1e-3,
                  method="softplus",
                  a=1, b=1, eps=1e-6,
                  return_coeffs=False):
    
    V_train = moment_like_features(X_train, N_feature)
    V_test  = moment_like_features(X_test, N_feature)
    targets_train = prepare_targets(y_train, N_target)

    models = []
    for n in range(N_target):
        models.append(fit_lasso(V_train, targets_train[n], lambda_val))

    ll = mean_log_likelihood(V_test, y_test, models,
                             method=method,
                             a=a, b=b, eps=eps)
    mse = mse_evaluation(V_test, y_test_original, models, y_denorm,
                         method=method, 
                         a=a, b=b, eps=eps)

    if return_coeffs:
        coeffs_dict = {}
        for deg, model in enumerate(models, 1):
            weights = analyze_weights(model, V_train)
            coeffs_dict[deg] = group_coeffs(weights)
        return ll, mse, coeffs_dict
    else:
        return ll, mse

def relevance(X_train, X_test,
              y_train, y_test,
              col,
              y_test_original, y_denorm,
              N_feature, N_target,
              lambda_val=1e-3,
              method="softplus",
              a=1, b=1, eps=1e-6):
    
    X_train_mod = X_train[[col]]
    X_test_mod  = X_test[[col]]

    ll, _ = evaluate_fold(X_train_mod, X_test_mod,
                         y_train, y_test,
                         y_test_original, y_denorm,
                         N_feature, N_target,
                         lambda_val=lambda_val,
                         method=method,
                         a=a, b=b, eps=eps)
    return ll

def report_relevance(columns_relevance, method="softplus"):
    relevance_dict = {
        col: scores[method]
        for col, scores in columns_relevance.items()
    }
    sorted_items = sorted(relevance_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"Feature Relevance Ranking (method: {method}):")
    for rank, (feature, relevance_score) in enumerate(sorted_items, start=1):
        print(f"{rank}. {feature}: {relevance_score:.4f}")

def relevance_to_df(softplus_results,
                    param_softplus_results,
                    clip_results):
    features = softplus_results.keys()
    methods = {
        "softplus": softplus_results,
        "param_softplus": param_softplus_results,
        "clip": clip_results
    }

    rows = []
    for feature in features:
        row = {}
        for method, res in methods.items():
            values = np.asarray(res[feature])
            row[(method, "mean")] = values.mean()
            row[(method, "std")] = values.std()
        rows.append(row)

    df = pd.DataFrame(rows, index=features)
    df.index.name = "feature"
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

def novelty(X_train, X_test,
            y_train, y_test,
            col,
            y_test_original, y_denorm,
            N_feature, N_target,
            lambda_val=1e-3,
            method="softplus",
            a=1, b=1, eps=1e-6):
    
    X_train_mod = X_train.drop(columns=[col])
    X_test_mod  = X_test.drop(columns=[col])

    ll_base, _ = evaluate_fold(X_train, X_test,
                               y_train, y_test,
                               y_test_original, y_denorm,
                               N_feature, N_target,
                               lambda_val=lambda_val,
                               method=method,
                               a=a, b=b, eps=eps)
    ll_mod, _  = evaluate_fold(X_train_mod, X_test_mod,
                               y_train, y_test,
                               y_test_original, y_denorm,
                               N_feature, N_target,
                               lambda_val=lambda_val,
                               method=method,
                               a=a, b=b, eps=eps)
    return ll_base-ll_mod

def report_novelty(columns_novelty, method="softplus"):
    novelty_dict = {
        col: scores[method]
        for col, scores in columns_novelty.items()
    }
    sorted_items = sorted(novelty_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"Feature Novelty Ranking (method: {method}):")
    for rank, (feature, novelty_score) in enumerate(sorted_items, start=1):
        print(f"{rank}. {feature}: {novelty_score:.4f}")

def novelty_to_df(softplus_results,
                  param_softplus_results,
                  clip_results):
    return relevance_to_df(softplus_results,
                           param_softplus_results,
                           clip_results)