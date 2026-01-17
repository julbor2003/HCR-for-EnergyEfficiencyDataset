import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
from src.edf import edf_normalize, col_denorm
from src.evaluation import evaluate_fold, relevance, novelty

def cross_validate(X, y, 
                   N_feature, N_target,
                   lambda_val=1e-3,
                   method="softplus",
                   a=1, b=1, eps=1e-6,
                   n_splits=10, seed=44,
                   return_coeffs=False):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ll_results, mse_results = [], []
    coeffs_list = []
    target = y.columns[0]

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, edf_models = edf_normalize(y_train, y_test)
        y_denorm = col_denorm(target, edf_models)

        if return_coeffs:
            ll, mse, coeffs_dict = evaluate_fold(X_train_norm, X_test_norm,
                                                 y_train_norm, y_test_norm,
                                                 y_test, y_denorm,
                                                 N_feature, N_target,
                                                 lambda_val=lambda_val,
                                                 method=method,
                                                 a=a, b=b, eps=eps,
                                                 return_coeffs=True)
            coeffs_list.append(coeffs_dict)
        else:
            ll, mse = evaluate_fold(X_train_norm, X_test_norm,
                                    y_train_norm, y_test_norm,
                                    y_test, y_denorm,
                                    N_feature, N_target,
                                    lambda_val=lambda_val,
                                    method=method,
                                    a=a, b=b, eps=eps)

        ll_results.append(ll)
        mse_results.append(mse)
        
    if return_coeffs: 
        return ll_results, mse_results, coeffs_list
    else:
        return ll_results, mse_results

def print_cv_results(ll_results, mse_results):
    values = np.array(ll_results)
    print("\nLog-likelihood:")
    print(f"  per fold: {np.round(values, 2)}")
    print(f"  mean LL : {values.mean():.4f}")
    print(f"  std LL  : {values.std():.4f}")

    values = np.array(mse_results)
    print("\nMean square error:")
    print(f"  per fold: {np.round(values, 2)}")
    print(f"  mean MSE: {values.mean():.4f}")
    print(f"  std MSE : {values.std():.4f}")

def cv_relevance(X, y, 
                 N_feature, N_target,
                 lambda_val=1e-3,
                 method="softplus",
                 a=1, b=1, eps=1e-6,
                 n_splits=10, seed=44):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = defaultdict(list)
    target = y.columns[0]

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, edf_models = edf_normalize(y_train, y_test)
        y_denorm = col_denorm(target, edf_models)

        for col in X.columns:
            print(f"  Evaluating relevance for feature: {col}")
            relevance_val = relevance(X_train_norm, X_test_norm,
                                      y_train_norm, y_test_norm,
                                      col,
                                      y_test, y_denorm,
                                      N_feature, N_target,
                                      lambda_val=lambda_val,
                                      method=method,
                                      a=a, b=b, eps=eps)

            results[col].append(relevance_val)
    return results

def cv_novelty(X, y, 
               N_feature, N_target,
               lambda_val=1e-3,
               method="softplus",
               a=1, b=1, eps=1e-6,
               n_splits=10, seed=44):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = defaultdict(list)
    target = y.columns[0]

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, edf_models = edf_normalize(y_train, y_test)
        y_denorm = col_denorm(target, edf_models)

        for col in X.columns:
            print(f"  Evaluating novelty for feature: {col}")
            novelty_val = novelty(X_train_norm, X_test_norm,
                                  y_train_norm, y_test_norm,
                                  col,
                                  y_test, y_denorm,
                                  N_feature, N_target,
                                  lambda_val=lambda_val,
                                  method=method,
                                  a=a, b=b, eps=eps)

            results[col].append(novelty_val)
    return results

def print_cv_relevance(results, columns, method="softplus"):
    print(f"CV Relevance Results (method: {method}):\n")
    for col in columns:
        values = np.array(results[col])
        print(f"Feature: {col}")
        print(f"  per fold: {np.round(values, 2)}")
        print(f"  mean relevance : {values.mean():.4f}")
        print(f"  std relevance  : {values.std():.4f}\n")

def print_cv_novelty(results, columns, method="softplus"):
    print(f"CV Novelty Results (method: {method}):\n")
    for col in columns:
        values = np.array(results[col])
        print(f"Feature: {col}")
        print(f"  per fold: {np.round(values, 2)}")
        print(f"  mean novelty : {values.mean():.4f}")
        print(f"  std novelty  : {values.std():.4f}\n")

def plot_cv_relevance(results, columns, method="softplus"):
    mean_vals = np.array([np.mean(results[col]) for col in columns])
    order = np.argsort(mean_vals)
    mean_vals = mean_vals[order]
    labels = np.array(columns)[order]

    plt.rcParams["font.family"] = "Times New Roman"
    _, ax = plt.subplots(figsize=(11, 0.4*len(columns)))
    bars = ax.barh(labels, mean_vals, color="green")

    ax.set_xlabel("Mean Relevance")
    ax.set_title(f"CV relevance (method: {method})")

    for bar, val in zip(bars, mean_vals):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val>0 else "right",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

def plot_cv_novelty(results, columns, method="softplus"):
    mean_vals = np.array([np.mean(results[col]) for col in columns])
    order = np.argsort(mean_vals)
    mean_vals = mean_vals[order]
    labels = np.array(columns)[order]

    plt.rcParams["font.family"] = "Times New Roman"
    _, ax = plt.subplots(figsize=(11, 0.4*len(columns)))
    bars = ax.barh(labels, mean_vals, color="orange")

    ax.set_xlabel("Mean Novelty")
    ax.set_title(f"CV novelty (method: {method})")

    for bar, val in zip(bars, mean_vals):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left" if val>0 else "right",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()