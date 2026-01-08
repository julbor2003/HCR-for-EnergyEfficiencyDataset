import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
from src.edf import edf_normalize
from src.evaluation import evaluate_fold, relevance, novelty

def cross_validate(X, y, N=4, lambda_val=1e-3, n_splits=10, seed=44):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {"softplus": [], "clip": []}
    coeffs_list = []

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, _ = edf_normalize(y_train, y_test)

        for method in ["softplus", "clip"]:
            ll, coeffs_dict = evaluate_fold(
                X_train_norm, X_test_norm,
                y_train_norm, y_test_norm,
                N=N,
                lambda_val=lambda_val,
                calibration_method=method,
                return_coeffs=True
            )
            results[method].append(ll)
            print(f"  {method}: {ll:.4f}")
        coeffs_list.append(coeffs_dict)
        
    return results, coeffs_list

def cv_relevance(X, y, N=4, lambda_val=1e-3, n_splits=10, seed=44):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {"softplus": defaultdict(list), "clip": defaultdict(list)}

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, _ = edf_normalize(y_train, y_test)

        for col in X.columns:
            print(f"  Evaluating relevance for feature: {col}")
            for method in ["softplus", "clip"]:
                relevance_val = relevance(
                    X_train_norm, X_test_norm,
                    y_train_norm, y_test_norm,
                    col, N=N,
                    lambda_val=lambda_val,
                    calibration_method=method
                )
                results[method][col].append(relevance_val)

    return results

def cv_novelty(X, y, N=4, lambda_val=1e-3, n_splits=10, seed=44):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {"softplus": defaultdict(list), "clip": defaultdict(list)}

    for fold, (train_ids, test_ids) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        
        X_train_norm, X_test_norm, _ = edf_normalize(X_train, X_test)
        y_train_norm, y_test_norm, _ = edf_normalize(y_train, y_test)

        for col in X.columns:
            print(f"  Evaluating novelty for feature: {col}")
            for method in ["softplus", "clip"]:
                novelty_val = novelty(
                    X_train_norm, X_test_norm,
                    y_train_norm, y_test_norm,
                    col, N=N,
                    lambda_val=lambda_val,
                    calibration_method=method
                )
                results[method][col].append(novelty_val)

    return results

def print_cv_relevance(results, columns, method="softplus"):
    print(f"CV Relevance Results (method: {method}):\n")
    for col in columns:
        values = np.array(results[method][col])
        print(f"Feature: {col}")
        print(f"  per fold: {np.round(values, 2)}")
        print(f"  mean relevance : {values.mean():.4f}")
        print(f"  std relevance  : {values.std():.4f}\n")

def print_cv_novelty(results, columns, method="softplus"):
    print(f"CV Novelty Results (method: {method}):\n")
    for col in columns:
        values = np.array(results[method][col])
        print(f"Feature: {col}")
        print(f"  per fold: {np.round(values, 2)}")
        print(f"  mean novelty : {values.mean():.4f}")
        print(f"  std novelty  : {values.std():.4f}\n")

def plot_cv_relevance(results, columns, method="softplus"):
    mean_vals = np.array([np.mean(results[method][col]) for col in columns])
    order = np.argsort(mean_vals)
    mean_vals = mean_vals[order]
    labels = np.array(columns)[order]

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(11, 0.4*len(columns)))
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
    mean_vals = np.array([np.mean(results[method][col]) for col in columns])
    order = np.argsort(mean_vals)
    mean_vals = mean_vals[order]
    labels = np.array(columns)[order]

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(11, 0.4*len(columns)))
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