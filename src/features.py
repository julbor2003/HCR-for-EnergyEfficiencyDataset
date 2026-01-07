from src.legendre import RescaledLegendre
import pandas as pd

def moment_like_features (X, N):
    features = {}
    for col in X.columns:
        x = X[col].values

        for n in range(1, N+1):
            poly = RescaledLegendre(n)
            features[f"L_{n}({col})"] = poly(x)

    features = pd.DataFrame(features, index=X.index)
    return features

def target_features (Y, n):
    features = {}
    for col in Y.columns:
        y = Y[col].values

        poly = RescaledLegendre(n)
        features[f"f_{n}({col})"] = poly(y)

    features = pd.DataFrame(features, index=Y.index)
    return features