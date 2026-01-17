import numpy as np
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_weights(model, V):
    coef = pd.Series(model.coef_, index=V.columns)
    return coef

def group_coeffs(coeffs):
    pattern = re.compile(r"L_(\d+)\((.+)\)")
    grouped = defaultdict(dict)

    for name, value in coeffs.items():
        match = pattern.fullmatch(name)
        if not match: continue

        deg = int(match.group(1))
        var = match.group(2)
        grouped[deg][var] = float(value)

    return grouped

def coeffs_mean(coeffs_list):
    coeffs = defaultdict(lambda: defaultdict(dict))
    n_folds = len(coeffs_list)

    target_degs = coeffs_list[0].keys()
    feature_degs = coeffs_list[0][1].keys()
    features = coeffs_list[0][1][1].keys()

    for target_deg in target_degs:
        for feature_deg in feature_degs:
            for feature in features:
                values = np.array([
                    coeffs_list[fold][target_deg][feature_deg][feature]
                    for fold in range(n_folds)
                ])
                coeffs[target_deg][feature_deg][feature] = values.mean()

    return coeffs

def print_coeffs(model=None, V=None, coeffs=None):
    if coeffs==None:
        coeffs = analyze_weights(model, V)
        coeffs = group_coeffs(coeffs)
    for deg, terms in coeffs.items():
        sorted_terms = sorted(
            terms.items(), 
            key=lambda x: abs(x[1]),
            reverse=True
            )  
        
        print(f"Degree {deg}:")
        for var, value in sorted_terms:
            print(f"  {var}: {value:.4f}")  
        print()

def plot_coeffs(model=None, V=None, coeffs=None, target_deg=1, base_deg=1, colors=None):
    if coeffs==None:
        coeffs = analyze_weights(model, V)
        coeffs = group_coeffs(coeffs)
    
    degs = sorted(coeffs.keys())
    n_degs = len(degs)
    if colors is None or len(colors)<len(degs):
        colors = plt.cm.viridis_r([i/max(len(degs)-1, 1) for i in range(len(degs))])
    plt.rcParams["font.family"] = "Times New Roman"

    _, ax = plt.subplots(figsize=(6, 4))

    base_terms = coeffs[base_deg]
    base_terms_sorted = sorted(
        base_terms.items(), 
        key=lambda x: abs(x[1]),
        reverse=True
        )
    order = [var for var, _ in base_terms_sorted]

    # for deg, terms in coeffs.items():
    #     color = colors[deg-1]
    #     terms = coeffs[deg]

    #     ys = [terms.get(var, 0.0) for var in order]
    #     ax.scatter(order, ys,
    #                color=color, alpha=0.6,
    #                label=f"deg. {deg}")

    x = np.arange(len(order))
    width = 0.8 / n_degs

    for i, deg in enumerate(degs):
        terms = coeffs[deg]
        ys = np.array([terms.get(var, 0.0) for var in order])

        ax.bar(
            x + i * width - 0.4 + width / 2,
            ys,
            width=width,
            color=colors[i],
            alpha=0.8,
            label=f"deg. {deg}"
        )

    for i in range(len(order) - 1):
        ax.axvline(
            i + 0.5,
            color="black",
            lw=0.3,
            alpha=0.4,
            zorder=0
        )
        
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order)

    ax.set_xlabel(f"Coefficient value for target degree={target_deg}")
    ax.legend(
        fontsize=7, ncol=n_degs, 
        loc="lower center",
        frameon=False
        )
    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha = "right",
        fontsize=7
    )
    plt.tight_layout()
    plt.show()