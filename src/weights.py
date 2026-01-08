import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_weights(model, V):
    coef = pd.Series(model.coef_, index=V.columns)
    return coef

def group_coeffs(coefs):
    pattern = re.compile(r"L_(\d+)\((.+)\)")
    grouped = defaultdict(dict)

    for name, value in coefs.items():
        match = pattern.fullmatch(name)
        if not match: continue

        deg = int(match.group(1))
        var = match.group(2)
        grouped[deg][var] = float(value)

    return grouped

def print_coeffs(model=None, V=None, coefs=None):
    if coefs==None:
        coefs = analyze_weights(model, V)
        coefs = group_coeffs(coefs)
    for deg, terms in coefs.items():
        sorted_terms = sorted(
            terms.items(), 
            key=lambda x: abs(x[1]),
            reverse=True
            )  
        
        print(f"Degree {deg}:")
        for var, value in sorted_terms:
            print(f"  {var}: {value:.4f}")  
        print()

def plot_coeffs(model=None, V=None, coefs=None, target_deg=1, base_deg=1, colors=None):
    if coefs==None:
        coefs = analyze_weights(model, V)
        coefs = group_coeffs(coefs)
    
    degs = sorted(coefs.keys())
    if colors is None or len(colors)<len(degs):
        colors = plt.cm.viridis_r([i/max(len(degs)-1, 1) for i in range(len(degs))])
    plt.rcParams["font.family"] = "Times New Roman"

    if V!=None:
        length = len(V.columns)/len(degs)
    elif coefs!=None:
        length = len(coefs[base_deg])
    else:
        raise ValueError("Either V or coefs must be provided.")

    _, ax = plt.subplots(figsize=(6, 0.4*length))

    base_terms = coefs[base_deg]
    base_terms_sorted = sorted(
        base_terms.items(), 
        key=lambda x: abs(x[1]),
        reverse=True
        )
    order = [var for var, _ in base_terms_sorted]

    for deg, terms in coefs.items():
        color = colors[deg-1]
        terms = coefs[deg]

        ys = [terms.get(var, 0.0) for var in order]
        ax.scatter(order, ys,
                   color=color, alpha=0.6,
                   label=f"feature deg. {deg}")

    ax.set_xlabel(f"Coefficient value for target degree={target_deg}")
    ax.legend(
        fontsize=7, ncol=len(degs), 
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