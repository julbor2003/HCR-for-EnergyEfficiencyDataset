import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L

class RescaledLegendre:
    """
    Orthonormal rescaled Legendre polynomial on [0,1]:
    f_n(x) = sqrt(2n+1) * Ln(2x - 1)
    where Ln is the Legendre polynomial on [-1,1].
    """

    def __init__(self, degree: int):
        self.degree = degree
        self._Ln = L.Legendre.basis(degree)
        self.norm = np.sqrt(2 * degree + 1)

    def __call__(self, x):
        return self.norm * self._Ln(2*x-1)

    def __repr__(self):
        return f"RescaledLegendre(degree={self.degree})"
    
def plot_rescaled_legendre(
    polynomials,
    x=None,
    domain=(0, 1),
    labels=None,
    n_points=500,
    title="Rescaled Legendre polynomials",
    grid = False
    ):

    polynomials = list(polynomials)
    if len(polynomials) == 0:
        raise ValueError("No polynomials given.")
    if x is None:
        x = np.linspace(domain[0], domain[1], n_points)
    if labels is None:
        labels = [f"deg {p.degree}" for p in polynomials]

    fig, ax = plt.subplots()
    for poly, label in zip(polynomials, labels):
        ax.plot(x, poly(x), label=label)

    ax.set_xlabel("x")
    ax.set_xlim(0, 1)
    ax.set_ylabel("P(x)")
    ax.set_title(title)
    ax.grid(grid)
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)

    ax.legend(
        loc="lower right",
        ncol=2,
        labelspacing=0.2,
        fontsize=10,
        frameon=False
        )

    plt.show()  