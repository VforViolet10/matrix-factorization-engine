import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(A, title="Matrix"):
    """
    Display a matrix as a heatmap.
    """

    A = np.asarray(A)

    plt.figure(figsize=(7, 5))
    plt.imshow(A, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()
