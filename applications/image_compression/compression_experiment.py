import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from examples.image_compression_example import (
    create_sample_image,
)

from .image_compressor import (
    SVDImageCompressor,
)


COMPONENTS = [
    5,
    10,
    20,
    40,
    80,
]

RESULTS_DIR = (
    "applications/image_compression/results"
)

CSV_PATH = os.path.join(
    RESULTS_DIR,
    "compression_results.csv"
)


def run_experiment():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    image = create_sample_image()

    results = []

    print("=" * 60)
    print("SVD IMAGE COMPRESSION EXPERIMENT")
    print("=" * 60)

    for components in COMPONENTS:

        print(
            f"\nTesting k = {components}"
        )

        compressor = SVDImageCompressor(
            n_components=components
        )

        compressed = compressor.compress(
            image
        )

        metrics = compressor.evaluate(
            image,
            compressed
        )

        ratio = compressor.compression_ratio(
            image
        )

        result = {
            "components": components,
            "compression_ratio": ratio,
            "reconstruction_error": (
                metrics[
                    "reconstruction_error"
                ]
            ),
            "relative_reconstruction_error": (
                metrics[
                    "relative_reconstruction_error"
                ]
            ),
            "rmse": metrics["rmse"],
        }

        results.append(
            result
        )

        print(
            f"  Compression Ratio: "
            f"{ratio:.2f}x"
        )

        print(
            f"  Relative Error: "
            f"{result['relative_reconstruction_error']:.6f}"
        )

        print(
            f"  RMSE: "
            f"{result['rmse']:.4f}"
        )

    return results


def save_results(results):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    fieldnames = [
        "components",
        "compression_ratio",
        "reconstruction_error",
        "relative_reconstruction_error",
        "rmse",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to: {CSV_PATH}"
    )


def generate_plots(results):

    components = [
        result["components"]
        for result in results
    ]

    compression_ratios = [
        result["compression_ratio"]
        for result in results
    ]

    relative_errors = [
        result[
            "relative_reconstruction_error"
        ]
        for result in results
    ]

    rmses = [
        result["rmse"]
        for result in results
    ]

    # Compression ratio
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        components,
        compression_ratios,
        marker="o"
    )

    plt.xlabel(
        "Number of SVD Components (k)"
    )

    plt.ylabel(
        "Compression Ratio"
    )

    plt.title(
        "Compression Ratio vs SVD Components"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "compression_ratio.png"
        ),
        dpi=300
    )

    plt.close()

    # Reconstruction error
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        components,
        relative_errors,
        marker="o"
    )

    plt.xlabel(
        "Number of SVD Components (k)"
    )

    plt.ylabel(
        "Relative Reconstruction Error"
    )

    plt.title(
        "Reconstruction Error vs SVD Components"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "reconstruction_error.png"
        ),
        dpi=300
    )

    plt.close()

    # RMSE
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        components,
        rmses,
        marker="o"
    )

    plt.xlabel(
        "Number of SVD Components (k)"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        "RMSE vs SVD Components"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "rmse.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "Plots generated successfully."
    )


def main():

    results = run_experiment()

    save_results(
        results
    )

    generate_plots(
        results
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
