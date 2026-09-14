import numpy as np
from PIL import Image

from applications.image_compression import (
    SVDImageCompressor,
)


def create_sample_image():
    """
    Create a synthetic RGB image for demonstration.
    """

    rows, cols = 200, 300

    x = np.linspace(
        0,
        1,
        cols
    )

    y = np.linspace(
        0,
        1,
        rows
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    red = (
        255 * X
    )

    green = (
        255 * Y
    )

    blue = (
        255 * (1 - X * Y)
    )

    image = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=2
    )

    return Image.fromarray(
        image.astype(np.uint8)
    )


def main():

    print("=" * 60)
    print("SVD IMAGE COMPRESSION")
    print("=" * 60)

    original = create_sample_image()

    components = 20

    compressor = SVDImageCompressor(
        n_components=components
    )

    compressed = compressor.compress(
        original
    )

    metrics = compressor.evaluate(
        original,
        compressed
    )

    ratio = compressor.compression_ratio(
        original
    )

    compressor.save(
        original,
        "applications/image_compression/original.png"
    )

    compressor.save(
        compressed,
        "applications/image_compression/compressed.png"
    )

    print(
        f"\nImage shape: {np.asarray(original).shape}"
    )

    print(
        f"Components: {components}"
    )

    print(
        f"Compression ratio: {ratio:.2f}x"
    )

    print(
        f"Reconstruction error: "
        f"{metrics['reconstruction_error']:.4f}"
    )

    print(
        f"Relative reconstruction error: "
        f"{metrics['relative_reconstruction_error']:.6f}"
    )

    print(
        f"RMSE: "
        f"{metrics['rmse']:.4f}"
    )

    print(
        "\nImages saved to:"
    )

    print(
        "  applications/image_compression/original.png"
    )

    print(
        "  applications/image_compression/compressed.png"
    )


if __name__ == "__main__":
    main()
