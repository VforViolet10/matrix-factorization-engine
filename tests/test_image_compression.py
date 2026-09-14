import numpy as np
import pytest
from PIL import Image

from applications.image_compression import (
    SVDImageCompressor,
)


def create_test_image():
    """
    Create a small synthetic grayscale image.
    """

    return Image.fromarray(
        np.array(
            [
                [10, 20, 30, 40],
                [20, 30, 40, 50],
                [30, 40, 50, 60],
                [40, 50, 60, 70],
            ],
            dtype=np.uint8
        )
    )


def test_compression_output():

    image = create_test_image()

    compressor = SVDImageCompressor(
        n_components=2
    )

    compressed = compressor.compress(
        image
    )

    assert isinstance(
        compressed,
        Image.Image
    )

    assert compressed.size == image.size


def test_compression_ratio():

    image = create_test_image()

    compressor = SVDImageCompressor(
        n_components=1
    )

    ratio = compressor.compression_ratio(
        image
    )

    assert ratio > 1.0


def test_evaluation_metrics():

    image = create_test_image()

    compressor = SVDImageCompressor(
        n_components=2
    )

    compressed = compressor.compress(
        image
    )

    metrics = compressor.evaluate(
        image,
        compressed
    )

    assert (
        "reconstruction_error"
        in metrics
    )

    assert (
        "relative_reconstruction_error"
        in metrics
    )

    assert "rmse" in metrics

    assert metrics[
        "reconstruction_error"
    ] >= 0

    assert metrics[
        "rmse"
    ] >= 0


def test_invalid_components():

    with pytest.raises(ValueError):

        SVDImageCompressor(
            n_components=0
        )


def test_save_image(tmp_path):

    image = create_test_image()

    compressor = SVDImageCompressor(
        n_components=2
    )

    compressed = compressor.compress(
        image
    )

    output_path = (
        tmp_path / "compressed.png"
    )

    compressor.save(
        compressed,
        str(output_path)
    )

    assert output_path.exists()
