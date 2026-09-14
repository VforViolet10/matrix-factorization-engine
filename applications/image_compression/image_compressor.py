import os

import numpy as np
from PIL import Image

from matrix_factorization.decompositions import SVD
from matrix_factorization.metrics import (
    reconstruction_error,
    relative_reconstruction_error,
    rmse,
)


class SVDImageCompressor:
    """
    Image compression using Singular Value Decomposition.

    The image is represented as matrix channels and
    approximated using the top-k singular components.
    """

    def __init__(self, n_components):
        if n_components <= 0:
            raise ValueError(
                "n_components must be greater than zero."
            )

        self.n_components = n_components
        self.channels = []
        self.original_shape = None

    def _prepare_image(self, image):
        """
        Convert an image into numerical matrix channels.
        """

        image_array = np.asarray(
            image,
            dtype=float
        )

        if image_array.ndim == 2:
            return [
                image_array
            ]

        if image_array.ndim == 3:
            return [
                image_array[:, :, channel]
                for channel in range(
                    image_array.shape[2]
                )
            ]

        raise ValueError(
            "Image must be grayscale or RGB/RGBA."
        )

    def compress(self, image):
        """
        Compress an image using truncated SVD.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image.

        Returns
        -------
        PIL.Image.Image
            Reconstructed compressed image.
        """

        image_array = np.asarray(
            image,
            dtype=float
        )

        self.original_shape = image_array.shape

        channels = self._prepare_image(
            image
        )

        reconstructed_channels = []

        for channel in channels:

            max_components = min(
                channel.shape
            )

            components = min(
                self.n_components,
                max_components
            )

            svd = SVD(
                n_components=components
            )

            svd.fit(
                channel
            )

            reconstructed = svd.reconstruct()

            reconstructed_channels.append(
                reconstructed
            )

        if image_array.ndim == 2:
            reconstructed_image = (
                reconstructed_channels[0]
            )
        else:
            reconstructed_image = np.stack(
                reconstructed_channels,
                axis=2
            )

        reconstructed_image = np.clip(
            reconstructed_image,
            0,
            255
        )

        reconstructed_image = (
            reconstructed_image.astype(
                np.uint8
            )
        )

        return Image.fromarray(
            reconstructed_image
        )

    def compression_ratio(self, image):
        """
        Estimate the storage compression ratio.

        A rank-k representation stores:

            U: m × k
            Sigma: k
            Vt: k × n

        compared with the original m × n matrix.
        """

        image_array = np.asarray(
            image
        )

        if image_array.ndim == 2:
            channels = 1
            rows, cols = image_array.shape

        elif image_array.ndim == 3:
            rows, cols, channels = (
                image_array.shape
            )

        else:
            raise ValueError(
                "Image must be grayscale or RGB/RGBA."
            )

        components = min(
            self.n_components,
            rows,
            cols
        )

        original_values = (
            rows
            * cols
            * channels
        )

        compressed_values = (
            channels
            * (
                rows * components
                + components
                + components * cols
            )
        )

        if compressed_values == 0:
            raise ValueError(
                "Compressed representation is empty."
            )

        return (
            original_values
            / compressed_values
        )

    def evaluate(self, original, reconstructed):
        """
        Calculate reconstruction metrics for an image.

        The general matrix metrics in this project operate
        on 2D matrices. RGB/RGBA image data is therefore
        reshaped into a 2D pixel-by-channel matrix before
        evaluation.
        """

        original_array = np.asarray(
            original,
            dtype=float
        )

        reconstructed_array = np.asarray(
            reconstructed,
            dtype=float
        )

        if (
            original_array.shape
            != reconstructed_array.shape
        ):
            raise ValueError(
                "Original and reconstructed images "
                "must have the same shape."
            )

        if original_array.ndim == 3:

            original_array = (
                original_array.reshape(
                    -1,
                    original_array.shape[-1]
                )
            )

            reconstructed_array = (
                reconstructed_array.reshape(
                    -1,
                    reconstructed_array.shape[-1]
                )
            )

        elif original_array.ndim != 2:
            raise ValueError(
                "Images must be grayscale or RGB/RGBA."
            )

        return {
            "reconstruction_error": (
                reconstruction_error(
                    original_array,
                    reconstructed_array
                )
            ),
            "relative_reconstruction_error": (
                relative_reconstruction_error(
                    original_array,
                    reconstructed_array
                )
            ),
            "rmse": rmse(
                original_array,
                reconstructed_array
            ),
        }

    def save(self, image, output_path):
        """
        Save a compressed image.
        """

        directory = os.path.dirname(
            output_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        image.save(
            output_path
        ) 
