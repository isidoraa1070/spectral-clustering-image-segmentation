import numpy as np
from spectral import build_similarity_matrix


def compute_superpixel_features(image, segments):
    """
    Computes the mean color and centroid position for each superpixel.

    Parameters
    ----------
    image : np.ndarray, shape (height, width, channels)
        Original image (RGB).
    segments : np.ndarray, shape (height, width)
        Superpixel labels, as returned by skimage.segmentation.slic.

    Returns
    -------
    colors : np.ndarray, shape (n_segments, channels)
        Mean color (e.g. RGB) for each superpixel.
    positions : np.ndarray, shape (n_segments, 2)
        Centroid (row, col) position for each superpixel.
    """
    n_segments = len(np.unique(segments))
    n_channels = image.shape[2]

    colors = np.zeros((n_segments, n_channels))
    positions = np.zeros((n_segments, 2))

    for label in range(n_segments):
        rows, cols = np.where(segments == label)

        colors[label] = np.mean(image[rows, cols], axis=0)
        positions[label] = np.mean(np.column_stack((rows, cols)), axis=0)

    return colors, positions


def build_image_similarity_matrix(colors, positions, sigma_color=1.0, sigma_position=1.0):
    """
    Builds a similarity matrix for superpixels, combining color and position.

    Parameters
    ----------
    colors : np.ndarray, shape (n_segments, n_channels)
        Mean color for each superpixel (from compute_superpixel_features).
    positions : np.ndarray, shape (n_segments, 2)
        Centroid position for each superpixel.
    sigma_color : float
        Bandwidth for the color kernel.
    sigma_position : float
        Bandwidth for the position kernel.

    Returns
    -------
    W : np.ndarray, shape (n_segments, n_segments)
        Combined similarity matrix.
    """

    W_color = build_similarity_matrix(colors, sigma=sigma_color)


    W_position = build_similarity_matrix(positions, sigma=sigma_position)


    W = W_color * W_position

    return W
