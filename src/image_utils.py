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


def scaled_sigma_position(image_shape, position_ratio=0.1):
    """
    Converts a relative spatial scale into sigma_position in pixels.

    sigma_position is defined as a fraction of the image diagonal, so the
    spatial kernel has comparable meaning for images of different sizes.

    Parameters
    ----------
    image_shape : tuple
        Image shape or any tuple whose first two entries are height and width.
    position_ratio : float, default=0.1
        Fraction of the image diagonal used as sigma_position.

    Returns
    -------
    float
        Sigma for the position kernel, scaled according to image size.
    """
    if position_ratio <= 0:
        raise ValueError('position_ratio must be positive.')

    height, width = image_shape[:2]
    diagonal = np.hypot(height, width)
    return position_ratio * diagonal


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
        Spatial bandwidth in pixels. In the image pipeline it is obtained from
        ``scaled_sigma_position`` so it scales with image size.

    Returns
    -------
    W : np.ndarray, shape (n_segments, n_segments)
        Combined similarity matrix.
    """
    W_color = build_similarity_matrix(colors, sigma=sigma_color)
    W_position = build_similarity_matrix(positions, sigma=sigma_position)
    return W_color * W_position
