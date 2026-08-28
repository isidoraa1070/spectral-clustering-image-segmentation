from skimage import io
from skimage.segmentation import slic, mark_boundaries
from matplotlib import pyplot as plt
from image_utils import build_image_similarity_matrix, compute_superpixel_features, spectral_clustering_from_similarity
import numpy as np
from sklearn.cluster import KMeans

def baseline_kmeans_segmentation(colors, positions, k, position_weight=1.0):
    """
    Segments superpixels using standard k-means directly on their features,
    without building a similarity graph or Laplacian (i.e. without the
    spectral clustering machinery).

    Parameters
    ----------
    colors : np.ndarray, shape (n_segments, n_channels)
        Mean color for each superpixel.
    positions : np.ndarray, shape (n_segments, 2)
        Centroid position for each superpixel.
    k : int
        Number of clusters.
    position_weight : float
        Relative weight given to position vs color when the two feature
        types are concatenated (they live on very different numeric scales,
        so a naive concatenation would let position dominate).

    Returns
    -------
    labels : np.ndarray, shape (n_segments,)
        Cluster assignment for each superpixel.
    """
    colors_norm = (colors - colors.mean(axis=0)) / (colors.std(axis=0) + 1e-8)
    positions_norm = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)

    features = np.hstack([colors_norm, position_weight * positions_norm])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    return labels