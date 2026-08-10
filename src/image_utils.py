import numpy as np
from sklearn.cluster import KMeans
from spectral import build_similarity_matrix, compute_laplacian
from scipy.linalg import eigh
from matplotlib import pyplot as plt
from skimage import io
from skimage.segmentation import slic, mark_boundaries

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

def spectral_clustering_from_similarity(W, k, normalized=True):
    """
    Performs spectral clustering given a precomputed similarity matrix.

    Parameters
    ----------
    W : np.ndarray, shape (n_samples, n_samples)
        Precomputed similarity matrix.
    k : int
        Number of clusters.
    normalized : bool
        Whether to use the normalized Laplacian.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignment for each point.
    """
    L = compute_laplacian(W, normalized=normalized)
    _, eigenvectors = eigh(L)
    U = eigenvectors[:, :k]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(U)
    labels = kmeans.labels_

    return labels

from skimage.transform import resize

def downsample_image(image, max_dimension=300):
    """
    Downsamples an image so its largest dimension does not exceed max_dimension,
    preserving aspect ratio. Speeds up SLIC and the overall pipeline for large images.

    Parameters
    ----------
    image : np.ndarray, shape (height, width, channels)
        Original image.
    max_dimension : int
        Maximum allowed size (in pixels) for the largest dimension.

    Returns
    -------
    np.ndarray
        Downsampled image.
    """
    h, w = image.shape[:2]
    scale = max_dimension / max(h, w)

    if scale >= 1:
        return image 

    new_h, new_w = int(h * scale), int(w * scale)
    return resize(image, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(image.dtype)

def analyze_image(image_path, n_segments=200, compactness=10, k_values=[2, 3, 4, 5]):
    """
    Runs the full segmentation pipeline for a single image: loads it, applies SLIC,
    computes superpixel features, and displays a k-value comparison grid.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    n_segments : int
        Target number of SLIC superpixels.
    compactness : float
        SLIC compactness parameter.
    k_values : list
        Values of k to compare in the grid.

    Returns
    -------
    image, segments, colors, positions : the intermediate results, in case further
        experiments (e.g. varying sigma_position) are needed for this image.
    """
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)
    colors, positions = compute_superpixel_features(image, segments)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Originalna slika')
    axes[0].axis('off')
    axes[1].imshow(mark_boundaries(image, segments))
    axes[1].set_title(f'SLIC (n_segments={n_segments})')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    run_segmentation_grid(segments, colors, positions, param_name='k', param_values=k_values)

    return image, segments, colors, positions

def run_segmentation_grid(segments, colors, positions, param_name, param_values,
                           fixed_k=4, fixed_sigma_color=20, fixed_sigma_position=50):
    """
    Runs spectral clustering segmentation over a range of values for one parameter,
    keeping the others fixed, and plots the results side by side.

    Parameters
    ----------
    segments : np.ndarray
        SLIC superpixel labels, shape (height, width).
    colors, positions : np.ndarray
        Superpixel features, from compute_superpixel_features.
    param_name : str
        Which parameter to vary: 'k', 'sigma_color', or 'sigma_position'.
    param_values : list
        Values to test for that parameter.
    fixed_k, fixed_sigma_color, fixed_sigma_position :
        Default values used for the parameters that are NOT being varied.
    """
    fig, axes = plt.subplots(1, len(param_values), figsize=(4 * len(param_values), 4))

    for ax, value in zip(axes, param_values):
        k = value if param_name == 'k' else fixed_k
        sigma_color = value if param_name == 'sigma_color' else fixed_sigma_color
        sigma_position = value if param_name == 'sigma_position' else fixed_sigma_position

        W = build_image_similarity_matrix(colors, positions, sigma_color, sigma_position)
        labels = spectral_clustering_from_similarity(W, k=k, normalized=True)
        segmented = labels[segments]

        ax.imshow(segmented, cmap='viridis')
        ax.set_title(f'{param_name} = {value}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

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

from scipy.io import loadmat

def load_bsds_ground_truth(mat_path, annotator_index=0):
    """
    Loads a BSDS500 ground-truth .mat file and returns the segmentation
    label map from one annotator.

    Parameters
    ----------
    mat_path : str
        Path to the .mat ground-truth file.
    annotator_index : int
        Which annotator's segmentation to use (BSDS500 provides several
        per image, since human segmentation is inherently subjective).

    Returns
    -------
    np.ndarray, shape (height, width)
        Ground-truth segment labels.
    """
    mat = loadmat(mat_path)
    ground_truth = mat['groundTruth'][0, annotator_index][0, 0]['Segmentation']
    return ground_truth

from sklearn.metrics import adjusted_rand_score

def evaluate_segmentation(image_path, gt_path, k, n_segments=200, sigma_color=20, sigma_position=50):
    """
    Runs both spectral clustering and the baseline method on a BSDS500 image,
    and compares each against the ground-truth segmentation using ARI.

    Parameters
    ----------
    image_path : str
        Path to the BSDS500 image.
    gt_path : str
        Path to the corresponding ground-truth .mat file.
    k : int
        Number of clusters to use for both methods.
    n_segments, sigma_color, sigma_position :
        Pipeline parameters (see 03_image_segmentation.ipynb for their meaning).

    Returns
    -------
    dict
        ARI scores for both methods, plus the intermediate results for plotting.
    """
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    colors, positions = compute_superpixel_features(image, segments)

    W = build_image_similarity_matrix(colors, positions, sigma_color, sigma_position)
    labels_spectral = spectral_clustering_from_similarity(W, k=k, normalized=True)
    labels_baseline = baseline_kmeans_segmentation(colors, positions, k=k)

    segmented_spectral = labels_spectral[segments]
    segmented_baseline = labels_baseline[segments]

    ground_truth = load_bsds_ground_truth(gt_path)

    ari_spectral = adjusted_rand_score(ground_truth.flatten(), segmented_spectral.flatten())
    ari_baseline = adjusted_rand_score(ground_truth.flatten(), segmented_baseline.flatten())

    return {
        'image': image,
        'ground_truth': ground_truth,
        'segmented_spectral': segmented_spectral,
        'segmented_baseline': segmented_baseline,
        'ari_spectral': ari_spectral,
        'ari_baseline': ari_baseline,
    }