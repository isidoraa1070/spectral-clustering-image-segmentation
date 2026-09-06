import numpy as np
import pandas as pd
from scipy.io import loadmat
from skimage import io
from skimage.segmentation import slic
from sklearn.metrics import adjusted_rand_score

from image_utils import (
    build_image_similarity_matrix,
    compute_superpixel_features,
    scaled_sigma_position,
)
from segmentation import baseline_kmeans_segmentation
from spectral import spectral_clustering_from_similarity


def load_bsds_ground_truth(mat_path):
    """
    Loads all BSDS500 ground-truth segmentations available for an image.

    Parameters
    ----------
    mat_path : str
        Path to the BSDS500 ground-truth .mat file.

    Returns
    -------
    list of np.ndarray
        One segmentation label map per human annotator.
    """
    mat = loadmat(mat_path)
    ground_truth_entries = mat['groundTruth'][0]
    return [entry[0, 0]['Segmentation'] for entry in ground_truth_entries]


def _ari_distribution(segmentation, ground_truths):
    """Returns ARI against each human ground-truth segmentation."""
    return np.array([
        adjusted_rand_score(gt.ravel(), segmentation.ravel())
        for gt in ground_truths
    ], dtype=float)


def evaluate_segmentation(
    image_path,
    gt_path,
    k,
    n_segments=200,
    sigma_color=20,
    position_ratio=0.1,
):
    """
    Evaluates spectral clustering and baseline K-means on one BSDS500 image.

    ARI is computed separately against every available human annotation. The
    full distribution is retained, while mean and standard deviation are used
    as compact summary statistics.

    Parameters
    ----------
    image_path : str
        Path to the BSDS500 image.
    gt_path : str
        Path to the corresponding ground-truth .mat file.
    k : int
        Number of clusters to use for both methods.
    n_segments : int, default=200
        Number of SLIC superpixels.
    sigma_color : float, default=20
        Bandwidth for the color kernel.
    position_ratio : float, default=0.1
        sigma_position as a fraction of the image diagonal.

    Returns
    -------
    dict
        Segmentations, all ground truths, ARI distributions and their
        mean/std summaries for both methods.
    """
    image = io.imread(image_path)
    segments = slic(image, n_segments=n_segments, compactness=10, start_label=0)
    colors, positions = compute_superpixel_features(image, segments)

    sigma_position = scaled_sigma_position(image.shape, position_ratio)
    W = build_image_similarity_matrix(
        colors,
        positions,
        sigma_color=sigma_color,
        sigma_position=sigma_position,
    )

    labels_spectral = spectral_clustering_from_similarity(W, k=k, normalized=True)
    labels_baseline = baseline_kmeans_segmentation(colors, positions, k=k)

    segmented_spectral = labels_spectral[segments]
    segmented_baseline = labels_baseline[segments]

    ground_truths = load_bsds_ground_truth(gt_path)
    ari_spectral = _ari_distribution(segmented_spectral, ground_truths)
    ari_baseline = _ari_distribution(segmented_baseline, ground_truths)

    return {
        'image': image,
        'ground_truths': ground_truths,
        # Kept as a representative map for compact visualizations only.
        'ground_truth': ground_truths[0],
        'segmented_spectral': segmented_spectral,
        'segmented_baseline': segmented_baseline,
        'sigma_position': sigma_position,
        'position_ratio': position_ratio,
        'ari_spectral': ari_spectral,
        'ari_baseline': ari_baseline,
        'ari_spectral_mean': float(np.mean(ari_spectral)),
        'ari_spectral_std': float(np.std(ari_spectral)),
        'ari_baseline_mean': float(np.mean(ari_baseline)),
        'ari_baseline_std': float(np.std(ari_baseline)),
    }


def evaluate_bsds_images(
    bsds_image_names,
    k_values,
    images_dir='../data/bsds500/images',
    ground_truth_dir='../data/bsds500/ground_truth',
    n_segments=200,
    sigma_color=20,
    position_ratio=0.1,
):
    """
    Evaluates both methods on multiple BSDS500 images and k values.

    Returns a detailed result dictionary and a compact table containing mean
    and standard deviation of ARI across all annotators for each image.
    """
    all_results = {}
    table_rows = []

    for k in k_values:
        results = []

        for name in bsds_image_names:
            image_path = f'{images_dir}/{name}.jpg'
            gt_path = f'{ground_truth_dir}/{name}.mat'

            result = evaluate_segmentation(
                image_path,
                gt_path,
                k=k,
                n_segments=n_segments,
                sigma_color=sigma_color,
                position_ratio=position_ratio,
            )
            results.append((name, result))

            table_rows.append({
                'Slika': name,
                'k': k,
                'Broj anotatora': len(result['ground_truths']),
                'ARI - Spektralno (mean)': result['ari_spectral_mean'],
                'ARI - Spektralno (std)': result['ari_spectral_std'],
                'ARI - Baseline (mean)': result['ari_baseline_mean'],
                'ARI - Baseline (std)': result['ari_baseline_std'],
            })

        all_results[k] = results

    return all_results, pd.DataFrame(table_rows)
