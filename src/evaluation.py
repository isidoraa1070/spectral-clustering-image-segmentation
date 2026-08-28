from scipy.io import loadmat
from skimage import io
from skimage.segmentation import slic, mark_boundaries
from image_utils import build_image_similarity_matrix, compute_superpixel_features
from segmentation import baseline_kmeans_segmentation
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from spectral import spectral_clustering_from_similarity

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

def evaluate_bsds_images(
    bsds_image_names,
    k_values,
    images_dir='../data/bsds500/images',
    ground_truth_dir='../data/bsds500/ground_truth'
):
    """
    Evaluates spectral clustering and the baseline K-means method on multiple
    BSDS500 images for different numbers of clusters.

    Parameters
    ----------
    bsds_image_names : list of str
        Names of the BSDS500 images to evaluate.
    k_values : list of int
        Numbers of clusters to test.
    images_dir : str
        Path to the directory containing BSDS500 images.
    ground_truth_dir : str
        Path to the directory containing BSDS500 ground-truth files.

    Returns
    -------
    all_results : dict
        Detailed segmentation results for each value of k.
    ari_table : pandas.DataFrame
        ARI scores for spectral clustering and the baseline method.
    """
    all_results = {}
    table_rows = []

    for k in k_values:
        results = []

        for name in bsds_image_names:
            image_path = f'{images_dir}/{name}.jpg'
            gt_path = f'{ground_truth_dir}/{name}.mat'

            result = evaluate_segmentation(image_path, gt_path, k=k)
            results.append((name, result))

            table_rows.append({
                'Slika': name,
                'k': k,
                'ARI - Spektralno': result['ari_spectral'],
                'ARI - Baseline': result['ari_baseline']
            })

        all_results[k] = results

    ari_table = pd.DataFrame(table_rows)

    return all_results, ari_table